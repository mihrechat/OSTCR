import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
import torch.nn.functional as F
from .ST_block import SpatioTemporalBlock
from .node_attribute_encoder import NodeAttributeEncoder
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import dropout_edge
from models.multiEncoder.cross_encoder import QuestionGuidedPooling, QuestionGuidedMotionPooling, CrossModalAttentionFusion
from .moe_fusion import QuestionGatedCausalFusion, CausalCrossAttentionFusion, CausalGatedFusion, MoELinearFusion
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank, ConceptNetExpectedPriorBank
from torch_scatter import scatter_mean


# ==================================================================
# 1. Triplet Prior Bank (M) — Edge-level Causal Mechanism
# ==================================================================
class ConceptNetTripletBank(nn.Module):
    """
    O(1) Lookup Table for Precomputed ConceptNet Triplets.
    Outputs the candidate causal mechanisms (M) for Cross-Attention.
    """
    def __init__(
        self,
        triplet_pt_path="M_triplets.pt",
        max_triplets=10,
        num_classes=80,
        text_dim=None
    ):
        super().__init__()
        self.max_triplets = max_triplets
        self.num_classes = num_classes

        # Load offline extracted dictionary
        triplet_dict = torch.load(triplet_pt_path, map_location="cpu")
        if len(triplet_dict) == 0:
            if text_dim is None:
                raise ValueError("text_dim is required when triplet dictionary is empty.")
            dim = text_dim
        else:
            sample_tensor = next(iter(triplet_dict.values()))
            dim = sample_tensor.shape[-1]

        self.lookup_dim = dim
        self.output_dim = text_dim if text_dim is not None else dim
        self.proj = None
        if text_dim is not None and text_dim != dim:
            self.proj = nn.Linear(dim, text_dim, bias=False)

        # Initialize dense lookup buffer: (Num_Classes, Num_Classes, Max_Triplets, Dim)
        lookup_table = torch.zeros((num_classes, num_classes, max_triplets, dim))

        for (src_idx, dst_idx), embs in triplet_dict.items():
            num_avail = min(len(embs), max_triplets)
            lookup_table[src_idx, dst_idx, :num_avail] = embs[:num_avail]

        # Registered as a buffer so it moves to GPU but does not receive gradients
        self.register_buffer("lookup_table", lookup_table)

    def forward(self, src_cls_ids: torch.Tensor, dst_cls_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_cls_ids: (E,) tensor of source node classes
            dst_cls_ids: (E,) tensor of target node classes
        Returns:
            (E, max_triplets, dim) tensor of retrieved textual affordances
        """
        out = self.lookup_table[src_cls_ids, dst_cls_ids]
        if self.proj is not None:
            out = self.proj(out)
        return out


# ==================================================================
# 2. Visual Prototype Memory (E_v) — Scene-level Backdoor
# ==================================================================

class VisualPrototypeMemory(nn.Module):
    """
    Video-level visual prototype memory bank representing scene-level priors P(v).
    Updated via EMA during training to track the network's shifting representations.
    """
    def __init__(self, num_prototypes: int = 64, dim: int = 512, momentum: float = 0.999, min_count: int = 10):
        super().__init__()
        self.momentum = momentum
        self.min_count = min_count
        self.K = num_prototypes

        target_norm = (dim ** 0.5)
        init_protos = torch.randn(num_prototypes, dim)
        init_protos = F.normalize(init_protos, dim=-1) * target_norm

        self.register_buffer("prototypes", init_protos)
        self.register_buffer("update_counts", torch.zeros(num_prototypes))

    @torch.no_grad()
    def update(self, node_feat: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int):
        video_reprs = []
        for b in range(num_graphs):
            mask = (batch_idx == b)
            if mask.sum() == 0:
                continue
            v_repr = F.normalize(node_feat[mask].mean(dim=0), dim=-1)
            video_reprs.append(v_repr)

        if not video_reprs:
            return

        video_reprs = torch.stack(video_reprs)        # (B, dim)
        sim = video_reprs @ self.prototypes.T         # (B, K)
        assignments = sim.argmax(dim=-1)              # (B,)

        for k in range(self.K):
            assigned = video_reprs[assignments == k]
            if assigned.shape[0] == 0:
                continue
            new_feat = assigned.mean(dim=0)
            self.prototypes[k] = F.normalize(
                self.momentum * self.prototypes[k] + (1 - self.momentum) * new_feat,
                dim=-1
            )
            self.update_counts[k] += assigned.shape[0]

    def get_expected_visual(self) -> torch.Tensor:
        """ Returns the global expected visual confounder E[v] of shape (1, dim) """
        reliable = self.update_counts >= self.min_count
        if reliable.sum() == 0:
            return self.prototypes.mean(dim=0, keepdim=True)

        valid_protos = self.prototypes[reliable]
        valid_counts = self.update_counts[reliable]

        weights = valid_counts / valid_counts.sum()
        E_v = (valid_protos * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

        return E_v


# ==================================================================
# 3. Robust Causal Deconfounding with Residuals
# ==================================================================

class RobustFiLMDeconfounding(nn.Module):
    """
    Improved FiLM deconfounding with residual connections to preserve original semantics.
    Prevents information collapse by blending original and deconfounded signals.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.blend_weight = nn.Parameter(torch.tensor(0.5))
        self.ln = nn.LayerNorm(dim)

    def forward(self, original: torch.Tensor, confounder: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original: (B, dim) original question embedding
            confounder: (B, dim) linguistic prior (confounder signal)
        Returns:
            (B, dim) deconfounded + residual blended representation
        """
        gamma = torch.sigmoid(self.gamma_mlp(confounder))
        beta = self.beta_mlp(confounder)

        # FiLM transformation with learned residual blend
        deconf = gamma * original + beta

        # Residual connection with learnable blend
        blend = torch.sigmoid(self.blend_weight)
        output = blend * original + (1 - blend) * deconf

        return self.ln(output)


# ==================================================================
# 4. Robust Triplet-Based Mediator Extraction
# ==================================================================

class RobustTripletMediator(nn.Module):
    """
    FIXED: Improved mediator extraction with:
    - Proper fallback for empty edges
    - Numerical stability checks
    - Consistent dimension handling
    """
    def __init__(self, query_dim: int, triplet_dim: int, attn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_q = nn.Sequential(
            nn.Linear(query_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.Dropout(dropout)
        )
        self.W_k = nn.Sequential(
            nn.Linear(triplet_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.Dropout(dropout)
        )
        self.W_v = nn.Sequential(
            nn.Linear(triplet_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.Dropout(dropout)
        )
        self.temperature = nn.Parameter(torch.tensor(1.0 / (attn_dim ** 0.5)))
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(attn_dim)

    def forward(
        self,
        v_graph: torch.Tensor,
        E_T: torch.Tensor,
        edge_batch: torch.Tensor,
        triplet_mask: torch.Tensor,
        B: int
    ) -> torch.Tensor:
        """
        Args:
            v_graph: (B, dim) question-guided graph representation
            E_T: (E, num_triplets, dim) triplet embeddings
            edge_batch: (E,) batch assignment per edge
            triplet_mask: (E, num_triplets) validity mask
            B: batch size
        Returns:
            (B, dim) deconfounded mediator M
        """
        if E_T.numel() == 0:
            return torch.zeros(B, self.W_v[-1].out_features, device=v_graph.device, dtype=v_graph.dtype)

        q_graph = self.W_q(v_graph)
        k_edges = self.W_k(E_T)
        v_edges = self.W_v(E_T)

        q_expanded = q_graph[edge_batch].unsqueeze(1)  # (E, 1, attn_dim)

        # Compute attention with temperature scaling
        attn_logits = (q_expanded * k_edges).sum(dim=-1) * self.temperature  # (E, num_triplets)

        # FIXED: Better masking
        attn_logits = attn_logits.masked_fill(~triplet_mask, -1e4)

        # Softmax
        alpha = torch.softmax(attn_logits, dim=-1)
        alpha = self.dropout(alpha)

        # Weighted aggregation
        m_weighted = v_edges * alpha.unsqueeze(-1)  # (E, num_triplets, attn_dim)
        m_edge = m_weighted.sum(dim=1)  # (E, attn_dim)

        # Use scatter_mean for stability
        M = scatter_mean(m_edge, edge_batch, dim=0, dim_size=B)  # (B, attn_dim)
        M = self.ln(M)

        return M


class STGraphTransformerNet(nn.Module):
    """
    PRODUCTION-READY: Complete causal video QA with all fixes.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.model.dropout
        self.dim = cfg.model.dim
        self.raw_dim = cfg.model.raw_dim
        self.edge_dim = cfg.model.edge_dim
        self.final_drop = cfg.train.final_drop

        self._validate_config(cfg)

        # ── Stage 1: Encoders
        self.node_encoder = NodeAttributeEncoder(
            self.raw_dim, self.dim, cfg.model.num_classes
        )
        self.edge_embed = nn.Embedding(cfg.model.num_preds, self.edge_dim)

        # ── Stage 2: ST-Graph Blocks
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                self.dim, cfg.model.num_heads, self.edge_dim, cfg.model.num_anchors
            )
            for _ in range(cfg.model.num_layers)
        ])
        self.node_feat_norm = nn.LayerNorm(self.dim)

        # ── Stage 3: Question Encoding (supports dynamic input dim)
        self.question_encoder = nn.Sequential(
            nn.LazyLinear(self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

        # ── Stage 4: Linguistic Prior
        self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(
            qtype_pt_path=cfg.model.qtype_pt_path,
            qrole_pt_path=cfg.model.qrole_pt_path,
            concept_dim=cfg.model.concept_dim
        )
        self.confounder_proj = nn.Sequential(
            nn.Linear(cfg.model.concept_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.Dropout(self.dropout)
        )

        # ── Stage 5: Deconfounding
        self.deconfounding = RobustFiLMDeconfounding(self.dim, self.dropout)

        # ── Stage 6: Motion Processing (supports dynamic input dim)
        self.motion_proj = nn.Sequential(
            nn.LazyLinear(self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

        # ── Stage 7: Cross-Modal Fusion
        self.cross_modal_fusion = CrossModalAttentionFusion(
            dim=self.dim,
            num_heads=cfg.model.num_heads,
            dropout=self.dropout
        )

        # ── Stage 8: Causal Memory
        self.visual_memory = VisualPrototypeMemory(
            num_prototypes=cfg.model.num_prototypes,
            dim=self.dim,
            momentum=cfg.model.prototype_momentum
        )

        self.conceptnet_prior = ConceptNetExpectedPriorBank(
            prior_pt_path=cfg.model.prior_pt_path,
            concept_dim=cfg.model.concept_dim
        )

        self.conceptnet_triplets = ConceptNetTripletBank(
            triplet_pt_path=cfg.model.triplet_pt_path,
            max_triplets=cfg.model.max_triplets,
            num_classes=cfg.model.num_classes,
            text_dim=cfg.model.text_dim
        )

        # ── Stage 9: Mediator Extraction
        self.triplet_mediator = RobustTripletMediator(
            query_dim=self.dim,
            triplet_dim=cfg.model.text_dim,
            attn_dim=self.dim,
            dropout=self.dropout
        )

        # ── Stage 10: Causal Signal Projections
        self.E_z_proj = nn.Sequential(
            nn.Linear(cfg.model.concept_dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.Dropout(self.dropout)
        )

        self.E_v_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.Dropout(self.dropout)
        )

        # ── Stage 11: Causal Fusion (MoE)
        self.causal_fusion = MoELinearFusion(
            in_dim=self.dim * 3,
            out_dim=self.dim,
            num_qtypes=cfg.data.num_qtypes
        )

        # ── Stage 12: Final Classification
        self.final_mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, cfg.model.msvd_vocab_size)
        )

        # ── Temperature & Gates (FIXED: dtype)
        self.logit_temp = nn.Parameter(
            torch.clamp(torch.tensor(1.0, dtype=torch.float32), min=0.1, max=2.0)
        )
        self.gate_graph = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def _validate_config(self, cfg):
        """Validate all required config parameters exist."""
        required_paths = ['prior_pt_path', 'triplet_pt_path', 'qtype_pt_path', 'qrole_pt_path']
        for path_key in required_paths:
            path = getattr(cfg.model, path_key, None)
            if path is None:
                raise ValueError(f"Config missing: cfg.model.{path_key}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        required_dims = [
            'raw_dim', 'dim', 'text_dim', 'concept_dim', 'motion_dim',
            'edge_dim', 'num_classes', 'num_preds', 'num_heads', 'num_layers',
            'num_anchors', 'num_prototypes', 'msvd_vocab_size', 'mediator_dim', 'max_triplets'
        ]

        for dim_key in required_dims:
            if not hasattr(cfg.model, dim_key):
                raise ValueError(f"Config missing: cfg.model.{dim_key}")

    def _prepare_question_input(self, question_emb: torch.Tensor) -> torch.Tensor:
        if question_emb.dim() == 3:
            question_emb = question_emb.mean(dim=1)
        elif question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)
        elif question_emb.dim() != 2:
            raise ValueError(f"Unsupported question_emb shape: {question_emb.shape}")
        return question_emb

    def _prepare_motion_input(self, motion_feat: torch.Tensor, lengths: torch.Tensor, device: torch.device):
        if motion_feat.dim() == 2:
            motion_feat = motion_feat.unsqueeze(1)
        if motion_feat.dim() != 3:
            raise ValueError(f"Unsupported motion_feat shape: {motion_feat.shape}")

        if lengths is None:
            lengths = torch.full(
                (motion_feat.size(0),),
                motion_feat.size(1),
                dtype=torch.long,
                device=device
            )
        else:
            lengths = lengths.squeeze(-1).to(device)

        return motion_feat, lengths

    def _compute_causal_signals(
        self,
        cls_id: torch.Tensor,
        s_idx: torch.Tensor,
        batch_idx: torch.Tensor,
        v_graph_targeted: torch.Tensor,
        B: int
    ) -> tuple:
        """
        Compute causal signals with proper error handling.
        """
        device = cls_id.device

        # 1. Expected Visual Prior
        E_v_global = self.visual_memory.get_expected_visual()  # (1, dim)
        E_v = E_v_global.expand(B, -1)  # (B, dim)
        E_v = self.E_v_proj(E_v)  # FIXED: Project E_v

        # 2. Expected Semantic Prior
        E_z_nodes = self.conceptnet_prior(cls_id)  # (N, concept_dim)
        E_z = global_mean_pool(E_z_nodes, batch_idx)  # (B, concept_dim)
        E_z = self.E_z_proj(E_z)  # (B, dim)

        # 3. Graph-Conditioned Mediator
        # FIXED: Handle empty edges
        if s_idx.numel() == 0 or s_idx.dim() < 2 or s_idx.size(0) < 2 or s_idx.size(1) == 0:
            M = torch.zeros(B, self.dim, device=device, dtype=E_z.dtype)
            return E_z, M, E_v

        src, dst = s_idx[0], s_idx[1]
        edge_batch = batch_idx[src]

        E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst])  # (E, num_triplets, text_dim)
        triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)  # (E, num_triplets)

        M = self.triplet_mediator(
            v_graph=v_graph_targeted,
            E_T=E_T,
            edge_batch=edge_batch,
            triplet_mask=triplet_mask,
            B=B
        )

        return E_z, M, E_v

    def forward(self, data) -> dict:
        """
        PRODUCTION-READY: Complete forward pass.
        """

        device = next(self.parameters()).device

        # FIXED: Move data to device
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        # ── Unpack
        node_raw = data["node_raw"]
        bbox = data["node_bbox"]
        cls_id = data["node_class"]
        conf = data["node_conf"]
        node_obj_ids_flat = data["node_obj_ids_flat"]
        node_is_keyframe = data["node_is_keyframe"]
        node_kf_list_idx = data["node_kf_list_idx"]
        batch_idx = data["batch"]

        motion_feat = data.get("motion_feat", data.get("motion_raw"))
        if motion_feat is None:
            raise KeyError("Expected motion_feat or motion_raw in data.")
        lengths = data.get("num_temporal_slots")

        s_idx = data["s_idx"]
        s_attr_ids = data["s_attr"]
        t_idx = data["t_idx"]
        t_attr_ids = data["t_attr"]

        question_emb = data.get("question_emb", data.get("question_raw"))
        if question_emb is None:
            raise KeyError("Expected question_emb or question_raw in data.")
        qtype_idx = data["qtype_idx"].squeeze(-1)

        question_emb = self._prepare_question_input(question_emb)
        motion_feat, lengths = self._prepare_motion_input(motion_feat, lengths, device)

        B = question_emb.size(0)

        # ── Question Encoding
        Q_emb = self.question_encoder(question_emb)  # (B, dim)

        # ── Linguistic Prior
        E_cl = self.linguistic_prior_bank(
            qtype_idx,
            data.get("triplet_idxs", torch.zeros(B, dtype=torch.long, device=device)),
            data.get("triplet_mask", torch.ones(B, dtype=torch.bool, device=device))
        )  # (B, concept_dim)
        E_cl = self.confounder_proj(E_cl)  # (B, dim)

        # ── Deconfounding
        Q_deconf = self.deconfounding(original=Q_emb, confounder=E_cl)  # (B, dim)

        # ── Motion Processing
        batch_max_T = lengths.max().item()
        motion_feat = motion_feat[:, :batch_max_T, :]

        motion_mask = (
            torch.arange(batch_max_T, device=device).expand(B, batch_max_T)
            < lengths.unsqueeze(1)
        )

        motion_feat = self.motion_proj(motion_feat)  # (B, T, dim)

        # ── Node Encoding & ST-Graph
        h = self.node_encoder(
            node_raw, bbox, cls_id, conf,
            node_obj_ids_flat, node_is_keyframe, node_kf_list_idx
        )  # (N, dim)

        if self.training:
            h = F.dropout(h, p=self.dropout, training=True)

        for block in self.blocks:
            h_residual = h
            h, s_attr, t_attr = block(
                h,
                s_idx, self.edge_embed(s_attr_ids),
                t_idx, self.edge_embed(t_attr_ids)
            )
            h = h + torch.sigmoid(self.gate_graph) * (h_residual - h)

        node_feat = self.node_feat_norm(h)  # (N, dim)

        # ── Cross-Modal Fusion (FIXED: All-in-one)
        cross_fused = self.cross_modal_fusion(
            Q_emb=Q_deconf,
            v_graph=node_feat,
            motion_feat=motion_feat,
            batch_idx=batch_idx,
            motion_mask=motion_mask
        )  # (B, dim)

        # ── Memory Update
        if self.training:
            self.visual_memory.update(
                node_feat=node_feat.detach(),
                batch_idx=batch_idx,
                num_graphs=B
            )

        # ── Causal Signals
        E_z, M, E_v = self._compute_causal_signals(
            cls_id=cls_id,
            s_idx=s_idx,
            batch_idx=batch_idx,
            v_graph_targeted=cross_fused,
            B=B
        )

        # ── Orthogonal Deconfounding
        E_z_unit = F.normalize(E_z, p=2, dim=-1)
        M_deconf = M - (torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit)

        # ── Causal Fusion
        causal_targeted = self.causal_fusion(E_z, E_v, M_deconf, qtype_idx)  # (B, dim)

        # ── Dropout Before Fusion
        if self.training:
            cross_fused = F.dropout(cross_fused, p=0.1, training=True)
            causal_targeted = F.dropout(causal_targeted, p=0.1, training=True)

        # ── Final Fusion & Classification (FIXED: Correct dims)
        final_representation = torch.cat([cross_fused, causal_targeted], dim=-1)  # (B, 2*dim)

        final_representation = F.dropout(
            final_representation,
            p=self.final_drop,
            training=self.training
        )

        causal_logits = self.final_mlp(final_representation)  # (B, vocab_size)

        temp = torch.clamp(self.logit_temp.abs(), min=0.1, max=2.0)
        causal_logits = causal_logits / (temp + 1e-8)

        return {
            "causal_logits": causal_logits,
        }
