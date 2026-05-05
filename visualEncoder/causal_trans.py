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
import os
from torch_geometric.utils import dropout_edge
from models.multiEncoder.cross_encoder import QuestionGuidedPooling, QuestionGuidedMotionPooling
from .moe_fusion import QuestionGatedCausalFusion, CausalGatedFusion
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank,ConceptNetExpectedPriorBank
import math
from torch_scatter import scatter_mean
     


# ==================================================================
# 1. Triplet Prior Bank (M) — Edge-level Causal Mechanism
# ==================================================================


class CausalGatedFusion(nn.Module):
    """
    Clean causal fusion:
    - Q acts as controller (gates), NOT as feature shortcut
    - No direct Q concatenation
    - Forces all modalities to contribute
    """
    def __init__(self, dim):
        super().__init__()
        
        # modality projections (stabilize)
        self.v_proj = nn.Linear(dim, dim)
        self.m_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)

        # question controller → produces gates
        self.q_controller = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3)   # 3 modalities
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, v, motion, causal, q):
        """
        v: visual graph (B, D)
        motion: motion (B, D)
        causal: causal (B, D)
        q: question (B, D)
        """

        # project modalities
        v = self.v_proj(v)
        motion = self.m_proj(motion)
        causal = self.c_proj(causal)

        modalities = torch.stack([v, motion, causal], dim=1)  # (B, 3, D)

        # Q produces gates
        # gates = torch.sigmoid(self.q_controller(q))  # (B, 3)
         # gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        gates = torch.softmax(self.q_controller(q), dim=-1)

        # apply gating
        fused = (modalities * gates.unsqueeze(-1)).sum(dim=1)

        return self.out_norm(fused), gates
    
class ConceptNetTripletBank(nn.Module):
    """
    O(1) Lookup Table for Precomputed ConceptNet Triplets.
    Outputs the candidate causal mechanisms (M) for Cross-Attention.
    """
    def __init__(self, triplet_pt_path="M_triplets.pt", max_triplets=10):
        super().__init__()
        self.max_triplets = max_triplets
        
        # Load offline extracted dictionary
        triplet_dict = torch.load(triplet_pt_path)
        sample_tensor = next(iter(triplet_dict.values()))
        dim = sample_tensor.shape[1]
        num_classes = 80 # Match your COCO/Visual Genome classes
        
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
        return self.lookup_table[src_cls_ids, dst_cls_ids]


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
        self.momentum  = momentum
        self.min_count = min_count
        self.K         = num_prototypes
        
        target_norm = (dim ** 0.5)  # 22.63
        init_protos = torch.randn(num_prototypes, dim)
        init_protos = F.normalize(init_protos, dim=-1) * target_norm
        
        self.register_buffer("prototypes", init_protos)
        self.register_buffer("update_counts", torch.zeros(num_prototypes))

        # self.register_buffer("prototypes", F.normalize(torch.randn(num_prototypes, dim), dim=-1))
        # self.register_buffer("update_counts", torch.zeros(num_prototypes))

    @torch.no_grad()
    def update(self, node_feat: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int):
        video_reprs = []
        for b in range(num_graphs):
            mask = (batch_idx == b)
            if mask.sum() == 0: continue
            v_repr = F.normalize(node_feat[mask].mean(dim=0), dim=-1)
            video_reprs.append(v_repr)

        if not video_reprs: return

        video_reprs = torch.stack(video_reprs)        # (B, dim)
        sim         = video_reprs @ self.prototypes.T # (B, K)
        assignments = sim.argmax(dim=-1)              # (B,)

        for k in range(self.K):
            assigned = video_reprs[assignments == k]
            if assigned.shape[0] == 0: continue
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
# 4. Main ST-Graph Transformer with Causal Intervention
# ==================================================================

class STGraphTransformerNet(nn.Module):
    """ ST-Graph Transformer fully integrated with Visual Dual-Deconfounding. """

    def __init__(self, cfg):
        super().__init__()
        self.dropout = cfg.model.dropout
        self.dim = cfg.model.dim
        self.raw_dim = cfg.model.raw_dim
        self.edge_dim = cfg.model.edge_dim
        self.final_drop  = cfg.train.final_drop
        self.h_drop      = cfg.train.h_drop
        self.q_drop      = cfg.train.q_drop
        self.m_drop      = cfg.train.m_drop
        self.v_drop      = cfg.train.v_drop
        self.c_drop      = cfg.train.c_drop
        self.ecl_drop    = cfg.train.ecl_drop
        self.q_cross     = cfg.train.q_cross
        self.ecl_drop    = cfg.train.ecl_drop
        # ── Stage 1: ST-Graph representation 
        self.node_encoder = NodeAttributeEncoder(self.raw_dim, self.dim, cfg.model.num_classes)
        self.edge_embed = nn.Embedding(cfg.model.num_preds, self.edge_dim)
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(self.dim, cfg.model.num_heads, self.edge_dim, cfg.model.num_anchors)
            for _ in range(cfg.model.num_layers)
        ])
        self.proj_head = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2), nn.GELU(), nn.Dropout(cfg.model.dropout), nn.Linear(self.dim * 2, cfg.model.proj_dim)
        )

        # ── Causal Memory Modules
        self.visual_memory = VisualPrototypeMemory(num_prototypes=cfg.model.num_prototypes, dim=self.dim, momentum=cfg.model.prototype_momentum)
        self.motion_memory = VisualPrototypeMemory(
            num_prototypes=cfg.model.num_prototypes, 
            dim=self.dim, 
            momentum=cfg.model.prototype_momentum
        )
        self.conceptnet_prior = ConceptNetExpectedPriorBank(cfg.model.prior_pt_path) # E_z (Node-level Backdoor)
        self.conceptnet_triplets = ConceptNetTripletBank(cfg.model.triplet_pt_path)    # M (Edge-level Frontdoor)

        # ── Cross-Attention Projections for Mediator M
        self.W_q = nn.Linear(cfg.model.mediator_dim, cfg.model.attn_dim)
        self.W_k = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        self.W_v = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        
        self.out_norm = nn.LayerNorm(self.dim)
        
        # ── Linguistic Prior Bank for E_cl
        # self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(qtype_pt_path=cfg.model.qtype_pt_path, qrole_pt_path=cfg.model.qrole_pt_path)
        
        # ── Causal Fusion Layer
        # ── MoE Linear Fusion for Deconfounded Representation
        in_dim = cfg.model.dim * 3
        
        self.causal_fusion = QuestionGatedCausalFusion(in_dim=in_dim, text_dim=self.dim, out_dim=self.dim)
        self.fusion        = CausalGatedFusion(self.dim)
     
        self.question_crossattention_pooling = QuestionGuidedPooling(text_dim=self.dim, node_dim=self.dim) 
        self.motion_proj = nn.Linear(cfg.model.motion_dim, cfg.model.dim)
        self.question_motion_crossattentionpooling = QuestionGuidedMotionPooling(model_dim=cfg.model.dim)
        
        self.logit_temp = nn.Parameter(torch.tensor(1.0)) 
       # Keep this one:
        self.motion_proj = nn.Sequential(
            nn.Linear(cfg.model.motion_dim, cfg.model.dim),
            nn.LayerNorm(cfg.model.dim)
        )
        
       
        self.ez_scale = nn.Parameter(torch.ones(1) * 1.0)  
        self.ecl_scale = nn.Parameter(torch.ones(1) * 1.0) 

        self.ez_proj = nn.Sequential(
            nn.Linear(cfg.model.concept_dim, cfg.model.dim),
            nn.LayerNorm(cfg.model.dim)
        )

        self.ecl_proj = nn.Sequential(
            nn.Linear(cfg.model.concept_dim, cfg.model.dim),
        )

        self.lang_proj = nn.Sequential(
            nn.Linear(cfg.model.text_dim, cfg.model.dim),
            nn.LayerNorm(cfg.model.dim),
            nn.Dropout(0.2)
        )
        self.option_projection = nn.Sequential(
            nn.Linear(cfg.model.text_dim, cfg.model.dim),
            nn.LayerNorm(cfg.model.dim),
            nn.Dropout(0.2)
        )

        self.msvd_output = nn.Sequential(
            nn.Dropout(cfg.train.final_drop),
            nn.Linear(self.dim, self.dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim*4, self.dim),
            nn.GELU(),
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, cfg.model.msvd_vocab_size)
        )
        self.aux_head = nn.Sequential(
            nn.Dropout(cfg.train.final_drop),
            nn.Linear(self.dim, self.dim *4),
            nn.GELU(),
            nn.LayerNorm(self.dim*4),
            nn.Linear(self.dim*4, cfg.model.msvd_vocab_size)
        )
        self.causal_attention = nn.MultiheadAttention(embed_dim=cfg.model.dim, num_heads=cfg.model.num_heads, batch_first=True)
        self.alpha_param = nn.Parameter(torch.tensor(1.0))
        
      
    def _compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
        B = int(batch_idx.max()) + 1  
        target_norm = (self.dim ** 0.5)  # 22.63 for 512-dim

        # 1. Expected Visual Prior (E_v) 
        E_v_global = self.visual_memory.get_expected_visual()  
        E_v = E_v_global.expand(B, -1)
        E_v = F.normalize(E_v, p=2, dim=-1) * target_norm

        # 2. Expected Semantic Prior (E_z) - Keep at concept_dim (1024)
        E_z_nodes = self.conceptnet_prior(cls_id)                 
        E_z = global_mean_pool(E_z_nodes, batch_idx)
        E_z = F.normalize(E_z, p=2, dim=-1)  #

        # 3. Graph-Conditioned Mediator (M) 
        src, dst = s_idx[0], s_idx[1]
        edge_batch = batch_idx[src]
        
        E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst])
        triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)
        
        # Normalize inputs before attention
        v_graph_norm = F.normalize(v_graph_targeted, p=2, dim=-1)
        E_T_norm = F.normalize(E_T, p=2, dim=-1)
        
        q_graph = self.W_q(v_graph_norm)
        k_edges = self.W_k(E_T_norm)
        v_edges = self.W_v(E_T_norm)
        
        q_expanded = q_graph[edge_batch].unsqueeze(1)
        scale = q_graph.size(-1) ** 0.5
        attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale
        attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
        alpha = torch.softmax(attn_logits, dim=-1)
        m_weighted = v_edges * alpha.unsqueeze(-1)
        m_edge = m_weighted.sum(dim=1)
        
        # Use scatter_mean
        M = scatter_mean(m_edge, edge_batch, dim=0, dim_size=B)
        
        batch_counts = torch.bincount(edge_batch, minlength=B)
        zero_batches = (batch_counts == 0)
        
        if zero_batches.any():
            # Use the existing ez_proj to project E_z to 512-dim
            with torch.no_grad():
                fallback = self.ez_proj(E_z)  # Project from 1024 to 512
                fallback = F.normalize(fallback, p=2, dim=-1) * target_norm
            M[zero_batches] = fallback[zero_batches]
        
        # Force M to have correct norm
        M = F.normalize(M, p=2, dim=-1) * target_norm
        
        return E_z, M, E_v

    def forward(self, data) -> dict:
        # ── Unpack 
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
        t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
        Q_emb, Opt_emb, qtype_idx = data["question_emb"], data["options_emb"], data["qtype_idx"].squeeze(-1)
        
        
        Q_emb = self.lang_proj(Q_emb)
        target_norm = (Q_emb.size(-1) ** 0.5)
        Q_emb = F.normalize(Q_emb, p=2, dim=-1) * target_norm
        
        Opt_emb_proj = self.option_projection(Opt_emb) 
        Opt_emb_proj = F.normalize(Q_emb, p=2, dim=-1) * target_norm
        
        # ── Motion features
        lengths = lengths.squeeze(-1)
        batch_max_T = lengths.max().item()
        motion_feat = motion_feat[:, :batch_max_T, :]
        B = motion_feat.size(0)
        motion_mask = torch.arange(batch_max_T, device=motion_feat.device).expand(B, batch_max_T) < lengths.unsqueeze(1)
        
        motion_feat = self.motion_proj(motion_feat)
        motion_targeted = self.question_motion_crossattentionpooling(Q_emb, motion_feat, motion_mask)
        
        # ── ST-Graph
        s_attr = self.edge_embed(s_attr_ids)
        t_attr = self.edge_embed(t_attr_ids)
        
        h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
        if self.training:
            h = F.dropout(h, p=self.h_drop, training=True)
        
        for block in self.blocks:
            h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)
        
        node_feat = self.out_norm(h)
        v_graph_targeted = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)
        
        
        # ── Causal mechanism
        if self.training:
            self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=data.num_graphs)
        
        E_z, M, E_v = self._compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_targeted)
        E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
        
        E_cl_targeted = self.ecl_proj(E_cl) * self.ecl_scale
        E_cl = F.normalize(E_cl_targeted, p=2, dim=-1) * target_norm
        E_z = self.ez_proj(E_z) * self.ez_scale
        
        
        # ── Strict Causal Deconfounding
        E_z_unit = F.normalize(E_z, p=2, dim=-1)
        projection = (torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit)
        alpha = torch.sigmoid(self.alpha_param)
        M_deconf = M - alpha* projection
        M_deconf = F.normalize(M_deconf, p=2, dim=-1) * target_norm
        
        causal_targeted = self.causal_fusion(E_z, E_v, M_deconf, Q_emb)
        
        
        # ── Normalize modalities
        v_graph_ln = v_graph_targeted
        motion_ln  = motion_targeted
        causal_ln  = causal_targeted
        Q_ln       = Q_emb

        if self.training:
            v_graph_ln = F.dropout(v_graph_ln, p=self.v_drop)
            motion_ln  = F.dropout(motion_ln,  p=self.m_drop)
            causal_ln  = F.dropout(causal_ln,  p=self.c_drop)
            E_cl       = F.dropout(E_cl,       p=self.ecl_drop)

        
        fused_context, fusion_gates = self.fusion(
            v_graph_ln,
            motion_ln,
            causal_ln,
            E_cl,
            Q_ln
        )
        
        # Handle shape safety depending on how your options are batched
        if Opt_emb_proj.dim() == 2:
            num_options = Opt_emb_proj.size(0) // B 
            Opt_emb_proj = Opt_emb_proj.view(B, num_options, -1)
        else:
            num_options = Opt_emb_proj.size(1)
        # Expand to match the options
        fused_expanded = fused_context.unsqueeze(1).expand(-1, num_options, -1) # (B, 4, dim)
        causal_expanded = causal_ln.unsqueeze(1).expand(-1, num_options, -1)
        
        # Scaled Weighted Dot-Product for Contrastive Logits
        causal_logits = torch.sum(fused_expanded * Opt_emb_proj, dim=-1) / math.sqrt(fused_expanded.size(-1))
        aux_logits   =  torch.sum(causal_expanded * Opt_emb_proj, dim=-1) / math.sqrt(causal_expanded.size(-1)) 
       
        causal_logits = causal_logits / (self.logit_temp.abs() + 1e-4)
        aux_logits =    aux_logits / (self.logit_temp.abs() + 1e-4)
        
        return {"causal_logits": causal_logits,
                "aux_logits": aux_logits,
                "gates": fusion_gates}
        
