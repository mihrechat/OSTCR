import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
import torch.nn.functional as F
from .node_attribute_encoder import NodeAttributeEncoder
from torch_geometric.utils import softmax as pyg_softmax
import os
from torch_geometric.utils import dropout_edge
from models.multiEncoder.cross_encoder import QuestionGuidedPooling, QuestionGuidedMotionPooling
from .moe_fusion import QuestionGatedCausalFusion
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank,ConceptNetExpectedPriorBank
import math
from torch_scatter import scatter_mean
     

# ==================================================================
# 1. Triplet Prior Bank (M) — Edge-level Causal Mechanism
# ==================================================================

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

        self.register_buffer("prototypes", F.normalize(torch.randn(num_prototypes, dim), dim=-1))
        self.register_buffer("update_counts", torch.zeros(num_prototypes))

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
        
        return F.normalize(E_v, dim=-1)


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
        self.q_emb_drop = cfg.train.q_emb_drop
        self.final_dop =  cfg.train.final_drop
        self.h_drop    = cfg.train.h_drop
        self.q_drop_cross = cfg.train.q_emb_drop_cross

        # ── Stage 1: ST-Graph representation 
        self.node_encoder = NodeAttributeEncoder(self.raw_dim, self.dim, cfg.model.num_classes)
        
        self.out_norm = nn.LayerNorm(self.dim)
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
        
        # ── Linguistic Prior Bank for E_cl
        self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(qtype_pt_path=cfg.model.qtype_pt_path, qrole_pt_path=cfg.model.qrole_pt_path)
        
        # ── Causal Fusion Layer
        # ── MoE Linear Fusion for Deconfounded Representation
        # in_dim = cfg.model.dim * 3
        self.causal_fusion = QuestionGatedCausalFusion(text_dim=self.dim, out_dim=self.dim)
        
        # self.causal_fusion = QuestionGatedCausalFusion(in_dim=in_dim, text_dim=self.dim, out_dim=self.dim)
     
        self.question_crossattention_pooling = QuestionGuidedPooling(text_dim=cfg.model.text_dim, node_dim=self.dim) 
        self.motion_proj = nn.Linear(cfg.model.motion_dim, cfg.model.dim)
        self.question_motion_crossattentionpooling = QuestionGuidedMotionPooling(model_dim=cfg.model.dim)
        self.lang_proj = nn.Linear(cfg.model.text_dim, cfg.model.dim)
        self.msvd_output = nn.Sequential(
            nn.Linear(cfg.model.dim * 5, cfg.model.dim * 2),
            nn.GELU(),
            nn.Dropout(0.3), 
            nn.Linear(cfg.model.dim * 2, cfg.model.msvd_vocab_size) 
        ) 
        self.causal_attention = nn.MultiheadAttention(embed_dim=cfg.model.dim, num_heads=cfg.model.num_heads, batch_first=True)
        self.ez_proj  = nn.Linear(cfg.model.concept_dim, cfg.model.dim) # 1024 -> 512
        self.ecl_proj = nn.Linear(cfg.model.concept_dim, cfg.model.dim) # 1024 -> 512

    def _compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
        B = int(batch_idx.max()) + 1  

        # 1. Expected Visual Prior (E_v) -> KEEP THIS GLOBAL (Correct Causal Math)
        E_v_global = self.visual_memory.get_expected_visual()  
        E_v = E_v_global.expand(B, -1)                         

        # 2. Expected Semantic Prior (E_z) 
        E_z_nodes = self.conceptnet_prior(cls_id)                 
        E_z = global_mean_pool(E_z_nodes, batch_idx)              

        # 3. Graph-Conditioned Mediator (M) 
        v_graph = v_graph_targeted
        
        src, dst = s_idx[0], s_idx[1]
        edge_batch = batch_idx[src]  
        
        E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst]) 
        triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)             
        
        q_graph = self.W_q(v_graph)                        
        k_edges = self.W_k(E_T)                                   
        v_edges = self.W_v(E_T)                                   
        
        # Triplet-level attention (Which ConceptNet word matters?)
        q_expanded = q_graph[edge_batch].unsqueeze(1)             
        scale = q_graph.size(-1) ** 0.5
        attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale  
        attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
        alpha = torch.softmax(attn_logits / 0.5, dim=-1)                
        
        m_weighted = v_edges * alpha.unsqueeze(-1)                
        m_edge = m_weighted.sum(dim=1) # [Num_Edges, Dim]
        
        q_graph_edges = q_graph[edge_batch]                      # [Num_Edges, Dim]
        edge_scores = (m_edge * q_graph_edges).sum(dim=-1)       # [Num_Edges]
        
        # Safe Per-Graph Softmax (to prevent dominant videos from crushing small videos)
        from torch_scatter import scatter_max, scatter_sum
        edge_scores_max, _ = scatter_max(edge_scores, edge_batch, dim=0, dim_size=B)
        exp_scores = torch.exp(edge_scores - edge_scores_max[edge_batch])
        sum_exp_scores = scatter_sum(exp_scores, edge_batch, dim=0, dim_size=B) + 1e-6
        edge_weights = exp_scores / sum_exp_scores[edge_batch]   # [Num_Edges]
        
        # Multiply each edge by its attention weight, then SUM (not mean!)
        M = scatter_sum(m_edge * edge_weights.unsqueeze(-1), edge_batch, dim=0, dim_size=B)
        
        return E_z, M, E_v
    

    def forward(self, data) -> dict:
        # ── Unpack 
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx = data["s_idx"]
        Q_emb_raw, qtype_idx = data["question_emb"], data["qtype_idx"].squeeze(-1)
        
        Q_emb = self.lang_proj(Q_emb_raw) 
        
        if self.training:
            Q_emb_drop = F.dropout(Q_emb, p=self.q_emb_drop, training=True)
        else:
            Q_emb_drop = Q_emb
        
       
        lengths = lengths.squeeze(-1)
        batch_max_T = lengths.max().item() 
        
        motion_feat = motion_feat[:, :batch_max_T, :] 
        B = motion_feat.size(0)
        motion_mask = torch.arange(batch_max_T, device=motion_feat.device).expand(B, batch_max_T) < lengths.unsqueeze(1)
        
        motion_feat = self.motion_proj(motion_feat)
        motion_targeted = self.question_motion_crossattentionpooling(Q_emb, motion_feat, motion_mask)

        h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
        if self.training:
            h = F.dropout(h, p=self.h_drop, training=True)
            
        node_feat = self.out_norm(h)               
        node_feat_proj = self.proj_head(node_feat) 
        
        # INJECTION 1: Question-Guided Graph Pooling 
        v_graph_targeted = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)

        # ── Stage 2: Causal Mechanism
        if self.training:
            self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=data.num_graphs)
        
        E_z, M, E_v = self._compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_targeted)
        E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
        
        E_z = self.ez_proj(E_z)
        
        # Orthogonal Deconfounding 
        # Mathematically remove the confounding vector (E_z) from the Mediator (M)
        E_z_unit = F.normalize(E_z, p=2, dim=-1)
        M_deconf = M - (torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit)

        # [NEW] 3. Pre-fusion Normalization for Multimodal Stability
        E_z_norm = F.normalize(E_z, dim=-1)
        E_v_norm = F.normalize(E_v, dim=-1)
        M_deconf_norm = F.normalize(M_deconf, dim=-1)
        
        # Pass the cleanly deconfounded and normalized M into the fusion
        visual_causal_fusion = self.causal_fusion(E_z_norm, E_v_norm, M_deconf_norm, Q_emb) 
        
        # Cross-Modal Alignment 
        if self.q_drop_cross:
            v_graph_gated = v_graph_targeted * Q_emb_drop
            motion_gated  = motion_targeted * Q_emb_drop
            causal_gated  = visual_causal_fusion * Q_emb_drop
        else:
            v_graph_gated = v_graph_targeted * Q_emb
            motion_gated  = motion_targeted * Q_emb
            causal_gated  = visual_causal_fusion * Q_emb 
            
        # Leaky Frontdoor Forcing (Path Dropout)
        # Suppress the direct shortcut 30% of the time, forcing M->A to do the heavy lifting
        if self.training and torch.rand(1).item() < 0.3:
            v_graph_gated = v_graph_gated * 0.2
        
        E_cl_proj = self.ecl_proj(E_cl)
        
        # ── JOINT EXPECTATION FOR NWGM ──
        final_representation = torch.cat([
            v_graph_gated,         
            motion_gated,          
            causal_gated,          
            Q_emb_drop,            
            E_cl_proj              
        ], dim=-1)                 
        
        final_representation = F.dropout(final_representation, p=self.final_dop, training=self.training)
        causal_logits = self.msvd_output(final_representation) 
       
        return {
            "node_feat":      node_feat,
            "node_feat_proj": node_feat_proj,      
            "causal_logits":  causal_logits,  
        }