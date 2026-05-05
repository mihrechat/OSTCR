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
from .moe_fusion import QuestionGatedCausalFusion
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank,ConceptNetExpectedPriorBank
import math
from torch_scatter import scatter_mean
from torch_scatter import scatter_max, scatter_sum
     

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
        self.dropout          =  cfg.model.dropout
        self.dim              =  cfg.model.dim
        self.raw_dim          =  cfg.model.raw_dim
        self.edge_dim         =  cfg.model.edge_dim
        self.q_drop           =  cfg.train.q_drop
        self.final_drop       =  cfg.train.final_drop
        self.h_drop           =  cfg.train.h_drop
        self.v_drop           =  cfg.train.v_drop
        self.m_drop           =  cfg.train.m_drop
        self.c_drop           =  cfg.train.c_drop
        self.m_gated          = cfg.train.m_gated
   

        # ── Stage 1: ST-Graph representation 
        self.node_encoder = NodeAttributeEncoder(self.raw_dim, self.dim, cfg.model.num_classes)
        self.edge_embed = nn.Embedding(cfg.model.num_preds, self.edge_dim)
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(self.dim, cfg.model.num_heads, self.edge_dim, cfg.model.num_anchors)
            for _ in range(cfg.model.num_layers)
        ])
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
            nn.Linear(cfg.model.dim * 3, cfg.model.dim * 1),
            nn.GELU(),
            nn.Dropout(0.3), 
            nn.Linear(cfg.model.dim * 1, cfg.model.msvd_vocab_size) 
        ) 
        self.causal_attention = nn.MultiheadAttention(embed_dim=cfg.model.dim, num_heads=cfg.model.num_heads, batch_first=True)
        self.ez_proj  = nn.Linear(cfg.model.concept_dim, cfg.model.dim) # 1024 -> 512
        self.ecl_proj = nn.Linear(cfg.model.concept_dim, cfg.model.dim) # 1024 -> 512
        self.mediator_scale = nn.Parameter(torch.tensor(0.1))
        self.gate_temp = 0.7
        
        self.branch_logits = nn.Parameter(torch.zeros(3)) 
        self.qtype_embedding = nn.Embedding(cfg.model.num_qtypes, cfg.model.qtype_emb_dim)  # e.g., qtype_emb_dim = 32
        
        self.branch_gate_mlp = nn.Sequential(
            nn.Linear(self.dim , self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 3))

        # self.branch_gate_mlp = nn.Sequential(
        #     nn.LayerNorm(self.dim),
        #     nn.Linear(self.dim, self.dim // 4),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.dim // 4, 3)  # 3 branches: v_graph, motion, causal
        # )
        # self.causal_only_head = nn.Linear(causal_gated.size(-1), num_classes)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim, self.dim)
        )
        self.branch_scale = nn.Parameter(torch.ones(3))
        self.q_gate_proj = nn.Linear(self.dim, self.dim)
        self.pre_fusion_mlp = nn.Sequential(
            nn.Linear(self.dim*3, self.dim*2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim*2, self.dim)
        )

    # def _compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
    #     B = int(batch_idx.max()) + 1  

    #     # 1. Expected Visual Prior (E_v) -> KEEP THIS GLOBAL (Correct Causal Math)
    #     E_v_global = self.visual_memory.get_expected_visual()  
    #     E_v = E_v_global.expand(B, -1)                         

    #     # 2. Expected Semantic Prior (E_z) 
    #     E_z_nodes = self.conceptnet_prior(cls_id)                 
    #     E_z = global_mean_pool(E_z_nodes, batch_idx)              

    #     # 3. Graph-Conditioned Mediator (M) 
    #     v_graph = v_graph_targeted
        
    #     src, dst = s_idx[0], s_idx[1]
    #     edge_batch = batch_idx[src]  
        
    #     E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst]) 
    #     triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)             
        
    #     q_graph = self.W_q(v_graph)                        
    #     k_edges = self.W_k(E_T)                                   
    #     v_edges = self.W_v(E_T)                                   
        
    #     # Triplet-level attention (Which ConceptNet word matters?)
    #     q_expanded = q_graph[edge_batch].unsqueeze(1)             
    #     scale = q_graph.size(-1) ** 0.5
    #     attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale  
    #     attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
    #     alpha = torch.softmax(attn_logits / 0.5, dim=-1)                
        
    #     m_weighted = v_edges * alpha.unsqueeze(-1)                
    #     m_edge = m_weighted.sum(dim=1) # [Num_Edges, Dim]
        
    #     q_graph_edges = q_graph[edge_batch]                      # [Num_Edges, Dim]
    #     edge_scores = (m_edge * q_graph_edges).sum(dim=-1)       # [Num_Edges]
        
    #     # Safe Per-Graph Softmax (to prevent dominant videos from crushing small videos)
    #     from torch_scatter import scatter_max, scatter_sum
    #     edge_scores_max, _ = scatter_max(edge_scores, edge_batch, dim=0, dim_size=B)
    #     exp_scores = torch.exp(edge_scores - edge_scores_max[edge_batch])
    #     sum_exp_scores = scatter_sum(exp_scores, edge_batch, dim=0, dim_size=B) + 1e-6
    #     edge_weights = exp_scores / sum_exp_scores[edge_batch]   # [Num_Edges]
        
    #     # Multiply each edge by its attention weight, then SUM (not mean!)
    #     M = scatter_sum(m_edge * edge_weights.unsqueeze(-1), edge_batch, dim=0, dim_size=B)
        
    #     return E_z, M, E_v
    

    # def forward(self, data) -> dict:
    #     # ── Unpack 
    #     node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
    #     node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
    #     motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
    #     node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
    #     s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
    #     t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
    #     Q_emb_raw, qtype_idx = data["question_emb"], data["qtype_idx"].squeeze(-1)
        
    #     Q_emb = self.lang_proj(Q_emb_raw) 
        
    #     if self.training:
    #         Q_emb_drop = F.dropout(Q_emb, p=self.q_drop, training=True)
    #     else:
    #         Q_emb_drop = Q_emb
        
    #     # ── Stage 1: ST-Graph Forward
    #     s_attr = self.edge_embed(s_attr_ids)
    #     t_attr = self.edge_embed(t_attr_ids)
    #     lengths = lengths.squeeze(-1)
    #     batch_max_T = lengths.max().item() 
        
    #     motion_feat = motion_feat[:, :batch_max_T, :] 
    #     B = motion_feat.size(0)
    #     motion_mask = torch.arange(batch_max_T, device=motion_feat.device).expand(B, batch_max_T) < lengths.unsqueeze(1)
        
    #     motion_feat = self.motion_proj(motion_feat)
    #     motion_targeted = self.question_motion_crossattentionpooling(Q_emb, motion_feat, motion_mask)

    #     h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
    #     if self.training:
    #         h = F.dropout(h, p=self.h_drop, training=True)
            
    #     for block in self.blocks:
    #         h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)

    #     node_feat = self.out_norm(h)               
    #     # node_feat_proj = self.proj_head(node_feat) 
        
    #     # INJECTION 1: Question-Guided Graph Pooling 
    #     v_graph_targeted = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)

    #     # ── Stage 2: Causal Mechanism
    #     if self.training:
    #         self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=data.num_graphs)
        
    #     E_z, M, E_v = self._compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_targeted)
    #     E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
        
    #     E_z = self.ez_proj(E_z)
    #     E_cl_proj = self.ecl_proj(E_cl)
        
    #     # --- Orthogonal deconfounding (Pearl-style backdoor)
    #     E_z_unit = F.normalize(E_z, dim=-1)
    #     M_deconf = M - (torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit)

    #     # --- Normalize (all branches)
    #     E_z_norm = F.normalize(E_z, dim=-1)
    #     E_v_norm = F.normalize(E_v, dim=-1)
    #     M_norm   = F.normalize(M_deconf, dim=-1)

    #     # --- Causal fusion (Mixture-of-Experts, fully learnable)
    #     visual_causal_fusion = self.causal_fusion(E_z_norm, E_v_norm, M_norm, Q_emb)
        
    #     '''
    #      # --- Question-guided gating
    #     # v_graph_gated = v_graph_targeted * Q_emb_drop
    #     # motion_gated  = motion_targeted  * Q_emb_drop
    #     # causal_gated  = visual_causal_fusion 
    #     '''
        
    #     # ─────────────────────────────────────────────
    #     # 1. Residual Question-Guided Gating (LEARNABLE)
    #     # ─────────────────────────────────────────────
       
    #     q_gate = torch.sigmoid(self.q_gate_proj(Q_emb_drop))

    #     v_graph_gated = v_graph_targeted * (1.0 + 0.3 * q_gate)
    #     motion_gated  = motion_targeted  * (1.0 + 0.3 * q_gate)
    #     causal_gated  = visual_causal_fusion

    #     def stable_norm(x):
    #         return x / (x.norm(dim=-1, keepdim=True) + 1e-6) * x.norm(dim=-1, keepdim=True).detach().clamp(0.8, 1.2)

    #     v_graph_gated = stable_norm(v_graph_gated)
    #     motion_gated  = stable_norm(motion_gated)
    #     causal_gated  = stable_norm(causal_gated)

    #     scales = 1.0 + 0.3 * torch.tanh(self.branch_gate_mlp(Q_emb_drop))

    #     v_graph_gated = scales[:, [0]] * v_graph_gated
    #     motion_gated  = scales[:, [1]] * motion_gated
    #     causal_gated  = scales[:, [2]] * causal_gated

    #     v_joint = torch.cat([v_graph_gated, motion_gated, causal_gated], dim=-1)
    #     v_joint = self.pre_fusion_mlp(v_joint)

    #     final_representation = torch.cat([v_joint, Q_emb_drop, E_cl_proj], dim=-1)

    #     final_representation = F.layer_norm(final_representation, final_representation.shape[-1:])
    #     final_representation = F.dropout(final_representation, p=self.final_drop, training=self.training)

    #     causal_logits = self.msvd_output(final_representation)
                
        
    #     return {
    #         "causal_logits": causal_logits,
            
        
    #     }
    def compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
        B = int(batch_idx.max()) + 1  

        # 1. Expected Visual Prior (E_v)
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
        
        # Triplet-level attention
        q_expanded = q_graph[edge_batch].unsqueeze(1)             
        scale = q_graph.size(-1) ** 0.5
        attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale  
        attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
        alpha = torch.softmax(attn_logits / 0.5, dim=-1)                
        
        m_weighted = v_edges * alpha.unsqueeze(-1)                
        m_edge = m_weighted.sum(dim=1) # [Num_Edges, Dim]
        
        q_graph_edges = q_graph[edge_batch]                      # [Num_Edges, Dim]
        edge_scores = (m_edge * q_graph_edges).sum(dim=-1)       # [Num_Edges]
        
        # Safe Per-Graph Softmax
        edge_scores_max, _ = scatter_max(edge_scores, edge_batch, dim=0, dim_size=B)
        exp_scores = torch.exp(edge_scores - edge_scores_max[edge_batch])
        sum_exp_scores = scatter_sum(exp_scores, edge_batch, dim=0, dim_size=B) + 1e-6
        edge_weights = exp_scores / sum_exp_scores[edge_batch]   # [Num_Edges]
        
        M = scatter_sum(m_edge * edge_weights.unsqueeze(-1), edge_batch, dim=0, dim_size=B)
        

        return E_z, M, E_v
    
    
    def forward(self, data) -> dict:
        # ── Unpack 
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
        t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
        Q_emb_raw, Opt_emb, qtype_idx = data["question_emb"], data["options_emb"], data["qtype_idx"].squeeze(-1)
        
        Q_emb = self.lang_proj(Q_emb_raw) 
        
        # ── Setup Motion Features
        motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
        motion_feat = F.layer_norm(motion_feat, (motion_feat.size(-1),))
        
        lengths = lengths.squeeze(-1)
        batch_max_T = lengths.max().item() 
        motion_feat = motion_feat[:, :batch_max_T, :] 
        B = motion_feat.size(0)
        motion_mask = torch.arange(batch_max_T, device=motion_feat.device).expand(B, batch_max_T) < lengths.unsqueeze(1)
        
        motion_feat_proj = self.motion_proj(motion_feat)
        motion_targeted = self.question_motion_crossattentionpooling(Q_emb, motion_feat_proj, motion_mask)

        # ── Stage 1: ST-Graph Forward
        s_attr = self.edge_embed(s_attr_ids)
        t_attr = self.edge_embed(t_attr_ids)
        
        h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
            
        if self.training:
            h = F.dropout(h, p=self.h_drop, training=True)
            
        for block in self.blocks:
            h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)
            
        node_feat = h            

        q_aware_nodes = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)

        # ── Stage 2: Causal Mechanism
        if self.training:
            num_graphs = int(batch_idx.max().item()) + 1
            self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=num_graphs)

        # v_graph_pooled_for_causal = scatter_add(q_aware_nodes, batch_idx, dim=0, dim_size=B)
        
        # E_z, M, E_v = self.compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_pooled_for_causal)
        # E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
        # E_cl_proj = self.ecl_proj(E_cl)
        
        # E_z = self.ez_proj(E_z)
        # E_z = self.causal_ln_ez(E_z)  
        # M   = self.causal_ln_m(M)    
        # E_v = self.causal_ln_ev(E_v)
        # E_z_unit = F.normalize(E_z, dim=-1)
        
        # projection = torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit
        # M_deconf = M - projection

        # causal_targeted = self.causal_fusion(E_z, E_v, M_deconf, Q_emb) 
        
    
         # ── Stage 3: Late Interaction MaxSim Scoring ──
        # causal_expanded = causal_targeted[batch_idx] 
        motion_expanded = motion_targeted[batch_idx] 
        # E_cl_expanded   =  E_cl_proj[batch_idx]
        
        final_node_repr = q_aware_nodes + motion_expanded #
        
        # final_node_repr = q_aware_nodes + motion_expanded
        
        node_logits_proj = self.visual_to_text_proj(final_node_repr) 
        
        Opt_emb_proj = self.option_projection(Opt_emb)
        if Opt_emb_proj.dim() == 2:
            Opt_emb_proj = Opt_emb_proj.view(B, -1, Opt_emb_proj.size(-1))
        
        # ── FIX 1: Balance Magnitudes ──
        opt_mag = Opt_emb_proj.norm(dim=-1, keepdim=True).detach()
        node_mag = node_logits_proj.norm(dim=-1).mean().detach()
        Opt_emb_proj = Opt_emb_proj * (node_mag / (opt_mag + 1e-6))

        # ── GPU-Parallelized MaxSim (Contiguous Batch Assumption) ──
        N, D = node_logits_proj.size()
        num_options = Opt_emb_proj.size(1)

        node_counts = torch.bincount(batch_idx, minlength=B)
        max_nodes = node_counts.max().item()

        # Fast offset calculation (Relies on [0,0,0, 1,1, 2,2] structure)
        offsets = torch.cat([
            torch.zeros(1, device=batch_idx.device, dtype=torch.long),
            node_counts.cumsum(0)[:-1]
        ])
        local_idx = torch.arange(N, device=batch_idx.device) - offsets[batch_idx]

        # Zero padding + Boolean Mask (Prevents any chance of NaNs)
        padded_nodes = torch.zeros((B, max_nodes, D), device=node_logits_proj.device, dtype=node_logits_proj.dtype)
        padded_nodes[batch_idx, local_idx] = node_logits_proj

        node_mask = torch.zeros(B, max_nodes, dtype=torch.bool, device=batch_idx.device)
        node_mask[batch_idx, local_idx] = True

        # Safe Temperature
        safe_temp = torch.clamp(self.logit_temperature, min=self.temp_clamp_min)
        
        # Compute similarity and mask out padding slots
        sim_matrix = torch.matmul(padded_nodes, Opt_emb_proj.transpose(1, 2)) / safe_temp
        sim_matrix = sim_matrix.masked_fill(~node_mask.unsqueeze(-1), float('-inf'))

        # MaxSim extraction
        causal_logits = sim_matrix.max(dim=1).values 

        empty_mask = (node_counts == 0)
        if empty_mask.any():
            causal_logits[empty_mask] = 0.0

        # # ── CLEAN DIAGNOSTIC ──
        # if self.training and torch.rand(1).item() < 0.02:
        #     with torch.no_grad():
        #         if not torch.isnan(causal_logits).any():
        #             logits_flat = causal_logits.view(-1)
        #             print(f"✅ MaxSim Healthy | Mean: {logits_flat.mean().item():6.2f} | "
        #                   f"Range: {logits_flat.max().item() - logits_flat.min().item():6.2f} | "
        #                   f"Temp: {safe_temp.item():.3f}")
        #         else:
        #             print("🚨 NaN detected outside of MaxSim (Check Causal/Motion modules)")

        return {  
            "causal_logits":  causal_logits,     
        }


 