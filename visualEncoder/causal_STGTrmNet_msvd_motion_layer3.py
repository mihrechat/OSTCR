
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
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
from models.multiEncoder.cross_encoder import QuestionGuidedPooling, QuestionGuidedMotionPooling, QuestionAwareNodeEncoder
from .moe_fusion import QuestionGatedCausalFusion
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank,ConceptNetExpectedPriorBank, CausalPriorBank
import math
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
        self.dropout = cfg.model.dropout
        self.dim = cfg.model.dim
        self.raw_dim = cfg.model.raw_dim
        self.edge_dim = cfg.model.edge_dim
        self.q_emb_drop = cfg.train.q_drop
        self.final_dop =  cfg.train.final_drop
        self.h_drop    = cfg.train.h_drop
        self.q_drop_cross = cfg.train.q_emb_drop_cross

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
        self.causal_prior_bank = CausalPriorBank(cfg.model.prior_pt_path, cfg.model.num_classes, self.dim)

        # ── Cross-Attention Projections for Mediator M
        # self.W_q = nn.Linear(cfg.model.mediator_dim, cfg.model.attn_dim)
        # self.W_k = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        # self.W_v = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        
        self.subj_emb = nn.Embedding(cfg.model.num_classes, self.dim)
        self.obj_emb  = nn.Embedding(cfg.model.num_classes, self.dim)
        self.rel_emb  = nn.Embedding(cfg.model.num_preds, self.dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

       
        self.W_q = nn.Linear(cfg.model.mediator_dim, cfg.model.attn_dim)
        self.W_k = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        self.W_v = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        
        # ── Linguistic Prior Bank for E_cl
        self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(qtype_pt_path=cfg.model.qtype_pt_path, qrole_pt_path=cfg.model.qrole_pt_path)
        
        # ── Causal Fusion Layer
        # ── MoE Linear Fusion for Deconfounded Representation
        
        self.causal_fusion = QuestionGatedCausalFusion(text_dim=self.dim, node_dim=self.dim)
     
        # self.question_crossattention_pooling = QuestionGuidedPooling(text_dim=cfg.model.text_dim, node_dim=self.dim) 
        self.question_crossattention_pooling = QuestionAwareNodeEncoder(text_dim=self.dim, node_dim=self.dim) 
        self.motion_proj = nn.Linear(cfg.model.motion_dim, cfg.model.dim)
        self.question_motion_crossattentionpooling = QuestionGuidedMotionPooling(model_dim=cfg.model.dim)
        self.lang_proj = nn.Linear(cfg.model.text_dim, cfg.model.dim)
        self.msvd_output = nn.Sequential(
            nn.Linear(cfg.model.dim * 3, cfg.model.dim * 2),
            nn.GELU(),
            nn.Dropout(0.1), 
            nn.Linear(cfg.model.dim * 2, cfg.model.msvd_vocab_size) 
        ) 
        self.causal_attention = nn.MultiheadAttention(embed_dim=cfg.model.dim, num_heads=cfg.model.num_heads, batch_first=True)
        self.ez_proj  = nn.Linear(cfg.model.concept_dim, self.dim) # 1024 -> 512
        self.ecl_proj = nn.Linear(cfg.model.concept_dim, self.dim) # 1024 -> 512
        # self.causal_film = nn.Linear(self.dim, self.dim * 2)
    
    
    # def _compute_causal_signals(self, cls_id, s_idx, edge_attr,batch_idx, v_graph_targeted) -> tuple:
        
    #     B = v_graph_targeted.size(0)
    #     temp= 0.7
    #     # 1. Expected Visual Prior (E_v) -> KEEP THIS GLOBAL (Correct Causal Math)
    #     E_v_global = self.visual_memory.get_expected_visual()  
    #     E_v = E_v_global.expand(B, -1)                         
     
    #     E_z_nodes = self.causal_prior_bank(cls_id)
    #     E_z = global_mean_pool(E_z_nodes, batch_idx) 
        
        
    #         # embeddings
        
    #     assert B == int(batch_idx.max()) + 1
            
    #     src, dst, r = s_idx[0], s_idx[1], edge_attr
    #     edge_batch = batch_idx[src]  # (E,)
    #     if s_idx.numel() == 0:
    #         M = torch.zeros_like(v_graph_targeted)
    #         return M, E_z, E_v
        
    #     E_s = self.subj_emb(cls_id[src])
    #     E_r = self.rel_emb(r)
    #     E_o = self.obj_emb(cls_id[dst])
        

    #     m_edge = self.edge_mlp(torch.cat([E_s, E_r, E_o], dim=-1))  # (E, D)
    #     m_edge = F.normalize(m_edge, dim=-1)
        
    #     # attention
    #     q = F.normalize(self.W_q(v_graph_targeted), dim=-1)      # (B, D)
    #     k = F.normalize(self.W_k(m_edge), dim=-1)       # (E, D)

    #     edge_scores = (q[edge_batch] * k).sum(-1)  # (E,)

    #     # per-graph softmax
        
    #     edge_scores_max, _ = scatter_max(edge_scores, edge_batch, dim=0, dim_size=B)
    #     # exp_scores = torch.exp(edge_scores - edge_scores_max[edge_batch] / 0.7 )
    #     exp_scores = torch.exp((edge_scores - edge_scores_max[edge_batch]) / temp)
    #     sum_exp = scatter_sum(exp_scores, edge_batch, dim=0, dim_size=B) + 1e-6

    #     alpha = exp_scores / sum_exp[edge_batch]

    #     # aggregation
    #     M = scatter_sum(alpha.unsqueeze(-1) * m_edge,
    #                     edge_batch,
    #                     dim=0,
    #                     dim_size=B)
        
    #     M = M + 0.5 * v_graph_targeted
    #     return M, E_z,  E_v
    
    # def compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
    #     B = int(batch_idx.max()) + 1  

    #     # 1. Expected Visual Prior (E_v)
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
        
    #     # Triplet-level attention
    #     q_expanded = q_graph[edge_batch].unsqueeze(1)             
    #     scale = q_graph.size(-1) ** 0.5
    #     attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale  
    #     attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
    #     alpha = torch.softmax(attn_logits / 0.5, dim=-1)                
        
    #     m_weighted = v_edges * alpha.unsqueeze(-1)                
    #     m_edge = m_weighted.sum(dim=1) # [Num_Edges, Dim]
        
    #     q_graph_edges = q_graph[edge_batch]                      # [Num_Edges, Dim]
    #     edge_scores = (m_edge * q_graph_edges).sum(dim=-1)       # [Num_Edges]
        
    #     # Safe Per-Graph Softmax
    #     edge_scores_max, _ = scatter_max(edge_scores, edge_batch, dim=0, dim_size=B)
    #     exp_scores = torch.exp(edge_scores - edge_scores_max[edge_batch])
    #     sum_exp_scores = scatter_sum(exp_scores, edge_batch, dim=0, dim_size=B) + 1e-6
    #     edge_weights = exp_scores / sum_exp_scores[edge_batch]   # [Num_Edges]
        
    #     M = scatter_sum(m_edge * edge_weights.unsqueeze(-1), edge_batch, dim=0, dim_size=B)
        
    def compute_causal_signals(self, cls_id, s_idx, batch_idx, node_feat) -> tuple:
        """
        Pure Node-Level signals: (N, Dim)
        """
        N = node_feat.size(0)
        B = int(batch_idx.max()) + 1 

        # 1. Expected Visual Prior (E_v)
        E_v_global = self.visual_memory.get_expected_visual()  
        E_v = E_v_global.expand(N, -1)                         

        # 2. Expected Semantic Prior (E_z) 
        E_z = self.conceptnet_prior(cls_id)                    

        # 3. Graph-Conditioned Mediator (M) -> NODE LEVEL
        src, dst = s_idx[0].contiguous(), s_idx[1].contiguous()
        edge_batch = batch_idx[src]  
        
        # E_T shape is (E, 10, dim)
        E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst]) 
        
        # Mask out empty triplet slots (assuming padded with zeros)
        triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)              # (E, 10)
        
        # Project Nodes and Triplets
        q_src_nodes = self.W_q(node_feat[src])                    # (E, Dim)
        k_edges = self.W_k(E_T)                                   # (E, 10, Dim)
        v_edges = self.W_v(E_T)                                   # (E, 10, Dim)
        
        scale = q_src_nodes.size(-1) ** 0.5
        
        # FIX: Unsqueeze query to (E, 1, Dim) to match the 10 triplets
        q_unsqueezed = q_src_nodes.unsqueeze(1)                   # (E, 1, Dim)
        
        # Node-to-Triplet Attention
        attn_logits = (q_unsqueezed * k_edges).sum(dim=-1) / scale # (E, 10)
        attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
        alpha = torch.softmax(attn_logits / 0.5, dim=-1)          # (E, 10)
        
        # Weighted sum of the 10 triplets
        m_weighted = v_edges * alpha.unsqueeze(-1)                 # (E, 10, Dim)
        m_edge = m_weighted.sum(dim=1)                             # (E, Dim)  <-- Collapses the 10 back to 1
        
        # Score how relevant this aggregated edge is to its source node
        edge_scores = (m_edge * q_src_nodes).sum(dim=-1)           # (E,)
        
        # Safe Per-Graph Softmax
        edge_scores_max, _ = scatter_max(edge_scores, edge_batch, dim=0, dim_size=B)
        exp_scores = torch.exp(edge_scores - edge_scores_max[edge_batch])
        sum_exp_scores = scatter_sum(exp_scores, edge_batch, dim=0, dim_size=B) + 1e-6
        edge_weights = exp_scores / sum_exp_scores[edge_batch]    # (E,)
        
        # Scatter BACK to the specific source nodes
        M = scatter_sum(m_edge * edge_weights.unsqueeze(-1), src, dim=0, dim_size=N) # (N, Dim)

        return M, E_z, E_v



    #     return E_z, M, E_v

    # def forward(self, data) -> dict:
    #     # ── Unpack 
    #     node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
    #     node_raw = node_raw / (torch.linalg.norm(node_raw, axis=-1, keepdim=True) + 1e-6)
    #     node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
    #     motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
    #     node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
    #     s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
    #     t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
    #     Q_emb_raw, qtype_idx = data["question_emb"], data["qtype_idx"].squeeze(-1)
        
    #     Q_emb = self.lang_proj(Q_emb_raw) 
        
    #     if self.training:
    #         Q_emb_drop = F.dropout(Q_emb, p=self.q_emb_drop, training=True) # 0.3 to 0.1
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
    #     motion_feat = motion_feat / (torch.linalg.norm(motion_feat, axis=-1, keepdim=True) + 1e-6)
    #     motion_targeted = self.question_motion_crossattentionpooling(Q_emb, motion_feat, motion_mask)

    #     h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
    #     if self.training:
    #         h = F.dropout(h, p=self.h_drop, training=True)
            
    #     for block in self.blocks:
    #         h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)

    #     node_feat = h  
                  
    #     # INJECTION 1: Question-Guided Graph Pooling (Use CLEAN Q_emb!)
    #     v_graph_targeted = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)

    #     # ── Stage 2: Causal Mechanism
    #     if self.training:
    #         num_graphs = int(batch_idx.max().item()) + 1
    #         self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=num_graphs)
    #     M, E_z, E_v = self._compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, edge_attr=s_attr_ids, v_graph_targeted=v_graph_targeted)
        
        
    #     E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
        
    #     E_z = self.ez_proj(E_z)
        
    #     E_z_unit = F.normalize(E_z, dim=-1)
    #     M_deconf = M - (torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit)
    #     # --- Normalize (all branches)
    #     E_z_norm = F.normalize(E_z, dim=-1)
    #     E_v_norm = F.normalize(E_v, dim=-1)
    #     M_norm   = F.normalize(M_deconf, dim=-1)


    #     visual_causal_fusion = self.causal_fusion(E_z_norm, E_v_norm, M_norm, Q_emb) 
        
    #     E_cl_proj = self.ecl_proj(E_cl)
    #     motion_targeted = motion_targeted / (motion_targeted.norm(dim=-1, keepdim=True) + 1e-6)
    #     v_graph_targeted = v_graph_targeted / (v_graph_targeted.norm(dim=-1, keepdim=True) + 1e-6)
    #     causal_targeted = visual_causal_fusion 
        
        
    #     v_graph_targeted = F.normalize(v_graph_targeted, dim=-1)
    #     motion_targeted  = F.normalize(motion_targeted, dim=-1)
    #     causal_targeted  = F.normalize(visual_causal_fusion, dim=-1)
    #     Q_emb_drop       = F.normalize(Q_emb_drop, dim=-1)
    #     E_cl_proj        = F.normalize(E_cl_proj, dim=-1)
        
        
    #     # ── JOINT EXPECTATION FOR NWGM ──
    #     final_representation = torch.cat([
    #         v_graph_targeted,         
    #         motion_targeted,          
    #         causal_targeted,          
    #         Q_emb_drop,            
    #         E_cl_proj              
    #     ], dim=-1)       
     
        

    #     # FIX #2 APPLIED HERE: Removed duplicate lines
    #     final_representation = F.dropout(final_representation, p=self.final_dop, training=self.training) 
    #     causal_logits = self.msvd_output(final_representation) 
        
    #     return {     
    #         "causal_logits":  causal_logits,  
    #     }
    
  
    def forward(self, data) -> dict:
        # ── Unpack ──
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
        frame_idx = data["node_frame_index"]
        
        # Layer norm for raw motion
        motion_feat = F.layer_norm(motion_feat, (motion_feat.size(-1),))
        
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
        t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
        Q_emb_raw, qtype_idx = data["question_emb"], data["qtype_idx"].squeeze(-1)
        
        Q_emb = self.lang_proj(Q_emb_raw) 
        
        # ── Stage 1: ST-Graph Forward & Motion Setup ──
        s_attr = self.edge_embed(s_attr_ids)
        t_attr = self.edge_embed(t_attr_ids)
        
        lengths = lengths.squeeze(-1)
        
        true_T = motion_feat.size(1) 
        lengths = torch.clamp(lengths, max=true_T) 
        
        # SAFETY 2: Cap lengths at the maximum valid frame in THIS specific batch
        batch_max_T = lengths.max().item() 
        motion_feat = motion_feat[:, :batch_max_T, :] 
        B = motion_feat.size(0)
        
        motion_mask = torch.arange(batch_max_T, device=motion_feat.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        motion_feat_proj = self.motion_proj(motion_feat)
        
        motion_targeted_BT = self.question_motion_crossattentionpooling(Q_emb, motion_feat_proj, motion_mask)  # (B, T, dim)

        h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
        if self.training:
            h = F.dropout(h, p=self.h_drop, training=True)
            
        for block in self.blocks:
            h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)

        node_feat = h            
        q_aware_nodes = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx) # (N, Dim)

        # ── Stage 2: Causal Mechanism ──
        if self.training:
            num_graphs = int(batch_idx.max().item()) + 1
            self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=num_graphs)

        M, E_z, E_v = self.compute_causal_signals(cls_id, s_idx, batch_idx, q_aware_nodes)
        E_z = self.ez_proj(E_z)
       
        E_z_unit = F.normalize(E_z, dim=-1)
        projection = torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit
        M_deconf = M - projection

        causal_targeted = self.causal_fusion(E_z, E_v, M_deconf, Q_emb, batch_idx) # (N, Dim)
        
         # ── Stage 3: Bulletproof GPU Motion Mapping ──
        valid_frame_mask = (frame_idx >= 0) & (frame_idx < lengths[batch_idx])
        safe_frame_indices = torch.where(valid_frame_mask, frame_idx, torch.zeros_like(frame_idx))
        
        video_offsets = torch.arange(B, device=motion_feat.device) * batch_max_T
        global_node_indices = video_offsets[batch_idx] + safe_frame_indices
        
        motion_flattened = motion_targeted_BT.reshape(B * batch_max_T, -1)
        motion_flattened = torch.nan_to_num(motion_flattened, nan=0.0, posinf=0.0, neginf=0.0)
        
        motion_per_node = motion_flattened[global_node_indices]
        motion_per_node[~valid_frame_mask] = 0.0

        # ── Stage 4: Normalized Fusion & Output ──
        concat_node_repr = torch.cat([
            q_aware_nodes,      
            motion_per_node,    
            causal_targeted    
        ], dim=-1)                     
        
        concat_node_repr = F.layer_norm(concat_node_repr, (concat_node_repr.size(-1),))
        concat_node_repr = F.dropout(concat_node_repr, p=self.final_dop, training=self.training)

        node_logits = self.msvd_output(concat_node_repr) 

        # ── Stage 5: Soft-Max Aggregation (LogSumExp) ──
        max_per_batch = scatter_max(node_logits, batch_idx, dim=0, dim_size=B)[0]
        stable_logits = node_logits - max_per_batch[batch_idx]
        exp_logits = torch.exp(stable_logits)

        sum_exp = scatter_add(exp_logits, batch_idx, dim=0, dim_size=B)
        causal_logits = max_per_batch + torch.log(sum_exp + 1e-8)

        # ── Empty Graph Safety ──
        empty_mask = (torch.bincount(batch_idx, minlength=B) == 0)
        if empty_mask.any():
            causal_logits[empty_mask] = 0.0

        return {"causal_logits": causal_logits}