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
from .moe_fusion import QuestionGatedCausalFusion, CausalCrossAttentionFusion, CausalGatedFusion, MoELinearFusion
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
        self.q_deconf    = cfg.train.q_deconf
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
        self.motion_norm = nn.LayerNorm(self.dim)
        self.causal_norm = nn.LayerNorm(self.dim)
        
        # ── Linguistic Prior Bank for E_cl
        self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(qtype_pt_path=cfg.model.qtype_pt_path, qrole_pt_path=cfg.model.qrole_pt_path)
        
        # ── Causal Fusion Layer
        # ── MoE Linear Fusion for Deconfounded Representation
        in_dim = cfg.model.dim * 3
       
        self.causal_fusion = MoELinearFusion(
            in_dim=in_dim, 
            out_dim=self.dim, 
            num_qtypes=cfg.data.num_qtypes
        )
        
        # self.causal_fusion = QuestionGatedCausalFusion(in_dim=in_dim, text_dim=self.dim, out_dim=self.dim)
        # self.causal_fusion = CausalCrossAttentionFusion(self.dim)
        # self.fusion        = CausalGatedFusion(self.dim)
     
        self.question_crossattention_pooling = QuestionGuidedPooling(text_dim=self.dim, node_dim=self.dim) 
        self.motion_proj = nn.Linear(cfg.model.motion_dim, cfg.model.dim)
        self.question_motion_crossattentionpooling = QuestionGuidedMotionPooling(model_dim=cfg.model.dim)

        
        self.logit_temp = nn.Parameter(torch.tensor(1.0)) 
        self.logit_aux_temp = nn.Parameter(torch.tensor(1.0))
        self.q_blend_param  = nn.Parameter(torch.tensor(1.0))
       # Keep this one:
        self.motion_proj = nn.Sequential(
            nn.Linear(cfg.model.motion_dim, cfg.model.dim),
            nn.Dropout(0.1),
            nn.LayerNorm(cfg.model.dim),
            
        )
        
    
        self.ez_proj = nn.Sequential(
            nn.Linear(cfg.model.concept_dim, cfg.model.dim),
            nn.Dropout(0.1),
            nn.LayerNorm(cfg.model.dim),
       
            
        )

        self.ecl_proj = nn.Sequential(
            nn.Linear(cfg.model.concept_dim, cfg.model.dim),
            nn.Dropout(0.1),
            nn.LayerNorm(cfg.model.dim)
        )

        self.lang_proj = nn.Sequential(
            nn.Linear(cfg.model.text_dim, cfg.model.dim),
            nn.Dropout(0.1),
            nn.LayerNorm(cfg.model.dim),
            
        )

     
      
        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, 1)
        )

        # Classifier
        self.msvd_output = nn.Sequential(
            # nn.LayerNorm(self.dim*4),
            nn.Linear(self.dim*4, self.dim * 2),
            nn.GELU(),
            nn.Dropout(0,1),
            nn.Linear(self.dim * 2, cfg.model.msvd_vocab_size)
        )
        # self.proj_head = nn.Sequential(
        #     nn.Linear(self.dim, self.dim * 2), nn.GELU(), nn.Dropout(cfg.model.dropout), nn.Linear(self.dim * 2, cfg.model.proj_dim)
        # )
        
        self.gamma_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim)
        )

        self.alpha_param = nn.Parameter(torch.tensor(1.0))
        
      
    #original
    # def _compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
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
        
    #     E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst]) # [E, num_triplets(various), dim] return from self.lookup_table[src_cls_ids, dst_cls_ids]
    #     '''
    #     lookup_table = torch.zeros((num_classes, num_classes, max_triplets, dim))
        
    #     for (src_idx, dst_idx), embs in triplet_dict.items():
    #         num_avail = min(len(embs), max_triplets)
    #         lookup_table[src_idx, dst_idx, :num_avail] = embs[:num_avail]
    #     '''
    #     triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)             
        
    #     q_graph = self.W_q(v_graph)                        
    #     k_edges = self.W_k(E_T)                                   
    #     v_edges = self.W_v(E_T)                                   
        
    #     q_expanded = q_graph[edge_batch].unsqueeze(1)             
    #     scale = q_graph.size(-1) ** 0.5
    #     attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale  
    #     attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
    #     alpha = torch.softmax(attn_logits, dim=-1)                
        
    #     m_weighted = v_edges * alpha.unsqueeze(-1)                
    #     m_edge = m_weighted.sum(dim=1)                            
    #     M = scatter_add(m_edge, edge_batch, dim=0, dim_size=B)    
        
    #     return E_z, M, E_v
    
    
    # def _compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted):

    #     B = int(batch_idx.max()) + 1  

    #     # ── 1. Expected Visual Prior (E_v) ─────────────
    #     E_v_global = self.visual_memory.get_expected_visual()
    #     E_v = E_v_global.expand(B, -1)
    #     E_v = F.layer_norm(E_v, (E_v.size(-1),))   # <-- FIX


    #     # ── 2. Expected Semantic Prior (E_z) ──────────
    #     E_z_nodes = self.conceptnet_prior(cls_id)
    #     E_z = global_mean_pool(E_z_nodes, batch_idx)
    #     E_z = F.layer_norm(E_z, (E_z.size(-1),))   # <-- FIX


    #     # ── 3. Graph-conditioned mediator (M) ─────────
    #     src, dst = s_idx[0], s_idx[1]
    #     edge_batch = batch_idx[src]

    #     E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst])
    #     triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)

    #     v_graph_norm = F.layer_norm(v_graph_targeted, (v_graph_targeted.size(-1),))
    #     E_T_norm = F.layer_norm(E_T, (E_T.size(-1),))

    #     q_graph = self.W_q(v_graph_norm)
    #     k_edges = self.W_k(E_T_norm)
    #     v_edges = self.W_v(E_T_norm)

    #     q_expanded = q_graph[edge_batch].unsqueeze(1)

    #     scale = q_graph.size(-1) ** 0.5
    #     attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale
    #     attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)

    #     alpha = torch.softmax(attn_logits, dim=-1)

    #     m_weighted = v_edges * alpha.unsqueeze(-1)
    #     m_edge = m_weighted.sum(dim=1)

    #     M = scatter_mean(m_edge, edge_batch, dim=0, dim_size=B)

    #     # ── fallback handling ─────────────────────────
    #     batch_counts = torch.bincount(edge_batch, minlength=B)
    #     zero_batches = (batch_counts == 0)

    #     if zero_batches.any():
    #         with torch.no_grad():
    #             fallback = self.ez_proj(E_z)
    #             fallback = F.layer_norm(fallback, (fallback.size(-1),))  # <-- FIX
    #         M[zero_batches] = fallback[zero_batches]

    #     # final normalization
    #     M = F.layer_norm(M, (M.size(-1),))   # <-- FIX

    #     return E_z, M, E_v
    
    def _compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
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
        
        E_T = self.conceptnet_triplets(cls_id[src], cls_id[dst]) # [E, num_triplets(various), dim] return from self.lookup_table[src_cls_ids, dst_cls_ids]
        '''
        lookup_table = torch.zeros((num_classes, num_classes, max_triplets, dim))
        
        for (src_idx, dst_idx), embs in triplet_dict.items():
            num_avail = min(len(embs), max_triplets)
            lookup_table[src_idx, dst_idx, :num_avail] = embs[:num_avail]
        '''
        triplet_mask = (E_T.abs().sum(dim=-1) > 1e-6)             
        
        q_graph = self.W_q(v_graph)                        
        k_edges = self.W_k(E_T)                                   
        v_edges = self.W_v(E_T)                                   
        
        q_expanded = q_graph[edge_batch].unsqueeze(1)             
        scale = q_graph.size(-1) ** 0.5
        attn_logits = (q_expanded * k_edges).sum(dim=-1) / scale  
        attn_logits = attn_logits.masked_fill(~triplet_mask, -1e9)
        
        alpha = torch.softmax(attn_logits, dim=-1)                
        
        m_weighted = v_edges * alpha.unsqueeze(-1)                
        m_edge = m_weighted.sum(dim=1)                            
        # M = scatter_add(m_edge, edge_batch, dim=0, dim_size=B)   
        M = scatter_mean(m_edge, edge_batch, dim=0, dim_size=B)  
        
        return E_z, M, E_v

    def forward(self, data) -> dict:
        # ── Unpack 
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
        t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
        Q_emb, qtype_idx = data["question_emb"], data["qtype_idx"].squeeze(-1)
        
        Q_emb = self.lang_proj(Q_emb)  # (B, dim)
        
        # Q_norm = F.layer_norm(Q_emb, (Q_emb.size(-1),))   


        # ── Linguistic prior (confounder) ─────────────────
        E_cl = self.linguistic_prior_bank(
            qtype_idx,
            data["triplet_idxs"],
            data["triplet_mask"]
        )

        E_cl = self.ecl_proj(E_cl)  


        # ── Deconfounding (FiLM) ─────────────────────────
        gamma = self.gamma_mlp(E_cl)
        beta  = self.beta_mlp(E_cl)

        Q_deconf = gamma * Q_emb + beta
        


        # ── Blend raw + deconfounded question ────────────
        # alpha = self.alpha_mlp(Q_emb)  
        # alpha = torch.sigmoid(alpha)
        # Q_attn = alpha * Q_emb + (1 - alpha) * Q_deconf
        # # Q_attn = F.layer_norm(Q_attn, (Q_attn.size(-1),)) 
        
        # ── Stage 1: ST-Graph Forward
        s_attr = self.edge_embed(s_attr_ids)
        t_attr = self.edge_embed(t_attr_ids)
        lengths = lengths.squeeze(-1)
        batch_max_T = lengths.max().item() 
        
        motion_feat = motion_feat[:, :batch_max_T, :] 
        B = motion_feat.size(0)
        motion_mask = torch.arange(batch_max_T, device=motion_feat.device).expand(B, batch_max_T) < lengths.unsqueeze(1)
        
        # CROSS-ATTENTION (Motion)
        motion_feat = self.motion_proj(motion_feat)
        motion_targeted = self.question_motion_crossattentionpooling(Q_emb, motion_feat, motion_mask) # (B, dim)

        h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
        if self.training:
            h = F.dropout(h, p=0.1, training=True)
            
        for block in self.blocks:
            h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)

        node_feat = self.out_norm(h)     
        # node_feat  = h
                 
        # node_feat_proj = self.proj_head(node_feat) 
        
        # INJECTION 1: Question-Guided Graph Pooling (Object)
        v_graph_targeted = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)

        # ── Stage 2: Causal Mechanism
        if self.training:
            self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=data.num_graphs)
            
        E_z, M, E_v = self._compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_targeted)
        E_z = self.ez_proj(E_z)

        
        E_z_unit = F.normalize(E_z, p=2, dim=-1)
        M_deconf = M - (torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit)

        # E_z = F.normalize(E_z, dim=-1)
        # E_v = F.normalize(E_v, dim=-1)
        # M_deconf = F.normalize(M_deconf, dim=-1) 

        causal_targeted = self.causal_fusion(E_z, E_v, M_deconf, qtype_idx)
        
        
        # causal_targeted = self.causal_norm(causal_targeted)
        # motion_targeted = self.motion_norm(motion_targeted)
        
        # v_graph_targeted = v_graph_targeted * Q_deconf
        # motion_targeted  = motion_targeted * Q_deconf
        # causal_targeted  = causal_targeted * Q_deconf
        
        # if self.training:
        #  v_graph_targeted = F.dropout(v_graph_targeted,p=self.v_drop)
        #  motion_targeted  = F.dropout(motion_targeted, p=self.m_drop)
        #  causal_targeted  = F.dropout(causal_targeted,p=self.c_drop)
        #  Q_deconf         = F.dropout(Q_deconf, p=self.q_drop)
        
    
        final_representation = torch.cat([
            v_graph_targeted,       
            motion_targeted,         
            causal_targeted,     
            Q_deconf             
        ], dim=-1)         
        # fused_context, fusion_gates = self.fusion(
        #     v_graph_targeted,
        #     motion_targeted,
        #     causal_targeted,
        #     Q_emb,
        #     Q_deconf     # semantic signal
        # )        
        
        final_representation = F.dropout(final_representation, p=self.final_drop, training=self.training)
        
        # if self.training and torch.rand(1).item() < 0.01:
        #     print('#'*100)
        #     with torch.no_grad():
        #         print(f"Question Norm: {Q_emb.norm(dim=-1).mean():.2f}(std: {Q_emb.std():.3f})")
        #         print(f"Question deconf Norm: {Q_deconf.norm(dim=-1).mean():.2f}(std: {Q_deconf.std():.3f})")
        #         print(f"visual Norm: {v_graph_targeted.norm(dim=-1).mean():.2f}(std: {v_graph_targeted.std():.3f})")
        #         print(f"Motion Norm: {motion_targeted.norm(dim=-1).mean():.2f}(std: {motion_targeted.std():.3f})")
        #         print(f"Causal Norm: {causal_targeted.norm(dim=-1).mean():.2f}(std: {causal_targeted.std():.3f})")
        #         print(f"Final Norm: {final_representation.norm(dim=-1).mean():.2f}(std: {final_representation.std():.3f})")
        #         print(f"M Norm: {M_deconf.norm(dim=-1).mean():.2f}(std: {M_deconf.std():.3f})")
        #         print(f"EV Norm: {E_v.norm(dim=-1).mean():.2f}(std: {E_v.std():.3f})")
        #         print(f"Ez Norm: {E_z.norm(dim=-1).mean():.2f}(std: {E_z.std():.3f})")
                
                
               
       
        
        causal_logits = self.msvd_output(final_representation) # (B, num_answers)
        causal_logits = causal_logits / (self.logit_temp.abs() + 1e-4)
       
        return {
            # "node_feat":      node_feat,
            # "node_feat_proj": node_feat_proj,      
            "causal_logits":  causal_logits,  
        }
