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
from models.multiEncoder.cross_encoder import QuestionGuidedPooling, QuestionGuidedMotionPooling, QuestionAwareNodeEncoder
from .moe_fusion import QuestionGatedCausalFusion
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank,ConceptNetExpectedPriorBank
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
        self.q_drop   = cfg.train.q_drop
        self.h_drop   = cfg.train.h_drop
        self.final_dop  = cfg.train.final_drop

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
        self.conceptnet_prior = ConceptNetExpectedPriorBank(cfg.data.z_path) # E_z (Node-level Backdoor)
        self.conceptnet_triplets = ConceptNetTripletBank(cfg.data.triplet_path)    # M (Edge-level Frontdoor)

        # ── Cross-Attention Projections for Mediator M
        self.W_q = nn.Linear(cfg.model.mediator_dim, cfg.model.attn_dim)
        self.W_k = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        self.W_v = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        
        # ── Linguistic Prior Bank for E_cl
        self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(qtype_pt_path=cfg.data.qtype, qrole_pt_path=cfg.data.qrole)
        

        # self.question_crossattention_pooling = QuestionGuidedPooling(text_dim=cfg.model.text_dim, node_dim=self.dim) 
        self.question_crossattention_pooling = QuestionAwareNodeEncoder(text_dim=self.dim, node_dim=self.dim) 
        self.question_motion_crossattentionpooling = QuestionGuidedMotionPooling(model_dim=cfg.model.dim)
        self.motion_proj = nn.Linear(cfg.model.motion_dim, cfg.model.dim)
        
        # nn.init.xavier_uniform_(self.motion_proj.weight, gain=1.5)  
        # if self.motion_proj.bias is not None:
        #     nn.init.zeros_(self.motion_proj.bias)
    
                
        
        # ── Option Projection for Contrastive Scoring
        self.option_projection = nn.Linear(cfg.model.text_dim, self.dim)
        
        self.lang_proj  =   nn.Linear(cfg.model.text_dim, self.dim)
        # self.visual_to_text_proj = nn.Sequential(
        #     nn.Linear(cfg.model.dim * 5, cfg.model.dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(cfg.model.dim * 2, cfg.model.dim) # Projects back down to match Opt_emb
        # )
        self.visual_to_text_proj = nn.Sequential(
            nn.Linear(cfg.model.dim, cfg.model.dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # self.visual_to_text_proj = nn.ModuleList([
        #     MultiHeadAttentionLayer(
        #         in_dim=cfg.model.dim, 
        #         out_dim=cfg.model.dim, 
        #         num_heads=4,          # Keep heads small to save memory
        #         edge_dim=cfg.model.edge_dim # Inject your graph structure one last time!
        #         ) for _ in range(1)       # Just 1 layer is enough
        #         ])
        self.causal_fusion = QuestionGatedCausalFusion(text_dim=self.dim, out_dim=self.dim)
        self.ez_proj  = nn.Linear(cfg.model.concept_dim, cfg.model.dim) # 1024 -> 512
        self.ecl_proj = nn.Linear(cfg.model.concept_dim, cfg.model.dim)
        
        self.causal_ln_ez = nn.LayerNorm(self.dim) # Use 1024 because E_z is Shape: [16, 1024]
        self.causal_ln_m  = nn.LayerNorm(self.dim)  # M is [16, 512]
        self.causal_ln_ev = nn.LayerNorm(self.dim)  # E_v is [16, 512]
        self.logit_temperature = nn.Parameter(torch.tensor(1.0)) 
        # Prevent division by zero or negative temperatures which cause -inf -> NaN
        self.register_buffer("temp_clamp_min", torch.tensor(0.01)) 
                        
        
       
    # def compute_causal_signals(self, cls_id, s_idx, batch_idx, v_graph_targeted) -> tuple:
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

        v_graph_pooled_for_causal = scatter_add(q_aware_nodes, batch_idx, dim=0, dim_size=B)
        
        E_z, M, E_v = self.compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_pooled_for_causal)
        E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
        E_cl_proj = self.ecl_proj(E_cl)
        
        E_z = self.ez_proj(E_z)
        E_z = self.causal_ln_ez(E_z)  
        M   = self.causal_ln_m(M)    
        E_v = self.causal_ln_ev(E_v)
        E_z_unit = F.normalize(E_z, dim=-1)
        
        projection = torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit
        M_deconf = M - projection

        causal_targeted = self.causal_fusion(E_z, E_v, M_deconf, Q_emb) 
        
         # ── Stage 3: Late Interaction MaxSim Scoring ──
        causal_expanded = causal_targeted[batch_idx] 
        motion_expanded = motion_targeted[batch_idx] 
        # E_cl_expanded   =  E_cl_proj[batch_idx]
        
        final_node_repr = q_aware_nodes + motion_expanded + causal_expanded 
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


        return {  
            "causal_logits":  causal_logits,     
        }

        
    # def forward(self, data) -> dict:
    #     # ── Unpack 
    #     node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
    #     node_raw = F.layer_norm(node_raw, (node_raw.size(-1),))
    #     node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
    #     node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
    #     s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
    #     t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
    #     Q_emb_raw, Opt_emb, qtype_idx = data["question_emb"], data["options_emb"], data["qtype_idx"].squeeze(-1)
        
    #     Q_emb = self.lang_proj(Q_emb_raw) 
        
    #     # if self.training:
    #     #     Q_emb_drop = F.dropout(Q_emb, p=self.q_drop, training=True) 
    #     # else:
    #     #     Q_emb_drop = Q_emb
        
    #     # ── Setup Motion Features
    #     motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
    #     motion_feat = F.layer_norm(motion_feat, (motion_feat.size(-1),))
        
    #     lengths = lengths.squeeze(-1)
    #     batch_max_T = lengths.max().item() 
    #     motion_feat = motion_feat[:, :batch_max_T, :] 
    #     B = motion_feat.size(0)
    #     motion_mask = torch.arange(batch_max_T, device=motion_feat.device).expand(B, batch_max_T) < lengths.unsqueeze(1)
        
    #     motion_feat_proj = self.motion_proj(motion_feat)
    #     motion_targeted = self.question_motion_crossattentionpooling(Q_emb, motion_feat_proj, motion_mask)

    #     # ── Stage 1: ST-Graph Forward
    #     s_attr = self.edge_embed(s_attr_ids)
    #     t_attr = self.edge_embed(t_attr_ids)
        
    #     h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
            
    #     if self.training:
    #         h = F.dropout(h, p=self.h_drop, training=True)
            
    #     for i, block in enumerate(self.blocks):
    #         h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)
            
    #     node_feat = h            

    #     # Factual Extraction (Visual Graph)
    #     q_aware_nodes = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)

    #     # ── Stage 2: Causal Mechanism
    #     if self.training:
    #         num_graphs = int(batch_idx.max().item()) + 1
    #         self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=num_graphs)

    #     E_z, M, E_v = self.compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_targeted)
    #     E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
    #     E_cl_proj = self.ecl_proj(E_cl)
        
    #     E_z = self.ez_proj(E_z)
        
    #     E_z = self.causal_ln_ez(E_z)  
    #     M   = self.causal_ln_m(M)    
    #     E_v = self.causal_ln_ev(E_v)
    #     E_z_unit = F.normalize(E_z, dim=-1)
        
    #     # Gram-Schmidt Orthogonalization (Deconfounding)
    #     projection = torch.sum(M * E_z_unit, dim=-1, keepdim=True) * E_z_unit
    #     M_deconf = M - projection


    #     causal_targeted = self.causal_fusion(E_z, E_v, M_deconf, Q_emb) 
        
    #     causal_expanded = causal_targeted[batch_idx] # (N, dim)
    #     motion_expanded = motion_targeted[batch_idx] # (N, dim)
    #     final_node_repr = q_aware_nodes + causal_expanded + motion_expanded
    #     node_logits_proj = self.visual_to_text_proj(final_node_repr) # (N, text_dim)
    #     Opt_emb_proj = self.option_projection(Opt_emb)
    #     B = int(batch_idx.max()) + 1
    #     num_options = Opt_emb_proj.size(1)
    #     causal_logits = []
    #     for b in range(B):
    #         # Get nodes for this specific video
    #         mask = (batch_idx == b)
    #         video_nodes = node_logits_proj[mask]  # (Num_nodes_in_video, text_dim)
            
    #         # Get options for this specific video
    #         video_opts = Opt_emb_proj[b]          # (num_options, text_dim)
            
    #         # Compute similarity matrix: (Num_nodes, num_options)
    #         # Each node "votes" for which option it supports most
    #         sim_matrix = torch.matmul(video_nodes, video_opts.transpose(0, 1)) / self.logit_temperature
            
    #         # MAXSIM: Take the maximum similarity score for each option across ALL nodes
    #         # If even ONE node strongly matches "Frisbee", that option gets a high score
    #         max_sim_scores, _ = sim_matrix.max(dim=0) # (num_options,)
    
    #         causal_logits.append(max_sim_scores)
    #     causal_logits = torch.stack(causal_logits)
    #     return {  
    #         "causal_logits":  causal_logits,     
    #     }
        # Combine Factual Evidence with Causal Logic ──
        # final_representation = torch.cat([
        #     v_graph_targeted,         
        #     motion_targeted,          
        #     causal_targeted,          
        #     Q_emb_drop,             
        #     E_cl_proj               
        # ], dim=-1)                 
        
        # final_representation = F.dropout(final_representation, p=self.final_dop, training=self.training)
        
        # ── Contrastive Option Scoring ──
        # Opt_emb_proj = self.option_projection(Opt_emb) 
        
        # if Opt_emb_proj.dim() == 2:
        #     num_options = Opt_emb_proj.size(0) // B 
        #     Opt_emb_proj = Opt_emb_proj.view(B, num_options, -1)
        # else:
        #     num_options = Opt_emb_proj.size(1)
        
        # fused_proj = self.visual_to_text_proj(final_representation) # (B, dim)
        
        # ========================================================
        # 🔬 DIAGNOSTIC: MULTI-MODAL ALIGNMENT & SCALING CHECK
        # ========================================================
        # if self.training and torch.rand(1).item() < 0.05:
        #     with torch.no_grad():
        #         print("\n" + "="*60)
        #         print("🔬 DIAGNOSTIC: COMPONENT SCALING & ALIGNMENT")
        #         print("="*60)
                
        #         # Helper to safely print stats
        #         def stats(tensor, name):
        #             # Ensure we are looking at a 2D (B, dim) tensor
        #             t = tensor.view(tensor.size(0), -1).float()
        #             mag = t.norm(dim=-1).mean().item()
        #             var = t.var().item()
        #             mean_dir = t.mean(dim=0).norm().item() # Where is the average vector pointing?
        #             print(f"  {name:<20}: Mag={mag:6.3f} | Var={var:7.4f} | MeanDir={mean_dir:6.3f}")
                
        #         stats(v_graph_targeted,   "v_graph_targeted")
        #         stats(motion_targeted,    "motion_targeted")
        #         stats(causal_targeted,"fused_causal")
        #         stats(Q_emb_drop,       "Q_emb_drop")
        #         stats(E_cl_proj,          "E_cl_proj")
        #         print("-" * 60)
        #         stats(final_representation,"FINAL_REP (Concat)")
                
        #         fused_proj_scaled = self.visual_to_text_proj(final_representation)
        #         stats(fused_proj_scaled, "fused_proj (Pre-Score)")
        #         stats(Opt_emb_proj.view(B, -1), "Opt_emb (Flattened)")
                
        #         # Check Logit scale to see if temperature is in the right ballpark
        #         dummy_logits = torch.sum(fused_proj_scaled.unsqueeze(1) * Opt_emb_proj, dim=-1)
        #         print("-" * 60)
        #         print(f"  Raw Logit Mean: {dummy_logits.mean().item():8.3f}")
        #         print(f"  Raw Logit Std:  {dummy_logits.std().item():8.3f}")
        #         print(f"  Temperature:    {self.logit_temperature.item():8.3f}")
        #         print("="*60 + "\n")

        # fused_proj = F.normalize(fused_proj, dim=-1)
        # Opt_emb_proj = F.normalize(Opt_emb_proj, dim=-1)
        # fused_expanded = fused_proj.unsqueeze(1).expand(-1, num_options, -1) # (B, num_options, dim)
        
        # # causal_logits = torch.sum(fused_expanded * Opt_emb_proj, dim=-1) / math.sqrt(fused_expanded.size(-1))
        # causal_logits = torch.sum(fused_expanded * Opt_emb_proj, dim=-1) / self.logit_temperature
        
        # return {  
        #     "causal_logits":  causal_logits,     
        # }
