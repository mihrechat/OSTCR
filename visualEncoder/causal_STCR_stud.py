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
from models.multiEncoder.cross_encoder import QuestionGuidedPooling
from .moe_fusion import MoELinearFusion
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank,ConceptNetExpectedPriorBank
import math
     


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
        self.conceptnet_prior = ConceptNetExpectedPriorBank(cfg.model.prior_pt_path) # E_z (Node-level Backdoor)
        self.conceptnet_triplets = ConceptNetTripletBank(cfg.model.triplet_pt_path)    # M (Edge-level Frontdoor)

        # ── Cross-Attention Projections for Mediator M
        self.W_q = nn.Linear(cfg.model.mediator_dim, cfg.model.attn_dim)
        self.W_k = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        self.W_v = nn.Linear(cfg.model.text_dim, cfg.model.attn_dim)
        
        # ── Linguistic Prior Bank for E_cl
        self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(qtype_pt_path=cfg.model.qtype_pt_path, qrole_pt_path=cfg.model.qrole_pt_path)
        
        # ── Causal Fusion Layer
        # self.causal_fusion = CausalFusionModule(dim=cfg.dim, num_answers=cfg.num_answers, dropout=cfg.dropout)  
        # ── MoE Linear Fusion for Deconfounded Representation
        moe_in_dim = cfg.model.concept_dim + self.dim + cfg.model.mediator_dim + cfg.model.text_dim
        moe_out_dim = self.dim 
        self.moe_linear_fusion = MoELinearFusion(
            in_dim=moe_in_dim, 
            out_dim=moe_out_dim, 
            num_qtypes=cfg.model.num_qtypes
        )
        # self.moe_linear_fusion = MoELinearFusion(dim=self.dim, num_qtypes=cfg.model.num_qtypes)  
        
        self.question_crossattention_pooling = QuestionGuidedPooling(text_dim=cfg.model.text_dim, node_dim=self.dim) 
        self.msvd_classifier = nn.Sequential(
            nn.Linear(cfg.model.dim, cfg.model.dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.model.dim, cfg.model.msvd_num_answers) # Maps your causal vector to the vocab!
        ) 
                
        
        # ── Option Projection for Contrastive Scoring
        self.option_projection = nn.Linear(cfg.model.text_dim, self.dim)
        
        self.lang_proj  =   nn.Linear(cfg.model.text_dim, self.dim)
        # self.option_scoring_head = nn.Linear(cfg.model.dim * 3, 1)
        # nn.init.xavier_uniform_(self.option_scoring_head.weight, gain=0.1)
        # nn.init.zeros_(self.option_scoring_head.bias)
        self.visual_to_text_proj = nn.Sequential(
            nn.Linear(cfg.model.dim * 3, cfg.model.dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.model.dim * 2, cfg.model.dim) # Projects back down to match Opt_emb
        )
        
        
       
       
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
        M = scatter_add(m_edge, edge_batch, dim=0, dim_size=B)    
        
        return E_z, M, E_v
    

    def forward(self, data) -> dict:
        # ── Unpack 
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
        t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
        Q_emb, Opt_emb, qtype_idx = data["question_emb"], data["options_emb"], data["qtype_idx"].squeeze(-1)
        
        
        # ── Stage 1: ST-Graph Forward
        s_attr = self.edge_embed(s_attr_ids)
        t_attr = self.edge_embed(t_attr_ids)

        h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
        if self.training:
            h = F.dropout(h, p=0.1, training=True)
            
        for block in self.blocks:
            h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)

        node_feat = self.out_norm(h)               # (N, dim)
        node_feat_proj = self.proj_head(node_feat) # (N, proj_dim)
        
        # ── Stage 1: Factual Extraction
        Q_emb_proj = self.lang_proj(Q_emb)  # (B, cfg.dim)
        v_graph_targeted = self.question_crossattention_pooling(Q_emb_proj, node_feat, batch_idx)

        # ── Stage 2: Causal Mechanism
        if self.training:
            self.visual_memory.update(node_feat=node_feat.detach(), batch_idx=batch_idx, num_graphs=data.num_graphs)

        E_z, M, E_v = self._compute_causal_signals(cls_id=cls_id, s_idx=s_idx, batch_idx=batch_idx, v_graph_targeted=v_graph_targeted)
        E_cl = self.linguistic_prior_bank(qtype_idx, data["triplet_idxs"], data["triplet_mask"])
        
        # This contains causal adjustments and deconfounding logic
        fused_causal_vector = self.moe_linear_fusion(E_z, E_v, M, E_cl, qtype_idx) # (B, dim)
       
        # Combine Factual Evidence with Causal Logic ──
        final_representation = torch.cat([
            v_graph_targeted,      # "What is physically happening in the video"
            Q_emb_proj,            # "What the question is specifically asking"
            fused_causal_vector    # "My structural/causal background knowledge"
        ], dim=-1)                 # Shape becomes (B, dim * 3)
        
        
        # ── INJECTION Contrastive Option Scoring
        Opt_emb_proj = self.option_projection(Opt_emb)
        
        B = final_representation.size(0)  
        num_options = Opt_emb_proj.size(0) // B 
        Opt_emb_proj = Opt_emb_proj.view(B, num_options, -1)
        
        # Project the combined (dim*3) vector down to (dim) to match the text options
        fused_proj = self.visual_to_text_proj(final_representation) # (B, dim)
        
        # 2. Expand to match the 4 options
        fused_expanded = fused_proj.unsqueeze(1).expand(-1, num_options, -1) # (B, 4, dim)
        
        # 3. Scaled Weighted Dot-Product 
        causal_logits = torch.sum(fused_expanded * Opt_emb_proj, dim=-1) / math.sqrt(fused_expanded.size(-1))

        return {
            "node_feat":      node_feat,
            "node_feat_proj": node_feat_proj,      
            "causal_logits":  causal_logits,      
        }
