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
        self.edge_embed = nn.Embedding(cfg.model.num_preds, self.edge_dim)
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(self.dim, cfg.model.num_heads, self.edge_dim, cfg.model.num_anchors)
            for _ in range(cfg.model.num_layers)
        ])
        self.out_norm = nn.LayerNorm(self.dim)
        self.proj_head = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2), nn.GELU(), nn.Dropout(cfg.model.dropout), nn.Linear(self.dim * 2, cfg.model.proj_dim)
        )

     
        self.question_crossattention_pooling = QuestionGuidedPooling(text_dim=cfg.model.text_dim, node_dim=self.dim) 
        self.motion_proj = nn.Linear(cfg.model.motion_dim, cfg.model.dim)
        self.question_motion_crossattentionpooling = QuestionGuidedMotionPooling(model_dim=cfg.model.dim)
        self.lang_proj = nn.Linear(cfg.model.text_dim, cfg.model.dim)
        self.msvd_output = nn.Sequential(
            nn.Linear(cfg.model.dim * 3, cfg.model.dim * 2),
            nn.GELU(),
            nn.Dropout(0.3), 
            nn.Linear(cfg.model.dim * 2, cfg.model.msvd_vocab_size) 
        ) 
        self.ez_proj  = nn.Linear(cfg.model.concept_dim, cfg.model.dim) # 1024 -> 512
        self.ecl_proj = nn.Linear(cfg.model.concept_dim, cfg.model.dim) # 1024 -> 512


    def forward(self, data) -> dict:
        # ── Unpack 
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
        t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
        Q_emb_raw, qtype_idx = data["question_emb"], data["qtype_idx"].squeeze(-1)
        
        Q_emb = self.lang_proj(Q_emb_raw) 
        
        if self.training:
            Q_emb_drop = F.dropout(Q_emb, p=self.q_emb_drop, training=True)
        else:
            Q_emb_drop = Q_emb
        
        # ── Stage 1: ST-Graph Forward
        s_attr = self.edge_embed(s_attr_ids)
        t_attr = self.edge_embed(t_attr_ids)
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
            
        for block in self.blocks:
            h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)

        node_feat = self.out_norm(h)               
        node_feat_proj = self.proj_head(node_feat) 
        
        # INJECTION 1: Question-Guided Graph Pooling 
        v_graph_targeted = self.question_crossattention_pooling(Q_emb, node_feat, batch_idx)

        # Cross-Modal Alignment 
        if self.q_drop_cross:
            v_graph_gated = v_graph_targeted * Q_emb_drop
            motion_gated  = motion_targeted * Q_emb_drop
        else:
            v_graph_gated = v_graph_targeted * Q_emb
            motion_gated  = motion_targeted * Q_emb
  
        final_representation = torch.cat([
            v_graph_gated,         
            motion_gated,          
            Q_emb_drop,                        
        ], dim=-1)                 
        final_representation = F.dropout(final_representation, p=self.final_dop, training=self.training)
        causal_logits = self.msvd_output(final_representation) 
        return {
            "node_feat":      node_feat,
            "node_feat_proj": node_feat_proj,      
            "causal_logits":  causal_logits,  
        }