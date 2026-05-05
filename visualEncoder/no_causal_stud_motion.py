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
import math
from torch_scatter import scatter_mean
     

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
        self.visual_to_text_proj = nn.Sequential(
            nn.Linear(cfg.model.dim * 3, cfg.model.dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.model.dim * 2, cfg.model.dim) # Projects back down to match Opt_emb
        )
        
        self.option_projection = nn.Linear(cfg.model.text_dim, self.dim)

    
    def forward(self, data) -> dict:
        # ── Unpack 
        node_raw, bbox, cls_id, conf = data["node_raw"], data["node_bbox"], data["node_class"], data["node_conf"]
        node_obj_ids_flat, node_is_keyframe = data["node_obj_ids_flat"], data["node_is_keyframe"]
        node_kf_list_idx, batch_idx = data["node_kf_list_idx"], data["batch"]
        s_idx, s_attr_ids = data["s_idx"], data["s_attr"]
        t_idx, t_attr_ids = data["t_idx"], data["t_attr"]
        Q_emb, Opt_emb, qtype_idx = data["question_emb"], data["options_emb"], data["qtype_idx"].squeeze(-1)
        
        # [NEW] Unpack Motion Features
        motion_feat, lengths = data["motion_feat"], data["num_temporal_slots"]
        
        # ── Setup Question Embedding
        Q_emb_proj = self.lang_proj(Q_emb)  # (B, cfg.dim)
        if self.training:
            Q_emb_drop = F.dropout(Q_emb_proj, p=self.q_emb_drop, training=True)
        else:
            Q_emb_drop = Q_emb_proj
        
        # ── Stage 1: Motion Processing
        lengths = lengths.squeeze(-1)
        batch_max_T = lengths.max().item() 
        motion_feat = motion_feat[:, :batch_max_T, :] 
        B = motion_feat.size(0)
        motion_mask = torch.arange(batch_max_T, device=motion_feat.device).expand(B, batch_max_T) < lengths.unsqueeze(1)
        
        motion_feat_proj = self.motion_proj(motion_feat)
        motion_targeted = self.question_motion_crossattentionpooling(Q_emb_proj, motion_feat_proj, motion_mask)

        # ── Stage 1: ST-Graph Forward
        s_attr = self.edge_embed(s_attr_ids)
        t_attr = self.edge_embed(t_attr_ids)

        h = self.node_encoder(node_raw, bbox, cls_id, conf, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
        if self.training:
            h = F.dropout(h, p=self.h_drop, training=True)
            
        for block in self.blocks:
            h, s_attr, t_attr = block(h, s_idx, s_attr, t_idx, t_attr)

        node_feat = self.out_norm(h)               
        node_feat_proj = self.proj_head(node_feat) 
        
        # Factual Extraction (Visual Graph)
        v_graph_targeted = self.question_crossattention_pooling(Q_emb_proj, node_feat, batch_idx)

        # ── Stage 2: Causal Mechanism
        
        # ── JOINT EXPECTATION FOR NWGM ──
        final_representation = torch.cat([
            v_graph_targeted,         
            motion_targeted,                    
            Q_emb_drop,                        
        ], dim=-1)  # Shape becomes (B, dim * 3)
        
        final_representation = F.dropout(final_representation, p=self.final_dop, training=self.training)
        
        # ── Contrastive Option Scoring (Specific to Multi-Choice STUDTraffic) ──
        Opt_emb_proj = self.option_projection(Opt_emb) # Expecting (B, Num_Options, dim)
        
        # Handle shape safety depending on how your options are batched
        if Opt_emb_proj.dim() == 2:
            num_options = Opt_emb_proj.size(0) // B 
            Opt_emb_proj = Opt_emb_proj.view(B, num_options, -1)
        else:
            num_options = Opt_emb_proj.size(1)
        
        # Project the combined (dim*5) vector down to (dim) to match the text options
        fused_proj = self.visual_to_text_proj(final_representation) # (B, dim)
        
        # Expand to match the options
        fused_expanded = fused_proj.unsqueeze(1).expand(-1, num_options, -1) # (B, 4, dim)
        
        # Scaled Weighted Dot-Product for Contrastive Logits
        causal_logits = torch.sum(fused_expanded * Opt_emb_proj, dim=-1) / math.sqrt(fused_expanded.size(-1))

        return {
            "node_feat":      node_feat,
            "node_feat_proj": node_feat_proj,      
            "causal_logits":  causal_logits,      
        }
        
     