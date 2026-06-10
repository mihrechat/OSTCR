from torch import nn
from .graph_layer import GraphTransformerLayer
from .anchor import AnchorAttention
import torch

class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, edge_dim: int, num_anchors: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.spatial_layer = GraphTransformerLayer(dim, num_heads, edge_dim, dropout=dropout)
        self.temporal_layer = GraphTransformerLayer(dim, num_heads, edge_dim, dropout=dropout)
        self.anchor_attn = AnchorAttention(dim, num_heads, num_anchors, dropout=dropout)
        
        # ── Block-level gating
        self.gate_spatial = nn.Parameter(torch.tensor(0.5))
        self.gate_temporal = nn.Parameter(torch.tensor(0.5))
        self.gate_anchor = nn.Parameter(torch.tensor(0.5))
        

        self.ln_out = nn.LayerNorm(dim)
    
    def forward(
        self,
        h: torch.Tensor,        # (N, dim)
        s_idx: torch.Tensor,    # (2, E_s) spatial edges
        s_attr: torch.Tensor,   # (E_s, edge_dim) spatial edge features
        t_idx: torch.Tensor,    # (2, E_t) temporal edges
        t_attr: torch.Tensor,   # (E_t, edge_dim) temporal edge features
    ):
        
        h_in = h
        s_attr_in = s_attr
        t_attr_in = t_attr
        
        # ── Phase 1: Spatial message passing (object relations)
        h_spatial, s_attr = self.spatial_layer(h, s_idx, s_attr)
        h = h_in + torch.sigmoid(self.gate_spatial) * (h_spatial - h_in)
        
        # ── Phase 2: Temporal message passing (tracking)
        h_temporal, t_attr = self.temporal_layer(h, t_idx, t_attr)
        h = h + torch.sigmoid(self.gate_temporal) * (h_temporal - h)
        
        # ── Phase 3: Global reasoning via anchors
        h_anchor = self.anchor_attn(h)
        h = h + torch.sigmoid(self.gate_anchor) * (h_anchor - h)
        
        # ── Final layer normalization
        h = self.ln_out(h)
        
        return h, s_attr, t_attr