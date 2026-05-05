from torch import nn
from .graph_layer import GraphTransformerLayer
from .anchor import AnchorAttention
import torch


# class SpatioTemporalBlock(nn.Module):
#     """
#     One full ST-Graph processing step:
#         1. Spatial layer   — message passing over relation edges
#         2. Temporal layer  — message passing over "follows" tracking edges
#         3. Anchor attention — global cross-object reasoning

#     Both graph layers use Pre-LN + residual (handled inside the layer).
#     Block-level residual adds skip connection around the full block.
#     """

#     def __init__(self, dim: int, num_heads: int, edge_dim: int, num_anchors: int = 8):
#         super().__init__()

#         self.spatial_layer  = GraphTransformerLayer(dim, num_heads, edge_dim)
#         self.temporal_layer = GraphTransformerLayer(dim, num_heads, edge_dim)
#         self.anchor_attn    = AnchorAttention(dim, num_heads, num_anchors)

#         # gating: learned interpolation between input and output
#         # prevents degradation in deep stacks
#         self.gate = nn.Parameter(torch.tensor(0.0))

#     def forward(
#         self,
#         h:      torch.Tensor,   # (N, dim)
#         s_idx:  torch.Tensor,   # (2, E_s)
#         s_attr: torch.Tensor,   # (E_s, edge_dim)
#         t_idx:  torch.Tensor,   # (2, E_t)
#         t_attr: torch.Tensor,   # (E_t, edge_dim)
#     ):
#         h_in = h

#         # spatial message passing (RelTR relations)
#         h, s_attr = self.spatial_layer(h, s_idx, s_attr)

#         # temporal message passing (tracking edges)
#         h, t_attr = self.temporal_layer(h, t_idx, t_attr)

#         # global reasoning via anchors
#         h = self.anchor_attn(h)

#         h = h_in + torch.sigmoid(self.gate) * (h - h_in)

#         return h, s_attr, t_attr


class SpatioTemporalBlock(nn.Module):
    """
    FIXED: Complete ST-Graph processing step with:
    - Better gate initialization (0.5 → meaningful updates from start)
    - Proper edge feature propagation
    - Block-level residual connection
    - Layer-wise feature normalization
    """
    
    def __init__(self, dim: int, num_heads: int, edge_dim: int, num_anchors: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.spatial_layer = GraphTransformerLayer(dim, num_heads, edge_dim, dropout=dropout)
        self.temporal_layer = GraphTransformerLayer(dim, num_heads, edge_dim, dropout=dropout)
        self.anchor_attn = AnchorAttention(dim, num_heads, num_anchors, dropout=dropout)
        
        # ── Block-level gating
        self.gate_spatial = nn.Parameter(torch.tensor(0.5))
        self.gate_temporal = nn.Parameter(torch.tensor(0.5))
        self.gate_anchor = nn.Parameter(torch.tensor(0.5))
        
        # ── Final normalization
        self.ln_out = nn.LayerNorm(dim)
    
    def forward(
        self,
        h: torch.Tensor,        # (N, dim)
        s_idx: torch.Tensor,    # (2, E_s) spatial edges
        s_attr: torch.Tensor,   # (E_s, edge_dim) spatial edge features
        t_idx: torch.Tensor,    # (2, E_t) temporal edges
        t_attr: torch.Tensor,   # (E_t, edge_dim) temporal edge features
    ):
        """
        Returns:
            h: (N, dim) updated node features
            s_attr: (E_s, edge_dim) updated spatial edge features
            t_attr: (E_t, edge_dim) updated temporal edge features
        """
        
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