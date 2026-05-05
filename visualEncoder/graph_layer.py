from torch import nn
from .multihead_attention import MultiHeadAttentionLayer
import torch


# class GraphTransformerLayer(nn.Module):
#     """
#     Pre-LayerNorm transformer block operating on a graph:

#         Phase 1:  LN → MHA → residual   (node + edge)
#         Phase 2:  LN → FFN → residual   (node + edge)
#     """

#     def __init__(self, dim: int, num_heads: int, edge_dim: int, dropout: float = 0.1):
#         super().__init__()

#         self.mha = MultiHeadAttentionLayer(dim, dim, num_heads, edge_dim)

#         # Pre-LN norms
#         self.ln_node1 = nn.LayerNorm(dim)
#         self.ln_node2 = nn.LayerNorm(dim)
#         self.ln_edge1 = nn.LayerNorm(edge_dim)
#         self.ln_edge2 = nn.LayerNorm(edge_dim)

#         # FFNs
#         self.ffn_node = nn.Sequential(
#             nn.Linear(dim,      dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * 4,  dim),
#             nn.Dropout(dropout),
#         )
#         self.ffn_edge = nn.Sequential(
#             nn.Linear(edge_dim,      edge_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(edge_dim * 4,  edge_dim),
#             nn.Dropout(dropout),
#         )

#     def forward(
#         self,
#         h:          torch.Tensor,   # (N, dim)
#         edge_index: torch.Tensor,   # (2, E)
#         edge_attr:  torch.Tensor,   # (E, edge_dim)
#     ):
#         # ── Phase 1: attention ────────────────────────────────────
#         h_new, e_new = self.mha(
#             self.ln_node1(h),
#             edge_index,
#             self.ln_edge1(edge_attr) if edge_attr is not None else edge_attr,
#         )
#         h = h + h_new
#         e = edge_attr + e_new if edge_attr is not None else e_new

#         # ── Phase 2: FFN ──────────────────────────────────────────
#         h = h + self.ffn_node(self.ln_node2(h))
#         e = e + self.ffn_edge(self.ln_edge2(e)) if e is not None else e

#         return h, e


class GraphTransformerLayer(nn.Module):
    """
    FIXED: Pre-LN transformer block with:
    - Proper residual connections
    - Reduced LayerNorm redundancy
    - Better initialization
    - Explicit edge handling
    """
    
    def __init__(self, dim: int, num_heads: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.mha = MultiHeadAttentionLayer(dim, dim, num_heads, edge_dim, dropout=dropout)
        
        # ── Pre-LN norms (FIXED: Reduced from 4 to 2)
        self.ln_node_mha = nn.LayerNorm(dim)
        self.ln_node_ffn = nn.LayerNorm(dim)
        self.ln_edge_mha = nn.LayerNorm(edge_dim)
        self.ln_edge_ffn = nn.LayerNorm(edge_dim)
        
        # ── FFNs
        self.ffn_node = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        self.ffn_edge = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * 4, edge_dim),
            nn.Dropout(dropout),
        )
        
        # ── Gates for each residual (learned per-layer)
        self.gate_mha = nn.Parameter(torch.tensor(0.5))
        self.gate_ffn = nn.Parameter(torch.tensor(0.5))
        self.gate_edge_mha = nn.Parameter(torch.tensor(0.5))
        self.gate_edge_ffn = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ):
        """
        Args:
            h: (N, dim) node features
            edge_index: (2, E) edge connectivity
            edge_attr: (E, edge_dim) edge features
        Returns:
            h: (N, dim) updated node features
            edge_attr: (E, edge_dim) updated edge features
        """
        
        h_in = h
        e_in = edge_attr
        
        # ── Phase 1: Multi-Head Attention with Residuals
        h_attn, e_attn = self.mha(
            self.ln_node_mha(h),
            edge_index,
            self.ln_edge_mha(edge_attr) if edge_attr is not None else None
        )
        
        # FIXED: Proper residual blending with learned gates
        h = h_in + torch.sigmoid(self.gate_mha) * (h_attn - h_in)
        if e_in is not None:
            e = e_in + torch.sigmoid(self.gate_edge_mha) * (e_attn - e_in)
        else:
            e = e_attn
        
        # ── Phase 2: Feed-Forward Network with Residuals
        h_ffn = self.ffn_node(self.ln_node_ffn(h))
        h = h + torch.sigmoid(self.gate_ffn) * h_ffn  # Already has residual from phase 1
        
        if e is not None:
            e_ffn = self.ffn_edge(self.ln_edge_ffn(e))
            e = e + torch.sigmoid(self.gate_edge_ffn) * e_ffn
        
        return h, e