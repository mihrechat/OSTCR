from torch import nn
from .multihead_attention import MultiHeadAttentionLayer
import torch

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
        
        self.ln_node_mha = nn.LayerNorm(dim)
        self.ln_node_ffn = nn.LayerNorm(dim)
        self.ln_edge_mha = nn.LayerNorm(edge_dim)
        self.ln_edge_ffn = nn.LayerNorm(edge_dim)
        
        # ── FFNs
        self.ffn_node = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        
        self.ffn_edge = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * 2, edge_dim),
            nn.Dropout(dropout),
        )
    
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
        
        h_in = h
        e_in = edge_attr
        
        # ── Phase 1: Multi-Head Attention with Residuals
        h_attn, e_attn = self.mha(
            self.ln_node_mha(h),
            edge_index,
            self.ln_edge_mha(edge_attr) if edge_attr is not None else None
        )
        
        h = h_in + torch.sigmoid(self.gate_mha) * (h_attn - h_in)
        if e_in is not None:
            e = e_in + torch.sigmoid(self.gate_edge_mha) * (e_attn - e_in)
        else:
            e = e_attn
        
        # ── Phase 2: Feed-Forward Network with Residuals
        h_ffn = self.ffn_node(self.ln_node_ffn(h))
        h = h + torch.sigmoid(self.gate_ffn) * h_ffn  
        
        if e is not None:
            e_ffn = self.ffn_edge(self.ln_edge_ffn(e))
            e = e + torch.sigmoid(self.gate_edge_ffn) * e_ffn
        
        return h, e