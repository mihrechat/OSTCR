
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AnchorAttention(nn.Module):
#     """
#     Anchor bottleneck attention for long-range dependencies:
#       1) Anchor tokens attend over all nodes (pool global summaries)
#       2) Nodes attend over updated anchors (broadcast global context)

#     Complexity: O(N*A) instead of O(N^2), with A << N.
#     """

#     def __init__(self, dim: int, num_heads: int, num_anchors: int = 8, dropout: float = 0.1):
#         super().__init__()
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"
#         self.dim = dim
#         self.num_heads = num_heads
#         self.dk = dim // num_heads
#         self.num_anchors = num_anchors

#         # learnable anchors
#         self.anchor_tokens = nn.Parameter(torch.randn(num_anchors, dim) * 0.02)

#         # ── projections (separate for the two directions) ─────────────
#         # anchors attending to nodes
#         self.q_anchor_pool = nn.Linear(dim, dim, bias=False)
#         self.k_node_pool   = nn.Linear(dim, dim, bias=False)
#         self.v_node_pool   = nn.Linear(dim, dim, bias=False)

#         # nodes attending to anchors
#         self.q_node_read   = nn.Linear(dim, dim, bias=False)
#         self.k_anchor_read = nn.Linear(dim, dim, bias=False)
#         self.v_anchor_read = nn.Linear(dim, dim, bias=False)

#         self.out_proj = nn.Linear(dim, dim, bias=False)

#         self.norm_anchor = nn.LayerNorm(dim)
#         self.norm_node   = nn.LayerNorm(dim)

#         self.dropout = nn.Dropout(dropout)

#         self.ffn = nn.Sequential(
#             nn.Linear(dim, 4 * dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(4 * dim, dim),
#             nn.Dropout(dropout),
#         )
#         self.norm_anchor_pool = nn.LayerNorm(dim)
#         self.norm_node_pool = nn.LayerNorm(dim)
#         self.norm_node_read = nn.LayerNorm(dim)
#         self.norm_anchor_read = nn.LayerNorm(dim)
        
#         self.norm_ffn = nn.LayerNorm(dim)

#     def forward(self, h: torch.Tensor) -> torch.Tensor:
#         """
#         h: (N, dim)
#         returns: (N, dim)
#         """
#         N = h.shape[0]
#         H = self.num_heads
#         dk = self.dk
#         A = self.num_anchors
        

#         anchors = self.norm_anchor(self.anchor_tokens)  # (A, dim)
#         h = self.norm_node_pool(h)
        

#         # ============================================================
#         # 1) Anchor pooling: anchors attend over all nodes
#         #    anchors_updated = anchors + Attn(Q=anchors, K/V=nodes)
#         # ============================================================
#         q_a = self.q_anchor_pool(anchors).view(A, H, dk)  # (A,H,dk)
#         k_n = self.k_node_pool(h).view(N, H, dk)          # (N,H,dk)
#         v_n = self.v_node_pool(h).view(N, H, dk)          # (N,H,dk)

#         # attn weights: (A,H,N)
#         attn_a2n = torch.einsum("ahd,nhd->ahn", q_a, k_n) / (dk ** 0.5)
#         attn_a2n = torch.softmax(attn_a2n, dim=-1)
#         attn_a2n = self.dropout(attn_a2n)

#         # pooled anchor content: (A,H,dk) -> (A,dim)
#         anchor_agg = torch.einsum("ahn,nhd->ahd", attn_a2n, v_n).reshape(A, -1)
#         anchors_updated = self.norm_anchor(anchors + anchor_agg)  # (A,dim)

#         # ============================================================
#         # 2) Anchor broadcast: nodes attend over updated anchors
#         #    h = h + Attn(Q=nodes, K/V=anchors_updated)
#         # ============================================================
#         anchors_updated = self.norm_anchor_read(anchors_updated)
#         h = self.norm_node_read(h)
#         q_n = self.q_node_read(h).view(N, H, dk)                   # (N,H,dk)
#         k_a = self.k_anchor_read(anchors_updated).view(A, H, dk)   # (A,H,dk)
#         v_a = self.v_anchor_read(anchors_updated).view(A, H, dk)   # (A,H,dk)

#         # attn weights: (N,H,A)
#         attn_n2a = torch.einsum("nhd,ahd->nha", q_n, k_a) / (dk ** 0.5)
#         attn_n2a = torch.softmax(attn_n2a, dim=-1)
#         attn_n2a = self.dropout(attn_n2a)

#         # broadcast node content: (N,H,dk) -> (N,dim)
#         node_agg = torch.einsum("nha,ahd->nhd", attn_n2a, v_a).reshape(N, -1)

#         # ============================================================
#         # 3) Residual + norm + FFN
#         # ============================================================
#         h = self.norm_node(h + self.out_proj(node_agg))
#         h = h + self.ffn(self.norm_ffn(h))

#         return h
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AnchorAttention(nn.Module):
    """
    FIXED: Anchor bottleneck attention with:
    - Two-stage bi-directional pooling (YOURS - GOOD)
    - Separate pool/read projections (YOURS - GOOD)
    - FFN for feature transformation (YOURS - GOOD)
    - Proper residual gating (NEW - CRITICAL FIX)
    - Score clamping to prevent divergence (NEW - CRITICAL FIX)
    - Reduced LayerNorm redundancy (NEW - CRITICAL FIX)
    - Attention dropout for regularization (NEW - CRITICAL FIX)
    
    Complexity: O(N*A) instead of O(N^2), where A << N
    """

    def __init__(self, dim: int, num_heads: int, num_anchors: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.dk = dim // num_heads
        self.num_anchors = num_anchors
        self.dropout_rate = dropout

        # ── Learnable anchor tokens (FIXED: Proper initialization)
        self.anchor_tokens = nn.Parameter(torch.empty(num_anchors, dim))
        nn.init.xavier_uniform_(self.anchor_tokens)  # FIXED: Xavier instead of 0.02 scale
        
        # ── Stage 1: Anchor Pooling (anchors attend over nodes)
        self.q_anchor_pool = nn.Linear(dim, dim, bias=False)
        self.k_node_pool = nn.Linear(dim, dim, bias=False)
        self.v_node_pool = nn.Linear(dim, dim, bias=False)
        
        nn.init.xavier_uniform_(self.q_anchor_pool.weight, gain=self.dk ** -0.5)
        nn.init.xavier_uniform_(self.k_node_pool.weight, gain=self.dk ** -0.5)
        
        # ── Stage 2: Node Reading (nodes attend over updated anchors)
        self.q_node_read = nn.Linear(dim, dim, bias=False)
        self.k_anchor_read = nn.Linear(dim, dim, bias=False)
        self.v_anchor_read = nn.Linear(dim, dim, bias=False)
        
        nn.init.xavier_uniform_(self.q_node_read.weight, gain=self.dk ** -0.5)
        nn.init.xavier_uniform_(self.k_anchor_read.weight, gain=self.dk ** -0.5)
        
        # ── Output projections
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
        # ── FIXED: Reduced LayerNorms from 6 to 3 + 1 for final output
        self.ln_anchor_in = nn.LayerNorm(dim)      # 1: Input normalization for anchor pool
        self.ln_node_in = nn.LayerNorm(dim)        # 2: Input normalization for node read
        self.ln_anchor_post_pool = nn.LayerNorm(dim)  # 3: After pooling
        
        # ── FFN for feature transformation (kept from your design)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.ln_ffn = nn.LayerNorm(dim)
        
        # ── FIXED: Learned residual gates (crucial for stability)
        self.gate_pool = nn.Parameter(torch.tensor(0.5))      # Stage 1 residual gate
        self.gate_read = nn.Parameter(torch.tensor(0.5))      # Stage 2 residual gate
        self.gate_ffn = nn.Parameter(torch.tensor(0.5))       # FFN residual gate
        
        # ── Dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, dim) node features
        Returns:
            (N, dim) updated node features
        """
        N = h.shape[0]
        H = self.num_heads
        dk = self.dk
        A = self.num_anchors
        
        h_in = h  # Save for final residual
        
        # ============================================================
        # STAGE 1: Anchor Pooling — anchors aggregate node information
        # ============================================================
        
        # FIXED: Normalize inputs before attention
        anchors = self.ln_anchor_in(self.anchor_tokens)  # (A, dim)
        h_pool = self.ln_node_in(h)                      # (N, dim)
        
        # Project to multi-head
        q_a = rearrange(self.q_anchor_pool(anchors), "a (h d) -> a h d", h=H)  # (A, H, dk)
        k_n = rearrange(self.k_node_pool(h_pool), "n (h d) -> n h d", h=H)     # (N, H, dk)
        v_n = rearrange(self.v_node_pool(h_pool), "n (h d) -> n h d", h=H)     # (N, H, dk)
        
        # Compute attention scores
        attn_scores_a2n = torch.einsum("ahd,nhd->ahn", q_a, k_n) / (dk ** 0.5)  # (A, H, N)
        
        # FIXED: Clamp scores to prevent divergence
        attn_scores_a2n = torch.clamp(attn_scores_a2n, min=-10.0, max=10.0)
        
        # Softmax over nodes for each anchor
        attn_a2n = torch.softmax(attn_scores_a2n, dim=-1)  # (A, H, N)
        
        # FIXED: Apply dropout to attention weights
        attn_a2n = self.attn_dropout(attn_a2n)
        
        # Aggregate node values into anchors
        anchor_agg = torch.einsum("ahn,nhd->ahd", attn_a2n, v_n)  # (A, H, dk)
        anchor_agg = rearrange(anchor_agg, "a h d -> a (h d)")    # (A, dim)
        
        # FIXED: Gated residual blend instead of hard addition
        anchors_updated = anchors + torch.sigmoid(self.gate_pool) * (anchor_agg - anchors)
        anchors_updated = self.ln_anchor_post_pool(anchors_updated)  # (A, dim)
        
        # ============================================================
        # STAGE 2: Node Reading — nodes extract info from updated anchors
        # ============================================================
        
        # Project to multi-head
        q_n = rearrange(self.q_node_read(h_pool), "n (h d) -> n h d", h=H)                  # (N, H, dk)
        k_a = rearrange(self.k_anchor_read(anchors_updated), "a (h d) -> a h d", h=H)       # (A, H, dk)
        v_a = rearrange(self.v_anchor_read(anchors_updated), "a (h d) -> a h d", h=H)       # (A, H, dk)
        
        # Compute attention scores
        attn_scores_n2a = torch.einsum("nhd,ahd->nha", q_n, k_a) / (dk ** 0.5)  # (N, H, A)
        
        # FIXED: Clamp scores
        attn_scores_n2a = torch.clamp(attn_scores_n2a, min=-10.0, max=10.0)
        
        # Softmax over anchors for each node
        attn_n2a = torch.softmax(attn_scores_n2a, dim=-1)  # (N, H, A)
        
        # FIXED: Apply dropout
        attn_n2a = self.attn_dropout(attn_n2a)
        
        # Aggregate anchor values into nodes
        node_agg = torch.einsum("nha,ahd->nhd", attn_n2a, v_a)  # (N, H, dk)
        node_agg = rearrange(node_agg, "n h d -> n (h d)")      # (N, dim)
        
        # FIXED: Apply output projection and gated residual
        node_agg = self.out_proj(node_agg)
        h = h_in + torch.sigmoid(self.gate_read) * (node_agg - h_in)  # (N, dim)
        
        # ============================================================
        # STAGE 3: Feed-Forward Network for feature mixing
        # ============================================================
        
        h_ffn = self.ffn(self.ln_ffn(h))
        h = h + torch.sigmoid(self.gate_ffn) * h_ffn
        
        return h
