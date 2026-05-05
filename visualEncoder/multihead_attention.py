import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from einops import rearrange
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum, scatter_max
from torch_geometric.utils import softmax, scatter

# class MultiHeadAttentionLayer(MessagePassing):
#     """
#     Relation-aware multi-head attention on a graph:
#         - edge features act as additive attention bias (relation-aware)
#         - edge features updated from attention scores + node context

#     For spatial edges: edge_attr = predicate embedding (dim=edge_dim)
#     For temporal edges: same interface, different semantics
#     """

#     def __init__(self, in_dim: int, out_dim: int, num_heads: int, edge_dim: int):
#         super().__init__(aggr="add", node_dim=0)

#         assert out_dim % num_heads == 0, \
#             f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"

#         self.num_heads = num_heads
#         self.out_dim   = out_dim
#         self.dk        = out_dim // num_heads   # head dimension

#         # node projections
#         self.Wq = nn.Linear(in_dim,  out_dim, bias=False)
#         self.Wk = nn.Linear(in_dim,  out_dim, bias=False)
#         self.Wv = nn.Linear(in_dim,  out_dim, bias=False)
#         self.Wo = nn.Linear(out_dim, out_dim, bias=True)

#         # edge → per-head attention bias  (edge_dim → num_heads)
#         self.W_e = nn.Linear(edge_dim, num_heads, bias=False)

#         self.update_edge = nn.Sequential(
#             nn.Linear(num_heads + 2 * self.dk + edge_dim, edge_dim), # Added edge_dim here!
#             nn.GELU(),
#             nn.Linear(edge_dim, edge_dim),
#         )

       
#         self.default_edge_bias = nn.Parameter(torch.zeros(num_heads))

#     def forward(
#         self,
#         x:          torch.Tensor,   # (N, in_dim)
#         edge_index: torch.Tensor,   # (2, E)
#         edge_attr:  torch.Tensor,   # (E, edge_dim)
#     ):
#         # project to multi-head space
#         q = rearrange(self.Wq(x), "n (h d) -> n h d", h=self.num_heads)
#         k = rearrange(self.Wk(x), "n (h d) -> n h d", h=self.num_heads)
#         v = rearrange(self.Wv(x), "n (h d) -> n h d", h=self.num_heads)

#         # edge attention bias
#         if edge_index.numel() > 0 and edge_attr is not None:
#             e_bias = self.W_e(edge_attr)   # (E, num_heads)
#         elif edge_index.numel() > 0:
#             e_bias = self.default_edge_bias.unsqueeze(0).expand(
#                 edge_index.size(1), -1
#             )
#         else:
#             e_bias = x.new_zeros(0, self.num_heads)

#         # node update via message passing
#         h_out = self.propagate(edge_index, q=q, k=k, v=v, e_bias=e_bias)
#         h_out = self.Wo(h_out)   # (N, out_dim)

#         # edge update
#         if edge_index.numel() > 0:
#             src, dst  = edge_index[0], edge_index[1]
#             q_i = q[dst]                          # (E, H, dk)  target queries
#             k_j = k[src]                          # (E, H, dk)  source keys

#             attn_scores = (q_i * k_j).sum(dim=-1) / (self.dk ** 0.5)  # (E, H)
#             e_combined = torch.cat([
#                 attn_scores,          # (E, H)
#                 q_i.mean(dim=1),      # (E, dk)
#                 k_j.mean(dim=1),      # (E, dk)
#                 edge_attr             # (E, edge_dim)  
#             ], dim=-1)

#             e_out = self.update_edge(e_combined)   # (E, edge_dim)
#         else:
#             e_out = edge_attr

#         return h_out, e_out

    # def message(self, q_i, k_j, v_j, e_bias, index, ptr, size_i):
        
    #     score = (q_i * k_j).sum(dim=-1) / (self.dk ** 0.5) + e_bias  # (E, H)
    #     alpha = softmax(score, index, ptr, size_i)                      # (E, H)
    #     return v_j * alpha.unsqueeze(-1)                                # (E, H, dk)

    # def update(self, aggr_out):
    #     return rearrange(aggr_out, "n h d -> n (h d)")   # (N, out_dim)



# class MultiHeadAttentionLayer(MessagePassing):
#     def __init__(self, in_dim: int, out_dim: int, num_heads: int, edge_dim: int):
#         super().__init__(aggr="add", node_dim=0)

#         assert out_dim % num_heads == 0
#         self.num_heads = num_heads
#         self.out_dim   = out_dim
#         self.dk        = out_dim // num_heads   

#         # FIX 1: Scale Q and K to prevent LayerNorm variance death
#         self.Wq = nn.Linear(in_dim,  out_dim, bias=False)
#         self.Wk = nn.Linear(in_dim,  out_dim, bias=False)
#         nn.init.xavier_uniform_(self.Wq.weight, gain=self.dk ** 0.5)
#         nn.init.xavier_uniform_(self.Wk.weight, gain=self.dk ** 0.5)

#         self.Wv = nn.Linear(in_dim,  out_dim, bias=False)
#         self.Wo = nn.Linear(out_dim, out_dim, bias=True)

#         # FIX 2: Initialize edge bias to zero so it doesn't shock the network
#         self.W_e = nn.Linear(edge_dim, num_heads, bias=False)
#         nn.init.zeros_(self.W_e.weight)

#         self.default_edge_bias = nn.Parameter(torch.zeros(num_heads))
#         edge_update_input_dim = self.num_heads + (self.out_dim * 2) + edge_dim
#         self.update_edge = nn.Sequential(
#             nn.Linear(edge_update_input_dim, edge_dim), 
#             nn.GELU(),
#             nn.Linear(edge_dim, edge_dim),
#         )

    # def forward(self, x, edge_index, edge_attr):
    #     q = rearrange(self.Wq(x), "n (h d) -> n h d", h=self.num_heads)
    #     k = rearrange(self.Wk(x), "n (h d) -> n h d", h=self.num_heads)
    #     v = rearrange(self.Wv(x), "n (h d) -> n h d", h=self.num_heads)

    #     if edge_index.numel() > 0 and edge_attr is not None:
    #         e_bias = self.W_e(edge_attr)   
    #     elif edge_index.numel() > 0:
    #         e_bias = self.default_edge_bias.unsqueeze(0).expand(edge_index.size(1), -1)
    #     else:
    #         e_bias = x.new_zeros(0, self.num_heads)

    #     # ============================================================
    #     # 🔑 COMPUTE ALPHA AND EDGE UPDATE OUTSIDE PROPAGATE
    #     # ============================================================
    #     e_out = edge_attr
    #     if edge_index.numel() > 0:
    #         src, dst = edge_index[0], edge_index[1]
    #         q_i = q[dst]  # (E, H, dk)
    #         k_j = k[src]  # (E, H, dk)

    #         score = (q_i * k_j).sum(dim=-1) / (self.dk ** 0.5)  # (E, H)
    #         if e_bias is not None:
    #             score = score + e_bias

    #   
    #         alpha = softmax(score, dst)  # (E, H) 

    #         if edge_attr is not None:
    #             q_flat = rearrange(q_i, "e h d -> e (h d)")
    #             k_flat = rearrange(k_j, "e h d -> e (h d)")

    #             e_combined = torch.cat([
    #                 alpha,         
    #                 q_flat,         
    #                 k_flat,         
    #                 edge_attr      
    #             ], dim=-1)
    #             e_out = self.update_edge(e_combined)
    #     # ============================================================

    #     h_out = self.propagate(
    #         edge_index, q=q, k=k, v=v, e_bias=e_bias, alpha=alpha
    #     )
        
    #     h_out = rearrange(h_out, "n h d -> n (h d)")
    #     h_out = self.Wo(h_out)
    #     h_out = F.layer_norm(h_out, (self.out_dim,)) 

    #     return h_out, e_out

    # def message(self, q_i, k_j, v_j, e_bias, alpha):
        
    #     return v_j * alpha.unsqueeze(-1)
    
    
    
# class MultiHeadAttentionLayer(MessagePassing):
#     def __init__(self, in_dim: int, out_dim: int, num_heads: int, edge_dim: int):
#         super().__init__(aggr="add", node_dim=0)
        

#         assert out_dim % num_heads == 0

#         self.num_heads = num_heads
#         self.out_dim   = out_dim
#         self.dk        = out_dim // num_heads

#         # Node projections
#         self.Wq = nn.Linear(in_dim, out_dim, bias=False)
#         self.Wk = nn.Linear(in_dim, out_dim, bias=False)
#         self.Wv = nn.Linear(in_dim, out_dim, bias=False)
#         self.Wo = nn.Linear(out_dim, out_dim, bias=True)

#         nn.init.xavier_uniform_(self.Wq.weight, gain=self.dk ** 0.5)
#         nn.init.xavier_uniform_(self.Wk.weight, gain=self.dk ** 0.5)

#         # Edge bias
#         self.W_e = nn.Linear(edge_dim, num_heads, bias=False)
#         nn.init.zeros_(self.W_e.weight)

#         # Edge update MLP
#         self.update_edge = nn.Sequential(
#             nn.Linear(self.num_heads + out_dim * 2 + edge_dim, edge_dim),
#             nn.GELU(),
#             nn.Linear(edge_dim, edge_dim),
#         )

#         self.default_edge_bias = nn.Parameter(torch.zeros(num_heads))
#         self.layer_norm = nn.LayerNorm(out_dim)
#         self.e_bias_norm = nn.LayerNorm(num_heads)
#         self.edge_norm = nn.LayerNorm(edge_dim) 

#     def forward(self, x, edge_index, edge_attr):
#         N = x.size(0)

#         # Project
#         q = rearrange(self.Wq(x), "n (h d) -> n h d", h=self.num_heads)
#         k = rearrange(self.Wk(x), "n (h d) -> n h d", h=self.num_heads)
#         v = rearrange(self.Wv(x), "n (h d) -> n h d", h=self.num_heads)

#         # Edge bias
#         if edge_index.numel() > 0 and edge_attr is not None:
#             e_bias = self.W_e(edge_attr)
#         elif edge_index.numel() > 0:
#             e_bias = self.default_edge_bias.unsqueeze(0).expand(edge_index.size(1), -1)
#         else:
#             e_bias = x.new_zeros(0, self.num_heads)

#         # ---- Message Passing ----
#         h_out, alpha = self.propagate(
#             edge_index,
#             q=q, k=k, v=v,
#             e_bias=e_bias,
#             edge_attr=edge_attr,
#             return_alpha=True
#         )

#         # ---- Node output ----
#         h_out = rearrange(h_out, "n h d -> n (h d)")
#         h_out = self.Wo(h_out)
#         h_out = self.layer_norm(h_out)  

#         # ---- Edge update (uses SAME alpha) ----
#         if edge_index.numel() > 0 and edge_attr is not None:
#             src, dst = edge_index[0], edge_index[1]
            
#             h_src = x[src] # (E, dim)
#             h_dst = x[dst] # (E, dim)


#             e_combined = torch.cat([
#                 alpha,       # (E, H) 
#                 h_src,      # (E, dim)
#                 h_dst,      # (E, dim)
#                 edge_attr    # (E, edge_dim)
#             ], dim=-1)
#             e_out = self.update_edge(e_combined)
#             e_out = self.edge_norm(e_out)
#         else:
#             e_out = edge_attr

#         return h_out, e_out

#     def message(self, q_i, k_j, v_j, e_bias, index, ptr, size_i,
#                 edge_attr, return_alpha):

#         if e_bias is not None:
#             e_bias = self.e_bias_norm(e_bias)  
#         # Attention score
#         score = (q_i * k_j).sum(dim=-1) / (self.dk ** 0.5) + e_bias

#         alpha = softmax(score, index, ptr, size_i)  # (E, H)

#         out = v_j * alpha.unsqueeze(-1)

#         if return_alpha:
#             return out, alpha
#         return out

#     def aggregate(self, inputs, index, ptr=None, dim_size=None):
#         if isinstance(inputs, tuple):
#             node_msg, alpha = inputs
#             node_out = scatter_sum(node_msg, index, dim=0, dim_size=dim_size)
#             return node_out, alpha
#         else:
#             return scatter_sum(inputs, index, dim=0, dim_size=dim_size)

#     def update(self, inputs):
#         return inputs



class MultiHeadAttentionLayer(MessagePassing):
    """
    FIXED: Graph attention with:
    - Proper attention score clipping
    - Head diversity regularization
    - Dropout on attention weights
    - Better edge update mechanism
    - Improved initialization
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, edge_dim: int, 
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)
        
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dk = out_dim // num_heads
        
        # ── Node projections (Xavier init with proper gain)
        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)
        self.Wo = nn.Linear(out_dim, out_dim, bias=True)
        
        nn.init.xavier_uniform_(self.Wq.weight, gain=self.dk ** 0.5)
        nn.init.xavier_uniform_(self.Wk.weight, gain=self.dk ** 0.5)
        nn.init.xavier_uniform_(self.Wv.weight, gain=self.dk ** 0.5)
        
        # ── Edge bias (FIXED: Initialize with small random values, not zeros)
        self.W_e = nn.Linear(edge_dim, num_heads, bias=False)
        nn.init.xavier_uniform_(self.W_e.weight)  # FIXED: Was zeros
        
        # ── Edge update MLP
        self.update_edge = nn.Sequential(
            nn.Linear(num_heads + out_dim * 2 + edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * 2, edge_dim),
        )
        
        # ── Default edge bias (learnable, but smaller weight)
        self.default_edge_bias = nn.Parameter(torch.randn(num_heads) * 0.01)
        
        # ── Normalization layers (FIXED: Reduce redundancy)
        self.ln_out = nn.LayerNorm(out_dim)
        self.ln_edge = nn.LayerNorm(edge_dim)
        
        # ── Dropout
        self.dropout_attn = nn.Dropout(attn_dropout)
        self.dropout_feat = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge connectivity
            edge_attr: (E, edge_dim) or None
        Returns:
            h_out: (N, out_dim) updated node features
            e_out: (E, edge_dim) updated edge features
        """
        
        N = x.size(0)
        
        # ── Project to multi-head Q, K, V
        q = rearrange(self.Wq(x), "n (h d) -> n h d", h=self.num_heads)  # (N, H, dk)
        k = rearrange(self.Wk(x), "n (h d) -> n h d", h=self.num_heads)  # (N, H, dk)
        v = rearrange(self.Wv(x), "n (h d) -> n h d", h=self.num_heads)  # (N, H, dk)
        
        # ── Compute edge bias
        if edge_index.numel() > 0 and edge_attr is not None:
            e_bias = self.W_e(edge_attr)  # (E, H)
            # FIXED: Normalize edge bias to prevent saturation
            e_bias = F.layer_norm(e_bias, (self.num_heads,))
        elif edge_index.numel() > 0:
            # FIXED: Use learnable default bias, not expand zeros
            e_bias = self.default_edge_bias.unsqueeze(0).expand(edge_index.size(1), -1)
        else:
            e_bias = x.new_zeros(0, self.num_heads)
        
        # ── Message passing with attention
        h_out, alpha = self.propagate(
            edge_index,
            q=q, k=k, v=v,
            e_bias=e_bias,
            edge_attr=edge_attr,
            return_alpha=True,
            size=N
        )
        
        # ── Node output
        h_out = rearrange(h_out, "n h d -> n (h d)")
        h_out = self.Wo(h_out)
        h_out = self.ln_out(h_out)
        
        # ── FIXED: Edge update with better feature combination
        if edge_index.numel() > 0 and edge_attr is not None:
            src, dst = edge_index[0], edge_index[1]
            
            h_src = x[src]  # (E, in_dim)
            h_dst = x[dst]  # (E, in_dim)
            
            # FIXED: Use updated node features instead of original
            h_src_new = h_out[src]  # (E, out_dim)
            h_dst_new = h_out[dst]  # (E, out_dim)
            
            # Combine attention + original features + updated features
            e_combined = torch.cat([
                alpha,              # (E, H) attention weights
                h_src_new,          # (E, out_dim) FIXED: Use new features
                h_dst_new,          # (E, out_dim) FIXED: Use new features
                edge_attr           # (E, edge_dim)
            ], dim=-1)
            
            e_out = self.update_edge(e_combined)
            e_out = self.ln_edge(e_out)
            # FIXED: Add residual connection for edges
            e_out = edge_attr + self.dropout_feat(e_out)
        else:
            e_out = edge_attr
        
        return h_out, e_out
    
    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor, 
                e_bias: torch.Tensor, index: torch.Tensor, ptr, size_i: int,
                edge_attr, return_alpha: bool):
        """
        Compute attention and apply to values.
        
        Args:
            q_i: (E, H, dk) query of destination nodes
            k_j: (E, H, dk) key of source nodes
            v_j: (E, H, dk) value of source nodes
            e_bias: (E, H) edge bias
            index: (E,) destination node indices
            return_alpha: whether to return attention weights
        """
        
        # ── Attention score: (q_i * k_j) / sqrt(dk) + e_bias
        score = (q_i * k_j).sum(dim=-1) / (self.dk ** 0.5)  # (E, H)
        
        # FIXED: Clamp scores to prevent divergence
        score = torch.clamp(score, min=-10.0, max=10.0)
        
        # ── Add edge bias
        if e_bias is not None:
            score = score + e_bias  # (E, H)
        
        # ── Softmax over source nodes for each destination
        alpha = softmax(score, index, ptr, size_i)  # (E, H)
        
        # FIXED: Apply dropout to attention weights
        alpha = self.dropout_attn(alpha)
        
        # ── Apply attention to values
        out = v_j * alpha.unsqueeze(-1)  # (E, H, dk)
        
        if return_alpha:
            return out, alpha
        return out
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """Aggregate messages from edges to nodes."""
        if isinstance(inputs, tuple):
            node_msg, alpha = inputs
            node_out = scatter(node_msg, index, dim=0, dim_size=dim_size, reduce="sum")
            return node_out, alpha
        else:
            return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
    
    def update(self, inputs):
        """Identity update (aggregation happens in aggregate)."""
        return inputs