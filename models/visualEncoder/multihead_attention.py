import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from einops import rearrange
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum, scatter_max
from torch_geometric.utils import softmax, scatter

class MultiHeadAttentionLayer(MessagePassing):
    """
    FIXED: Graph attention with proper MessagePassing API usage.
    
    Key fixes:
    - size parameter must be tuple (num_src, num_dst)
    - Proper return_alpha handling in propagate
    - Correct edge update logic
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, edge_dim: int, 
                 dropout: float = 0.1, attn_dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)
        
        assert out_dim % num_heads == 0, f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dk = out_dim // num_heads
        
    
        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)
        self.Wo = nn.Linear(out_dim, out_dim, bias=True)
        
        nn.init.xavier_uniform_(self.Wq.weight, gain=self.dk ** 0.5)
        nn.init.xavier_uniform_(self.Wk.weight, gain=self.dk ** 0.5)
        nn.init.xavier_uniform_(self.Wv.weight, gain=self.dk ** 0.5)
        
    
        self.W_e = nn.Linear(edge_dim, num_heads, bias=False)
        nn.init.xavier_uniform_(self.W_e.weight)
        
   
        self.update_edge = nn.Sequential(
            nn.Linear(num_heads + out_dim * 2 + edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * 2, edge_dim),
        )
        
        # ── Default edge bias
        self.default_edge_bias = nn.Parameter(torch.randn(num_heads) * 0.01)
        
        # ── Normalization layers
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
            e_bias = F.layer_norm(e_bias, (self.num_heads,))
        elif edge_index.numel() > 0:
            e_bias = self.default_edge_bias.unsqueeze(0).expand(edge_index.size(1), -1)
        else:
            e_bias = x.new_zeros(0, self.num_heads)
        out = self.propagate(
            edge_index,
            q=q, k=k, v=v,
            e_bias=e_bias,
            edge_attr=edge_attr,
            size=(N, N)  
        )
        
        if isinstance(out, tuple):
            h_out, alpha = out
        else:
            h_out = out
            alpha = None
        
        # ── Node output
        h_out = rearrange(h_out, "n h d -> n (h d)")
        h_out = self.Wo(h_out)
        h_out = self.ln_out(h_out)
        
   
        if edge_index.numel() > 0 and edge_attr is not None:
            src, dst = edge_index[0], edge_index[1]
            
            h_src_new = h_out[src]  # (E, out_dim)
            h_dst_new = h_out[dst]  # (E, out_dim)
            
    
            if alpha is not None:
                e_combined = torch.cat([
                    alpha,              # (E, H) attention weights
                    h_src_new,          # (E, out_dim)
                    h_dst_new,          # (E, out_dim)
                    edge_attr           # (E, edge_dim)
                ], dim=-1)
            else:
                e_combined = torch.cat([
                    h_src_new,
                    h_dst_new,
                    edge_attr
                ], dim=-1)
            
            e_out = self.update_edge(e_combined)
            e_out = self.ln_edge(e_out)
            e_out = edge_attr + self.dropout_feat(e_out)
        else:
            e_out = edge_attr
        
        return h_out, e_out
    
    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor, 
                e_bias: torch.Tensor, index: torch.Tensor, ptr, size_i: int):
    
        # ── Attention score
        score = (q_i * k_j).sum(dim=-1) / (self.dk ** 0.5)  # (E, H)
        
 
        score = torch.clamp(score, min=-10.0, max=10.0)
        

        if e_bias is not None:
            score = score + e_bias  # (E, H)
        

        alpha = softmax(score, index, ptr, size_i)  # (E, H)
        
  
        alpha = self.dropout_attn(alpha)
        
        out = v_j * alpha.unsqueeze(-1)  # (E, H, dk)
        
        return out, alpha
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        if isinstance(inputs, tuple):
            node_msg, alpha = inputs
            node_out = scatter(node_msg, index, dim=0, dim_size=dim_size, reduce="sum")
            return node_out, alpha
        else:
            return scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
    
    def update(self, inputs):
   
        return inputs



