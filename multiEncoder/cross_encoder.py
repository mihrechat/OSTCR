import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax as pyg_softmax
import torch.nn.functional as F


# class QuestionAwareNodeEncoder(nn.Module):
#     def __init__(self, text_dim: int, node_dim: int):
#         super().__init__()
#         # We fuse the question into the nodes, but KEEP them separate
#         self.fuse_proj = nn.Sequential(
#             nn.Linear(node_dim + text_dim, node_dim),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(node_dim, node_dim)
#         )

#     def forward(self, Q_emb: torch.Tensor, node_feat: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Returns: (N, node_dim) - Question-conditioned node features.
#         """
#         # Expand Q_emb to match every node
#         Q_expanded = Q_emb[batch_idx]      # (N, text_dim)
        
#         # Fuse question context INTO each node
#         fused_input = torch.cat([node_feat, Q_expanded], dim=-1) # (N, node_dim + text_dim)
#         q_aware_nodes = self.fuse_proj(fused_input)              # (N, node_dim)
        
#         return q_aware_nodes
    
# import torch
# import torch.nn as nn

# class QuestionGuidedPooling(nn.Module):
#     def __init__(self, text_dim: int, node_dim: int):
#         super().__init__()
        
#         self.q_proj = nn.Linear(text_dim, node_dim)
#         self.kv_proj = nn.Linear(node_dim, node_dim * 2)
#         self.node_dim = node_dim
        
#         self.output_scale = nn.Parameter(torch.ones(1) * 1.0)
#         self.out_norm = nn.LayerNorm(node_dim)
        
#         self.input_norm_q = nn.LayerNorm(node_dim)
#         self.input_norm_kv = nn.LayerNorm(node_dim)

#     def forward(self, Q_emb: torch.Tensor, node_feat: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
#         B = int(batch_idx.max()) + 1
        
#         Q_emb = self.input_norm_q(Q_emb)
#         node_feat = self.input_norm_kv(node_feat)
        
#         Q = self.q_proj(Q_emb)
#         KV = self.kv_proj(node_feat)
#         K, V = KV.chunk(2, dim=-1)
        
#         Q_expanded = Q[batch_idx]
#         scale = self.node_dim ** 0.5
#         attn_logits = (Q_expanded * K).sum(dim=-1) / scale
#         alpha = pyg_softmax(attn_logits, batch_idx, num_nodes=B)
#         v_weighted = V * alpha.unsqueeze(-1)
#         v_graph_targeted = scatter_add(v_weighted, batch_idx, dim=0, dim_size=B)
        
#         v_graph_targeted = self.out_norm(v_graph_targeted)
#         v_graph_targeted = v_graph_targeted * self.output_scale
        
#         return v_graph_targeted
    

# class QuestionGuidedMotionPooling(nn.Module):
#     def __init__(self, model_dim: int):
#         super().__init__()
#         self.q_proj = nn.Linear(model_dim, model_dim)  
#         self.kv_proj = nn.Linear(model_dim, model_dim * 2)
#         self.model_dim = model_dim
        
#         self.output_scale = nn.Parameter(torch.ones(1) * 1.0)
#         self.out_norm = nn.LayerNorm(model_dim)
#         self.input_norm_q = nn.LayerNorm(model_dim)
#         self.input_norm_kv = nn.LayerNorm(model_dim)

#     def forward(self, Q_emb: torch.Tensor, motion_feat: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
        
#         Q_emb = self.input_norm_q(Q_emb)
#         motion_feat = self.input_norm_kv(motion_feat)
        
#         Q = self.q_proj(Q_emb)
#         KV = self.kv_proj(motion_feat)
#         K, V = KV.chunk(2, dim=-1)
        
        
#         Q = Q.unsqueeze(1)
#         scale = Q.size(-1) ** 0.5
#         attn_logits = torch.bmm(Q, K.transpose(1, 2)) / scale
#         attn_logits = attn_logits.masked_fill(~motion_mask.unsqueeze(1), float('-inf'))
#         alpha = F.softmax(attn_logits, dim=-1)
#         motion_targeted = torch.bmm(alpha, V)
#         motion_targeted = motion_targeted.squeeze(1)
        
#         motion_targeted = self.out_norm(motion_targeted)
#         motion_targeted = motion_targeted * self.output_scale
        
        # return motion_targeted
# class QuestionGuidedPooling(nn.Module):
#     def __init__(self, text_dim: int, node_dim: int):
#         super().__init__()
        
#         self.q_proj = nn.Linear(text_dim, node_dim)

#         self.kv_proj = nn.Linear(node_dim, node_dim * 2)
        
#         self.node_dim = node_dim

#     def forward(self, Q_emb: torch.Tensor, node_feat: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Q_emb: (B, text_dim) - Question text embedding
#         node_feat: (N, node_dim) - ST-Graph node features
#         batch_idx: (N,) - Maps nodes to their respective graphs
#         Returns: (B, node_dim) - The question-aware visual vector
#         """
#         B = int(batch_idx.max()) + 1
        
#         # 1. Project Text (Query) and Nodes (Keys & Values)
#         Q = self.q_proj(Q_emb)                                     # (B, node_dim)
#         KV = self.kv_proj(node_feat)                               # (N, node_dim * 2)
#         K, V = KV.chunk(2, dim=-1)                                 # Each: (N, node_dim)

#         # 2. Expand graph-level queries to node-level for dot product
#         Q_expanded = Q[batch_idx]                                  # (N, node_dim)

#         # 3. Compute Attention Scores
#         scale = self.node_dim ** 0.5
#         attn_logits = (Q_expanded * K).sum(dim=-1) / scale         # (N,)

#         # 4. Softmax strictly over the nodes belonging to the SAME graph
#         alpha = pyg_softmax(attn_logits, batch_idx, num_nodes=B)   # (N,)

#         # 5. Weighted sum of the node values
#         v_weighted = V * alpha.unsqueeze(-1)                        # (N, node_dim)
        
#         # Aggregate back to graph level
#         v_graph_targeted = scatter_add(v_weighted, batch_idx, dim=0, dim_size=B) # (B, node_dim)
        
#         return v_graph_targeted



    
# class QuestionGuidedMotionPooling(nn.Module):
#     def __init__(self, model_dim: int):
#         super().__init__()
        
#         self.q_proj = nn.Linear(model_dim, model_dim)  
#         self.kv_proj = nn.Linear(model_dim, model_dim*2)

#     def forward(self, Q_emb: torch.Tensor, motion_feat: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Q_emb:       (B, model_dim) - Question text embedding
#         motion_feat: (B, T, model_dim) - Projected motion features
#         motion_mask: (B, T) - Boolean mask where True = valid frame, False = padded zero
        
#         Returns:     (B, model_dim) - The question-aware global motion vector
#         """
       
#         Q = self.q_proj(Q_emb)             # (B, model_dim)
#         KV = self.kv_proj(motion_feat)       # (B, T, model_dim)
#         K, V = KV.chunk(2, dim=-1)      # (B, T, model_dim)
        
#         # Unsqueeze Q to allow batched matrix multiplication against T frames
#         Q = Q.unsqueeze(1)                 # (B, 1, model_dim)

#         # 2. Compute Attention Scores: (Q * K^T) / sqrt(dim)
#         scale = Q.size(-1) ** 0.5
#         attn_logits = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B, 1, T)
        
#         # 3. Apply the Padding Mask
#         attn_logits = attn_logits.masked_fill(~motion_mask.unsqueeze(1), float('-inf'))

#         # 4. Softmax over the temporal dimension (T)
#         alpha = F.softmax(attn_logits, dim=-1)   # (B, 1, T)

#         # 5. Weighted sum of the temporal frames
#         motion_targeted = torch.bmm(alpha, V)    # (B, 1, model_dim)
#         motion_targeted = motion_targeted.squeeze(1)
        
        
#         return motion_targeted       # (B, model_dim)



# class QuestionGuidedPooling(nn.Module):
#     def __init__(self, text_dim: int, node_dim: int):
#         super().__init__()
#         # Project text to match node dimensions if they differ
#         self.q_proj = nn.Linear(node_dim, node_dim)
#         self.k_proj = nn.Linear(node_dim, node_dim)
#         self.v_proj = nn.Linear(node_dim, node_dim)

#     def forward(self, Q_emb: torch.Tensor, node_feat: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Q_emb: (B, text_dim) - Frozen question text embedding
#         node_feat: (N, node_dim) - ST-Graph node features
#         batch_idx: (N,) - Maps nodes to their respective graphs
#         Returns: (B, node_dim) - The question-aware visual vector
#         """
#         B = int(batch_idx.max()) + 1
        
#         # 1. Project Question (Queries) and Nodes (Keys, Values)
#         Q = self.q_proj(Q_emb)         # (B, node_dim)
#         K = self.k_proj(node_feat)     # (N, node_dim)
#         V = self.v_proj(node_feat)     # (N, node_dim)

#         # 2. Expand graph-level queries to node-level for dot product
#         Q_expanded = Q[batch_idx]      # (N, node_dim)

#         # 3. Compute Attention Scores: (Q * K) / sqrt(dim)
#         scale = Q.size(-1) ** 0.5
#         attn_logits = (Q_expanded * K).sum(dim=-1) / scale  # (N,)

#         # 4. Softmax strictly over the nodes belonging to the SAME graph
#         alpha = pyg_softmax(attn_logits, batch_idx, num_nodes=B)  # (N,)

#         # 5. Weighted sum of the node values
#         v_weighted = V * alpha.unsqueeze(-1)                # (N, node_dim)
        
#         # Aggregate back to graph level (B, node_dim)
#         v_graph_targeted = scatter_add(v_weighted, batch_idx, dim=0, dim_size=B)
        
#         return v_graph_targeted
    


# class QuestionGuidedMotionPooling(nn.Module):
#     def __init__(self, model_dim: int):
#         super().__init__()
#         # Everything is now model_dim because motion_feat was already projected
#         # in the main forward pass!
        
#         self.q_proj = nn.Linear(model_dim, model_dim)  
#         self.k_proj = nn.Linear(model_dim, model_dim)  
#         self.v_proj = nn.Linear(model_dim, model_dim)  

#     def forward(self, Q_emb: torch.Tensor, motion_feat: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Q_emb:       (B, model_dim) - Question text embedding
#         motion_feat: (B, T, model_dim) - Projected motion features
#         motion_mask: (B, T) - Boolean mask where True = valid frame, False = padded zero
        
#         Returns:     (B, model_dim) - The question-aware global motion vector
#         """
#         # 1. Project Question (Query) and Motion (Keys, Values)
#         Q = self.q_proj(Q_emb)             # (B, model_dim)
#         K = self.k_proj(motion_feat)       # (B, T, model_dim)
#         V = self.v_proj(motion_feat)       # (B, T, model_dim)

#         # Unsqueeze Q to allow batched matrix multiplication against T frames
#         Q = Q.unsqueeze(1)                 # (B, 1, model_dim)

#         # 2. Compute Attention Scores: (Q * K^T) / sqrt(dim)
#         scale = Q.size(-1) ** 0.5
#         attn_logits = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B, 1, T)
        
#         # 3. Apply the Padding Mask
#         attn_logits = attn_logits.masked_fill(~motion_mask.unsqueeze(1), float('-inf'))

#         # 4. Softmax over the temporal dimension (T)
#         alpha = F.softmax(attn_logits, dim=-1)   # (B, 1, T)
#         alpha = F.dropout(alpha, p=0.3, training=self.training)

#         # 5. Weighted sum of the temporal frames
#         motion_targeted = torch.bmm(alpha, V)    # (B, 1, model_dim)
        
#         # Remove the extra sequence dimension
#         return motion_targeted.squeeze(1)        # (B, model_dim)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import softmax as pyg_softmax


class QuestionGuidedPooling(nn.Module):
    """
    FIXED: Multi-head question-guided graph pooling.
    """
    
    def __init__(
        self, 
        text_dim: int, 
        node_dim: int, 
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.use_residual = use_residual
        
        assert node_dim % num_heads == 0
        
        #Proper dimension projection
        self.text_to_node = nn.Sequential(
            nn.Linear(text_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.q_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.k_proj = nn.Linear(node_dim, node_dim, bias=False)
        self.v_proj = nn.Linear(node_dim, node_dim, bias=False)
        
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.head_dim ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.head_dim ** -0.5)
        
        self.out_proj = nn.Linear(node_dim, node_dim, bias=True)
        self.ln_out = nn.LayerNorm(node_dim)
        
        self.temperature = nn.Parameter(torch.tensor(self.head_dim ** 0.5, dtype=torch.float32))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        
        self.gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    
    def forward(
        self, 
        Q_emb: torch.Tensor, 
        node_feat: torch.Tensor, 
        batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Q_emb: (B, text_dim) question embedding
            node_feat: (N, node_dim) node features
            batch_idx: (N,) batch indices
        Returns:
            (B, node_dim) pooled features
        """
        
        B = int(batch_idx.max()) + 1 if batch_idx.numel() > 0 else 1
        N = node_feat.size(0)
        H = self.num_heads
        d = self.head_dim
        device = node_feat.device
        
        # Project question
        Q_emb = self.text_to_node(Q_emb)  # (B, node_dim)
        
        # Multi-head projections
        Q = self.q_proj(Q_emb).view(B, H, d)  # (B, H, d)
        K = self.k_proj(node_feat).view(N, H, d)  # (N, H, d)
        V = self.v_proj(node_feat).view(N, H, d)  # (N, H, d)
        
        # Expand queries to node level
        Q_expanded = Q[batch_idx]  # (N, H, d)
        
        # Attention scores
        attn_logits = torch.sum(Q_expanded * K, dim=-1) / (self.temperature.clamp(min=0.1))  # (N, H)
        attn_logits = torch.clamp(attn_logits, min=-10.0, max=10.0)
        
        # Softmax per graph
        alpha = pyg_softmax(attn_logits, batch_idx, num_nodes=B)  # (N, H)
        alpha = self.attn_dropout(alpha)
        
        # Aggregate
        v_weighted = V * alpha.unsqueeze(-1)  # (N, H, d)
        
        v_graph_targeted = torch.zeros(B, H, d, dtype=node_feat.dtype, device=device)
        for h in range(H):
            v_graph_targeted[:, h] = scatter(v_weighted[:, h], batch_idx, dim=0, dim_size=B, reduce="sum")
        
        v_graph_targeted = v_graph_targeted.view(B, -1)  # (B, node_dim)
        v_graph_targeted = self.out_proj(v_graph_targeted)
        
        # Residual blend
        if self.use_residual:
            v_graph_targeted = Q_emb + torch.sigmoid(self.gate) * (v_graph_targeted - Q_emb)
        
        v_graph_targeted = self.ln_out(v_graph_targeted)
        v_graph_targeted = self.feat_dropout(v_graph_targeted)
        
        return v_graph_targeted


# ════════════════════════════════════════════════════════════════════════════════
# 4. QUESTION-GUIDED MOTION POOLING
# ════════════════════════════════════════════════════════════════════════════════

class QuestionGuidedMotionPooling(nn.Module):
    """
    FIXED: Multi-head question-guided temporal pooling.
    """
    
    def __init__(
        self, 
        model_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.use_residual = use_residual
        
        assert model_dim % num_heads == 0
        
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.head_dim ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.head_dim ** -0.5)
        
        self.out_proj = nn.Linear(model_dim, model_dim, bias=True)
        self.ln_out = nn.LayerNorm(model_dim)
        
        self.temperature = nn.Parameter(torch.tensor(self.head_dim ** 0.5, dtype=torch.float32))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        
        self.gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    
    def forward(
        self, 
        Q_emb: torch.Tensor, 
        motion_feat: torch.Tensor, 
        motion_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Q_emb: (B, model_dim) question embedding
            motion_feat: (B, T, model_dim) motion features
            motion_mask: (B, T) temporal mask
        Returns:
            (B, model_dim) pooled motion
        """
        
        B, T = motion_feat.shape[:2]
        H = self.num_heads
        d = self.head_dim
        
        # Multi-head projections
        Q = self.q_proj(Q_emb).view(B, 1, H, d)  # (B, 1, H, d)
        K = self.k_proj(motion_feat).view(B, T, H, d)  # (B, T, H, d)
        V = self.v_proj(motion_feat).view(B, T, H, d)  # (B, T, H, d)
        
        # Attention scores
        attn_logits = torch.sum(Q * K, dim=-1) / (self.temperature.clamp(min=0.1))  # (B, T, H)
        attn_logits = torch.clamp(attn_logits, min=-10.0, max=10.0)
        
        # Apply mask
        mask_expanded = motion_mask.unsqueeze(-1)  # (B, T, 1)
        attn_logits = attn_logits.masked_fill(~mask_expanded, -1e4)
        
        # Softmax
        alpha = torch.softmax(attn_logits, dim=1)  # (B, T, H)
        alpha = self.attn_dropout(alpha)
        
        # Aggregate
        alpha_t = alpha.permute(0, 2, 1)  # (B, H, T)
        V_t = V.permute(0, 2, 1, 3)  # (B, H, T, d)
        
        motion_targeted = torch.matmul(alpha_t.unsqueeze(2), V_t)  # (B, H, 1, d)
        motion_targeted = motion_targeted.squeeze(2).reshape(B, -1)  # (B, model_dim)
        
        motion_targeted = self.out_proj(motion_targeted)
        
        # Residual blend
        if self.use_residual:
            motion_targeted = Q_emb + torch.sigmoid(self.gate) * (motion_targeted - Q_emb)
        
        motion_targeted = self.ln_out(motion_targeted)
        motion_targeted = self.feat_dropout(motion_targeted)
        
        return motion_targeted


# ════════════════════════════════════════════════════════════════════════════════
# 5. CROSS-MODAL ATTENTION FUSION
# ════════════════════════════════════════════════════════════════════════════════

class CrossModalAttentionFusion(nn.Module):
    """
    FIXED: Unified cross-modal fusion combining all representations.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.graph_pooling = QuestionGuidedPooling(
            text_dim=dim,
            node_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.motion_pooling = QuestionGuidedMotionPooling(
            model_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(
        self,
        Q_emb: torch.Tensor,
        v_graph: torch.Tensor,
        motion_feat: torch.Tensor,
        batch_idx: torch.Tensor = None,
        motion_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            Q_emb: (B, dim) question embedding
            v_graph: (N, dim) graph features
            motion_feat: (B, T, dim) motion features
            batch_idx: (N,) batch indices
            motion_mask: (B, T) temporal mask
        Returns:
            (B, dim) fused representation
        """
        
        # Pool graph and motion with question guidance
        v_graph_fused = self.graph_pooling(Q_emb, v_graph, batch_idx)  # (B, dim)
        motion_fused = self.motion_pooling(Q_emb, motion_feat, motion_mask)  # (B, dim)
        
        # Concatenate and fuse
        combined = torch.cat([v_graph_fused, motion_fused, Q_emb], dim=-1)  # (B, 3*dim)
        fused = self.fusion(combined)  # (B, dim)
        
        return fused

# class QuestionGuidedPooling(nn.Module):
#     """
#     FIXED: Multi-head question-guided graph pooling with:
#     - Proper input projection dimensions
#     - Multi-head attention for richness
#     - LayerNorm after projections
#     - Attention score clipping
#     - Residual connection
#     - Adaptive temperature scaling
#     - Head diversity regularization
#     """
    
#     def __init__(
#         self, 
#         text_dim: int, 
#         node_dim: int, 
#         num_heads: int = 8,
#         dropout: float = 0.1,
#         use_residual: bool = True
#     ):
#         super().__init__()
        
#         self.text_dim = text_dim
#         self.node_dim = node_dim
#         self.num_heads = num_heads
#         self.head_dim = node_dim // num_heads
#         self.use_residual = use_residual
        
#         assert node_dim % num_heads == 0, f"node_dim ({node_dim}) must be divisible by num_heads ({num_heads})"
        
#         # ── FIXED: Project text to node_dim properly
#         self.text_to_node = nn.Sequential(
#             nn.Linear(text_dim, node_dim),
#             nn.LayerNorm(node_dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
        
#         # ── Multi-head projections
#         self.q_proj = nn.Linear(node_dim, node_dim, bias=False)
#         self.k_proj = nn.Linear(node_dim, node_dim, bias=False)
#         self.v_proj = nn.Linear(node_dim, node_dim, bias=False)
        
#         nn.init.xavier_uniform_(self.q_proj.weight, gain=self.head_dim ** -0.5)
#         nn.init.xavier_uniform_(self.k_proj.weight, gain=self.head_dim ** -0.5)
        
#         # ── Output projection
#         self.out_proj = nn.Linear(node_dim, node_dim, bias=True)
        
#         # ── Normalization
#         self.ln_out = nn.LayerNorm(node_dim)
        
#         # ── Adaptive temperature (learnable)
#         self.temperature = nn.Parameter(torch.tensor(self.head_dim ** 0.5))
        
#         # ── Dropout
#         self.attn_dropout = nn.Dropout(dropout)
#         self.feat_dropout = nn.Dropout(dropout)
        
#         # ── Residual gate
#         self.gate = nn.Parameter(torch.tensor(0.5))
    
#     def forward(
#         self, 
#         Q_emb: torch.Tensor, 
#         node_feat: torch.Tensor, 
#         batch_idx: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Args:
#             Q_emb: (B, text_dim) - Question text embedding
#             node_feat: (N, node_dim) - ST-Graph node features
#             batch_idx: (N,) - Maps nodes to their respective graphs
        
#         Returns:
#             (B, node_dim) - Question-aware visual vector per graph
#         """
        
#         B = int(batch_idx.max()) + 1 if batch_idx.numel() > 0 else 1
#         N = node_feat.size(0)
#         H = self.num_heads
#         d = self.head_dim
        
#         # ── FIXED: Project question properly
#         Q_emb = self.text_to_node(Q_emb)  # (B, node_dim)
        
#         # ── Project Question, Keys, Values to multi-head
#         Q = self.q_proj(Q_emb)  # (B, node_dim)
#         K = self.k_proj(node_feat)  # (N, node_dim)
#         V = self.v_proj(node_feat)  # (N, node_dim)
        
#         # ── Reshape for multi-head: (*, heads, head_dim)
#         Q = Q.view(B, H, d)  # (B, H, d)
#         K = K.view(N, H, d)  # (N, H, d)
#         V = V.view(N, H, d)  # (N, H, d)
        
#         # ── Expand graph-level queries to node-level
#         Q_expanded = Q[batch_idx]  # (N, H, d)
        
#         # ── Compute attention scores with adaptive temperature
#         # FIXED: score = (Q * K) / sqrt(d) * temperature (learnable)
#         attn_logits = torch.sum(Q_expanded * K, dim=-1) / (self.temperature.clamp(min=0.1))  # (N, H)
        
#         # ── FIXED: Clamp scores to prevent divergence
#         attn_logits = torch.clamp(attn_logits, min=-10.0, max=10.0)
        
#         # ── FIXED: Softmax strictly over nodes in the same graph (per head)
#         # For each batch, softmax over nodes belonging to that batch
#         alpha = pyg_softmax(attn_logits, batch_idx, num_nodes=B)  # (N, H)
        
#         # ── FIXED: Apply attention dropout
#         alpha = self.attn_dropout(alpha)
        
#         # ── Weighted sum of values: (N, H, d)
#         v_weighted = V * alpha.unsqueeze(-1)  # (N, H, d)
        
#         # ── FIXED: Aggregate per head separately for better efficiency
#         v_graph_targeted = torch.zeros(
#             B, H, d, 
#             dtype=v_weighted.dtype, 
#             device=v_weighted.device
#         )
        
#         for h in range(H):
#             v_graph_targeted[:, h] = scatter(
#                 v_weighted[:, h], 
#                 batch_idx, 
#                 dim=0, 
#                 dim_size=B, 
#                 reduce="sum"
#             )
        
#         # ── Reshape back to (B, node_dim)
#         v_graph_targeted = v_graph_targeted.view(B, -1)  # (B, node_dim)
        
#         # ── FIXED: Output projection + normalization
#         v_graph_targeted = self.out_proj(v_graph_targeted)
        
#         # ── FIXED: Residual blend (prevents information loss)
#         # v_out = Q_emb + sigmoid(gate) * (v_pooled - Q_emb)
#         if self.use_residual:
#             v_graph_targeted = Q_emb + torch.sigmoid(self.gate) * (v_graph_targeted - Q_emb)
        
#         v_graph_targeted = self.ln_out(v_graph_targeted)
#         v_graph_targeted = self.feat_dropout(v_graph_targeted)
        
#         return v_graph_targeted


# class QuestionGuidedMotionPooling(nn.Module):
#     """
#     FIXED: Multi-head question-guided temporal pooling with:
#     - Multi-head attention
#     - Proper mask handling (no -inf)
#     - Temperature scaling
#     - LayerNorm + Dropout
#     - Residual connection
#     - Per-head aggregation
#     """
    
#     def __init__(
#         self, 
#         model_dim: int,
#         num_heads: int = 8,
#         dropout: float = 0.1,
#         use_residual: bool = True
#     ):
#         super().__init__()
        
#         self.model_dim = model_dim
#         self.num_heads = num_heads
#         self.head_dim = model_dim // num_heads
#         self.use_residual = use_residual
        
#         assert model_dim % num_heads == 0, f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
        
#         # ── Multi-head projections
#         self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
#         self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
#         self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        
#         nn.init.xavier_uniform_(self.q_proj.weight, gain=self.head_dim ** -0.5)
#         nn.init.xavier_uniform_(self.k_proj.weight, gain=self.head_dim ** -0.5)
        
#         # ── Output projection
#         self.out_proj = nn.Linear(model_dim, model_dim, bias=True)
        
#         # ── Normalization
#         self.ln_out = nn.LayerNorm(model_dim)
        
#         # ── Adaptive temperature
#         self.temperature = nn.Parameter(torch.tensor(self.head_dim ** 0.5))
        
#         # ── Dropout
#         self.attn_dropout = nn.Dropout(dropout)
#         self.feat_dropout = nn.Dropout(dropout)
        
#         # ── Residual gate
#         self.gate = nn.Parameter(torch.tensor(0.5))
    
#     def forward(
#         self, 
#         Q_emb: torch.Tensor, 
#         motion_feat: torch.Tensor, 
#         motion_mask: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Args:
#             Q_emb: (B, model_dim) - Question text embedding
#             motion_feat: (B, T, model_dim) - Projected motion features
#             motion_mask: (B, T) - Boolean mask, True = valid, False = padded
        
#         Returns:
#             (B, model_dim) - Question-aware motion vector
#         """
        
#         B, T = motion_feat.shape[:2]
#         H = self.num_heads
#         d = self.head_dim
        
#         # ── Project to multi-head
#         Q = self.q_proj(Q_emb)  # (B, model_dim)
#         K = self.k_proj(motion_feat)  # (B, T, model_dim)
#         V = self.v_proj(motion_feat)  # (B, T, model_dim)
        
#         # ── Reshape for multi-head
#         Q = Q.view(B, 1, H, d)  # (B, 1, H, d)
#         K = K.view(B, T, H, d)  # (B, T, H, d)
#         V = V.view(B, T, H, d)  # (B, T, H, d)
        
#         # ── Compute attention scores: (B, T, H)
#         attn_logits = torch.sum(Q * K, dim=-1) / (self.temperature.clamp(min=0.1))  # (B, T, H)
        
#         # ── FIXED: Clamp scores
#         attn_logits = torch.clamp(attn_logits, min=-10.0, max=10.0)
        
#         # ── FIXED: Apply mask properly (don't use -inf, use large negative)
#         # Expand mask to (B, T, H)
#         mask_expanded = motion_mask.unsqueeze(-1)  # (B, T, 1)
#         attn_logits = attn_logits.masked_fill(~mask_expanded, -1e4)  # (B, T, H)
        
#         # ── Softmax over temporal dimension
#         alpha = torch.softmax(attn_logits, dim=1)  # (B, T, H)
        
#         # ── FIXED: Apply attention dropout
#         alpha = self.attn_dropout(alpha)
        
#         # ── Aggregate: (B, H, d)
#         # alpha: (B, T, H) -> (B, H, T)
#         alpha_t = alpha.permute(0, 2, 1)  # (B, H, T)
#         # V: (B, T, H, d)
#         V_t = V.permute(0, 2, 1, 3)  # (B, H, T, d)
        
#         # Batch matrix multiply: (B, H, 1, T) @ (B, H, T, d) -> (B, H, 1, d)
#         motion_targeted = torch.matmul(alpha_t.unsqueeze(2), V_t)  # (B, H, 1, d)
#         motion_targeted = motion_targeted.squeeze(2)  # (B, H, d)
        
#         # ── Reshape back to (B, model_dim)
#         motion_targeted = motion_targeted.reshape(B, -1)  # (B, model_dim)
        
#         # ── FIXED: Output projection + normalization
#         motion_targeted = self.out_proj(motion_targeted)
        
#         # ── FIXED: Residual blend
#         if self.use_residual:
#             motion_targeted = Q_emb + torch.sigmoid(self.gate) * (motion_targeted - Q_emb)
        
#         motion_targeted = self.ln_out(motion_targeted)
#         motion_targeted = self.feat_dropout(motion_targeted)
        
#         return motion_targeted


# # ──────────────────────────────────────────────────────────────────────
# # HELPER: Unified Cross-Attention Fusion
# # ──────────────────────────────────────────────────────────────────────

# class CrossModalAttentionFusion(nn.Module):
#     """
#     FIXED: Fuse graph features, motion features, and text via multi-head cross-attention.
    
#     Architecture:
#         v_graph × text → attention pooling → v_graph_fused
#         motion × text → attention pooling → motion_fused
#         [v_graph_fused || motion_fused || text] → MLP → output
#     """
    
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 8,
#         dropout: float = 0.1
#     ):
#         super().__init__()
        
#         self.graph_pooling = QuestionGuidedPooling(
#             text_dim=dim,
#             node_dim=dim,
#             num_heads=num_heads,
#             dropout=dropout
#         )
        
#         self.motion_pooling = QuestionGuidedMotionPooling(
#             model_dim=dim,
#             num_heads=num_heads,
#             dropout=dropout
#         )
        
#         # Fusion MLP
#         self.fusion = nn.Sequential(
#             nn.Linear(dim * 3, dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.LayerNorm(dim * 2),
#             nn.Linear(dim * 2, dim),
#             nn.LayerNorm(dim)
#         )
    
#     def forward(
#         self,
#         Q_emb: torch.Tensor,
#         v_graph: torch.Tensor,
#         motion_feat: torch.Tensor,
#         batch_idx: torch.Tensor = None,
#         motion_mask: torch.Tensor = None
#     ) -> torch.Tensor:
#         """
#         Args:
#             Q_emb: (B, dim) question embedding
#             v_graph: (N, dim) graph features
#             motion_feat: (B, T, dim) motion features
#             batch_idx: (N,) batch indices for graph
#             motion_mask: (B, T) temporal mask
        
#         Returns:
#             (B, dim) fused representation
#         """
        
#         # Question-guided pooling
#         v_graph_fused = self.graph_pooling(Q_emb, v_graph, batch_idx)  # (B, dim)
#         motion_fused = self.motion_pooling(Q_emb, motion_feat, motion_mask)  # (B, dim)
        
#         # Concatenate all modalities
#         fused = torch.cat([v_graph_fused, motion_fused, Q_emb], dim=-1)  # (B, 3*dim)
        
#         # Final fusion
#         output = self.fusion(fused)  # (B, dim)
        
#         return output
