
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import softmax


# # ==================================================================
# # 1. Node Attribute Encoder
# # ==================================================================
# class NodeAttributeEncoder(nn.Module):

#     def __init__(self, raw_dim, dim, num_classes):
#         super().__init__()

#         # project backbone raw features (1280) → dim
#         self.raw_proj = nn.Linear(raw_dim, dim)

#         # attribute encoders
#         self.bbox_mlp = nn.Sequential(
#             nn.Linear(4, dim),
#             nn.GELU(),
#             nn.Linear(dim, dim),
#         )
#         self.class_embed = nn.Embedding(num_classes, dim)
#         self.conf_mlp = nn.Sequential(
#             nn.Linear(1, dim),
#             nn.GELU(),
#             nn.Linear(dim, dim),
#         )

#         # trajectory fusion: [h || e_traj] → dim
#         # self.traj_proj = nn.Linear(2 * dim, dim)

#         # self.norm = nn.LayerNorm(dim)
#         self.traj_gate = nn.Linear(dim, dim)
#         self.attr_gate = nn.Linear(dim, dim)

#     # ------------------------------------------------------------------
#     @staticmethod
#     def encode_bbox(bbox: torch.Tensor) -> torch.Tensor:
#         """
#         bbox: (N, 6) — [x1, y1, x2, y2, H, W]  pixel coords + frame size
#         returns: (N, 4) — [xc, yc, w, h] normalised to [0, 1]
#         """
#         x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
#         W, H            = bbox[:, 4], bbox[:, 5]   # note: col 4=W, col 5=H
#         xc = (x1 + x2) / (2.0 * W.clamp(min=1))
#         yc = (y1 + y2) / (2.0 * H.clamp(min=1))
#         w  = (x2 - x1) / W.clamp(min=1)
#         h  = (y2 - y1) / H.clamp(min=1)
#         return torch.stack([xc, yc, w, h], dim=-1)   # (N, 4)

#     # ------------------------------------------------------------------
#     def _build_trajectory_tokens(
#         self,
#         h:                 torch.Tensor,   # (N, D)
#         node_obj_ids_flat: torch.Tensor,   # (N,)
#         node_is_keyframe:  torch.Tensor,   # (N,) bool
#         node_kf_list_idx:  torch.Tensor,   # (N,)
#     ) -> torch.Tensor:
#         """
#         For each node, compute a trajectory token = temporally-ordered mean
#         of that object's keyframe features.

#         Uses scatter operations instead of Python loops for GPU efficiency.

#         Returns: (N, D) — trajectory token broadcast back to every node
#         """
#         D = h.shape[1]
#         device = h.device

#         kf_mask    = node_is_keyframe.bool()
#         kf_feats   = h[kf_mask]                       # (N_kf, D)
#         kf_obj_ids = node_obj_ids_flat[kf_mask]        # (N_kf,)

#         # ── map obj_ids → contiguous indices ──────────────────────
#         unique_ids, inverse = torch.unique(kf_obj_ids, return_inverse=True)
#         n_objs = unique_ids.shape[0]

#         # ── scatter mean: sum features per object ─────────────────
#         traj_sum   = torch.zeros(n_objs, D, device=device, dtype=h.dtype)
#         traj_count = torch.zeros(n_objs,    device=device, dtype=h.dtype)

#         traj_sum  .scatter_add_(0, inverse.unsqueeze(1).expand_as(kf_feats), kf_feats)
#         traj_count.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=h.dtype))

#         traj_tokens = traj_sum / traj_count.unsqueeze(1).clamp(min=1)  # (n_objs, D)

#         # ── broadcast back to all N nodes by obj_id ───────────────
#         # build obj_id → traj_token lookup via index
#         id_to_idx = torch.zeros(
#             unique_ids.max().item() + 1, dtype=torch.long, device=device
#         )
#         id_to_idx[unique_ids] = torch.arange(n_objs, device=device)

#         node_traj_idx = id_to_idx[node_obj_ids_flat]      # (N,)
#         e_traj = traj_tokens[node_traj_idx]                # (N, D)

#         return e_traj

#     # ------------------------------------------------------------------
#     def forward(
#         self,
#         node_raw:          torch.Tensor,   # (N, 1280)
#         bbox:              torch.Tensor,   # (N, 6)
#         cls_id:            torch.Tensor,   # (N,)
#         conf:              torch.Tensor,   # (N,)
#         node_obj_ids_flat: torch.Tensor,   # (N,)
#         node_is_keyframe:  torch.Tensor,   # (N,) bool
#         node_kf_list_idx:  torch.Tensor,   # (N,)
#     ) -> torch.Tensor:                     # (N, dim)

#         f_raw = self.raw_proj(node_raw)                       # (N, dim)

#         # 2. attribute embeddings
#         bbox_norm = self.encode_bbox(bbox)                     # (N, 4)
#         e_bbox    = self.bbox_mlp(bbox_norm)                   # (N, dim)
#         e_cls     = self.class_embed(cls_id)                   # (N, dim)
#         e_conf    = self.conf_mlp(conf.unsqueeze(-1))          # (N, dim)
        
#         # attr gating
#         attrs = e_bbox + e_cls + e_conf                     # Combine non-visual attributes first
#         gate_attrs = torch.sigmoid(self.attr_gate(f_raw))    # Visual feature decides the gate!
#         h = f_raw + gate_attrs * attrs       

#         # 4. trajectory enrichment (scatter-based, no Python loop)
#         f_traj = self._build_trajectory_tokens(f_raw, node_obj_ids_flat, node_is_keyframe, node_kf_list_idx)
        
#         # 5. gated addition
#         gate   = torch.sigmoid(self.traj_gate(h))
#         h      = h + gate * f_traj
#         return h                                  # (N, dim)




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean


# ==================================================================
# FIXED: Node Attribute Encoder with Proper Residuals & Normalization
# ==================================================================

class ResidualGate(nn.Module):
    """Residual gated fusion: y = x + sigmoid(w) * (z - x)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, z: torch.Tensor, gate_param: nn.Parameter) -> torch.Tensor:
        """
        Args:
            x: (N, D) residual connection
            z: (N, D) enhancement signal
            gate_param: (D,) or scalar gate weight
        Returns:
            (N, D) blended output with proper skip connection
        """
        gate = torch.sigmoid(gate_param)
        return x + gate * (z - x)  # Ensures bounded update to x


class NodeAttributeEncoder(nn.Module):
    """
    FIXED: Multi-stage attribute encoder with residual connections,
    proper normalization, and efficient trajectory aggregation.
    
    Architecture:
        raw_feat → [Stage 1: Attribute Fusion] → [Stage 2: Trajectory Enhancement] → output
                         ↓ skip                              ↓ skip
    """

    def __init__(self, raw_dim: int, dim: int, num_classes: int, use_normalization: bool = True):
        super().__init__()
        self.dim = dim
        self.use_normalization = use_normalization

        # ── Stage 0: Raw Feature Projection
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim, dim),
            nn.LayerNorm(dim) if use_normalization else nn.Identity(),
            nn.GELU()
        )

        # ── Stage 1: Attribute Encoders (Parallel, Not Sequential)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim) if use_normalization else nn.Identity()
        )
        
        self.class_embed = nn.Embedding(num_classes, dim)
        
        self.conf_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim) if use_normalization else nn.Identity()
        )

        # ── Stage 1: Multi-head Attribute Fusion (learned weighted combination)
        self.attr_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.LayerNorm(dim) if use_normalization else nn.Identity()
        )
        
        # ── Stage 1: Residual Gate for attribute blend
        self.attr_gate_w = nn.Parameter(torch.ones(dim) * 0.5)  # Initialized to 0.5 gating
        
        # ── Stage 2: Trajectory Enhancement
        self.traj_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim) if use_normalization else nn.Identity(),
            nn.GELU()
        )
        
        # ── Stage 2: Residual Gate for trajectory blend  
        self.traj_gate_w = nn.Parameter(torch.ones(dim) * 0.5)
        
        # ── Final output normalization
        self.output_norm = nn.LayerNorm(dim) if use_normalization else nn.Identity()

    # ------------------------------------------------------------------
    @staticmethod
    def encode_bbox(bbox: torch.Tensor) -> torch.Tensor:
        """
        FIXED: More numerically stable bbox encoding
        
        bbox: (N, 6) — [x1, y1, x2, y2, H, W]  pixel coords + frame size
        returns: (N, 4) — [xc, yc, w, h] normalised to [0, 1]
        """
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        W, H = bbox[:, 4], bbox[:, 5]
        
        # Clamp to avoid division by zero
        W = W.clamp(min=1e-6)
        H = H.clamp(min=1e-6)
        
        xc = (x1 + x2) / (2.0 * W)
        yc = (y1 + y2) / (2.0 * H)
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        
        # Clip to [0, 1] to remove outliers
        bbox_norm = torch.stack([xc, yc, w, h], dim=-1)
        bbox_norm = torch.clamp(bbox_norm, min=0.0, max=1.0)
        
        return bbox_norm

    # ------------------------------------------------------------------
    def _build_trajectory_tokens(
        self,
        h: torch.Tensor,              # (N, D)
        node_obj_ids_flat: torch.Tensor,  # (N,)
        node_is_keyframe: torch.Tensor,   # (N,) bool
        node_kf_list_idx: torch.Tensor,   # (N,)
    ) -> torch.Tensor:
        """
        OPTIMIZED: Efficient trajectory aggregation using torch_scatter.
        Computes trajectory token = mean of all keyframe features per object.
        
        Returns: (N, D) — trajectory token broadcast back to every node
        """
        D = h.shape[1]
        device = h.device

        # ── Filter to keyframes only
        kf_mask = node_is_keyframe.bool()
        if kf_mask.sum() == 0:
            # If no keyframes, return zero trajectory
            return torch.zeros_like(h)
        
        kf_feats = h[kf_mask]  # (N_kf, D)
        kf_obj_ids = node_obj_ids_flat[kf_mask]  # (N_kf,)

        # ── Map object IDs to contiguous indices for scatter
        unique_ids, inverse_indices = torch.unique(kf_obj_ids, return_inverse=True)
        n_objs = unique_ids.shape[0]

        # ── Use torch_scatter for efficient aggregation
        # FIXED: Use scatter_mean directly instead of manual sum/count
        try:
            traj_tokens = scatter_mean(kf_feats, inverse_indices, dim=0, dim_size=n_objs)  # (n_objs, D)
        except:
            # Fallback if torch_scatter unavailable
            traj_sum = torch.zeros(n_objs, D, device=device, dtype=h.dtype)
            traj_count = torch.zeros(n_objs, device=device, dtype=h.dtype)
            
            traj_sum.scatter_add_(0, inverse_indices.unsqueeze(1).expand_as(kf_feats), kf_feats)
            traj_count.scatter_add_(0, inverse_indices, torch.ones(len(inverse_indices), device=device))
            
            traj_tokens = traj_sum / traj_count.clamp(min=1).unsqueeze(1)

        # ── Broadcast trajectory tokens back to all N nodes
        id_to_idx = torch.full(
            (unique_ids.max().item() + 1,), 
            -1, 
            dtype=torch.long, 
            device=device
        )
        id_to_idx[unique_ids] = torch.arange(n_objs, device=device)
        
        # ── Handle nodes that don't belong to any object (edge case)
        node_traj_idx = id_to_idx[node_obj_ids_flat.clamp(min=0)]
        valid_mask = node_traj_idx >= 0
        
        e_traj = torch.zeros_like(h)
        e_traj[valid_mask] = traj_tokens[node_traj_idx[valid_mask]]
        
        return e_traj

    # ------------------------------------------------------------------
    def forward(
        self,
        node_raw: torch.Tensor,          # (N, 1280)
        bbox: torch.Tensor,              # (N, 6)
        cls_id: torch.Tensor,            # (N,)
        conf: torch.Tensor,              # (N,)
        node_obj_ids_flat: torch.Tensor, # (N,)
        node_is_keyframe: torch.Tensor,  # (N,) bool
        node_kf_list_idx: torch.Tensor,  # (N,)
    ) -> torch.Tensor:                   # (N, dim)
        """
        FIXED: Multi-stage encoding with residual connections and proper normalization.
        
        Flow:
            1. Project raw features
            2. Encode 3 attributes in parallel
            3. Fuse attributes with residual blend
            4. Aggregate trajectory information
            5. Fuse trajectory with residual blend
            6. Final normalization
        """
        
        # ── STAGE 0: Raw Feature Projection
        f_raw = self.raw_proj(node_raw)  # (N, dim)
        
        # ── STAGE 1: Parallel Attribute Encoding
        bbox_norm = self.encode_bbox(bbox)  # (N, 4)
        e_bbox = self.bbox_mlp(bbox_norm)   # (N, dim)
        e_cls = self.class_embed(cls_id)    # (N, dim)
        e_conf = self.conf_mlp(conf.unsqueeze(-1))  # (N, dim)
        
        # ── STAGE 1: Fuse attributes using learned combination (NOT simple sum)
        attrs_concat = torch.cat([e_bbox, e_cls, e_conf], dim=-1)  # (N, dim*3)
        attrs_fused = self.attr_fusion(attrs_concat)  # (N, dim)
        
        # ── STAGE 1: Residual blend - mix raw features with attributes
        # Formula: h = f_raw + sigmoid(w) * (attrs_fused - f_raw)
        # This ensures: 
        #   - w=0 → h=f_raw (pure residual)
        #   - w=1 → h=attrs_fused (pure blend)
        h = f_raw + torch.sigmoid(self.attr_gate_w) * (attrs_fused - f_raw)
        
        # ── STAGE 2: Trajectory Enrichment
        f_traj = self._build_trajectory_tokens(
            f_raw,  # Use raw features for trajectory (more direct signal)
            node_obj_ids_flat, 
            node_is_keyframe, 
            node_kf_list_idx
        )  # (N, dim)
        
        # ── STAGE 2: Project trajectory features
        f_traj = self.traj_proj(f_traj)  # (N, dim)
        
        # ── STAGE 2: Residual blend for trajectory
        # Formula: h = h + sigmoid(w_traj) * (f_traj - h)
        h = h + torch.sigmoid(self.traj_gate_w) * (f_traj - h)
        
        # ── FINAL: Output Normalization
        h = self.output_norm(h)
        
        return h  # (N, dim)


