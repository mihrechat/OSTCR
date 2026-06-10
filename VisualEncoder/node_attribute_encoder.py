
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean



class ResidualGate(nn.Module):
    """Residual gated fusion: y = x + sigmoid(w) * (z - x)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, z: torch.Tensor, gate_param: nn.Parameter) -> torch.Tensor:
        gate = torch.sigmoid(gate_param)
        return x + gate * (z - x)  # Ensures bounded update to x


class NodeAttributeEncoder(nn.Module):


    def __init__(self, raw_dim: int, dim: int, num_classes: int, use_normalization: bool = True):
        super().__init__()
        self.dim = dim
        self.use_normalization = use_normalization

        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim, dim),
            nn.LayerNorm(dim) if use_normalization else nn.Identity(),
            nn.GELU()
        )

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

   
        self.attr_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.LayerNorm(dim) if use_normalization else nn.Identity()
        )
        
   
        self.attr_gate_w = nn.Parameter(torch.ones(dim) * 0.5)  # Initialized to 0.5 gating
        
        self.traj_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim) if use_normalization else nn.Identity(),
            nn.GELU()
        )
          
        self.traj_gate_w = nn.Parameter(torch.ones(dim) * 0.5)
        

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

    

    def forward(
        self,
        node_raw: torch.Tensor,          # (N, 1280)
        bbox: torch.Tensor,              # (N, 6)
        cls_id: torch.Tensor,            # (N,)
        conf: torch.Tensor,              # (N,)
        # node_obj_ids_flat: torch.Tensor, # (N,)
        # node_is_keyframe: torch.Tensor,  # (N,) bool
        # node_kf_list_idx: torch.Tensor,  # (N,)
    ) -> torch.Tensor:                   # (N, dim)

        f_raw = self.raw_proj(node_raw)  # (N, dim)
    
        bbox_norm = self.encode_bbox(bbox)  # (N, 4)
        e_bbox = self.bbox_mlp(bbox_norm)   # (N, dim)
        e_cls = self.class_embed(cls_id)    # (N, dim)
        e_conf = self.conf_mlp(conf.unsqueeze(-1))  # (N, dim)
        
        attrs_concat = torch.cat([e_bbox, e_cls, e_conf], dim=-1)  # (N, dim*3)
        attrs_fused = self.attr_fusion(attrs_concat)  # (N, dim)
        
       
        h = f_raw + torch.sigmoid(self.attr_gate_w) * (attrs_fused - f_raw)
        
        return h  # (N, dim)


