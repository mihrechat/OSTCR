import torch
import torch.nn as nn

class CausalFusionModule(nn.Module):
    """
    Implements g_φ(E_z, E_v, M) -> answer logits.
    Follows strictly Linear fusion AFTER expectations are resolved to satisfy NWGM.
    """
    def __init__(self, dim: int, num_answers: int, dropout: float = 0.1):
        super().__init__()

        # Stage 1: Nonlinear per-source enrichment BEFORE expectations merge
        self.phi_z = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.phi_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.phi_m = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

        # Stage 2: Strictly linear fusion mapping to vocabulary (Satisfies Eq. 8)
        self.g_phi = nn.Linear(dim * 3, num_answers, bias=True)

    def forward(self, E_z: torch.Tensor, E_v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        # All inputs are strictly Graph-Level: (B, dim)
        z_enc = self.phi_z(E_z)
        v_enc = self.phi_v(E_v) 
        m_enc = self.phi_m(M)   

        # Final NWGM prediction
        combined = torch.cat([z_enc, v_enc, m_enc], dim=-1)  # (B, 3*dim)
        return self.g_phi(combined)                          # (B, num_answers)
