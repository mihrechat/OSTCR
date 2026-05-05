import torch
import torch.nn as nn
import torch.nn.functional as F


# class MoELinearFusion(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, num_qtypes: int):
#         super().__init__()
#         # in_dim is the sum of all your uneven vector dimensions!
#         self.moe_weights = nn.Parameter(torch.randn(num_qtypes, in_dim, out_dim) / (in_dim ** 0.5))
#         self.moe_biases  = nn.Parameter(torch.zeros(num_qtypes, out_dim))

#     def forward(self, E_z, E_v, M, E_cl, qtype_idx: torch.Tensor) -> torch.Tensor:
#         # E_z(1024) + E_v(512) + M(512) + E_cl(1024) = 3072 total size!
#         combined = torch.cat([E_z, E_v, M, E_cl], dim=-1)
#         qtype_idx = qtype_idx.view(-1)
        
#         selected_weights = self.moe_weights[qtype_idx] 
#         selected_biases  = self.moe_biases[qtype_idx]

#         fused_vector = torch.bmm(combined.unsqueeze(1), selected_weights).squeeze(1)
#         return fused_vector + selected_biases

# class MoELinearFusion(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, num_qtypes: int):
#         super().__init__()
#         # in_dim is now exactly 2048 (E_z + E_v + M)
#         self.moe_weights = nn.Parameter(torch.randn(num_qtypes, in_dim, out_dim) / (in_dim ** 0.5))
#         self.moe_biases  = nn.Parameter(torch.zeros(num_qtypes, out_dim))
#         self.layernorm   = nn.LayerNorm(out_dim)

#     def forward(self, E_z, E_v, M, qtype_idx: torch.Tensor) -> torch.Tensor:
#         # E_z(1024) + E_v(512) + M(512) = 2048 total size!
#         # Notice E_cl is completely removed, making this a PURE Visual Causal Fusion
#         combined = torch.cat([E_z, E_v, M], dim=-1)
        
#         qtype_idx = qtype_idx.view(-1)
        
#         selected_weights = self.moe_weights[qtype_idx] 
#         selected_biases  = self.moe_biases[qtype_idx]

#         fused_vector = torch.bmm(combined.unsqueeze(1), selected_weights).squeeze(1)
#         fused_vector = fused_vector + selected_biases
#         return self.layernorm(fused_vector)

class MoELinearFusion(nn.Module):
    """
    FIXED: MoE fusion with proper expert weighting.
    """
    def __init__(self, in_dim: int, out_dim: int, num_qtypes: int):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_qtypes = num_qtypes
        
        self.moe_weights = nn.Parameter(
            torch.randn(num_qtypes, in_dim, out_dim) / (in_dim ** 0.5)
        )
        self.moe_biases = nn.Parameter(torch.zeros(num_qtypes, out_dim))
        
        self.layernorm = nn.LayerNorm(out_dim)
        
        self.gate_network = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, num_qtypes),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self, 
        E_z: torch.Tensor, 
        E_v: torch.Tensor, 
        M: torch.Tensor,
        qtype_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            E_z: (B, dim) semantic prior
            E_v: (B, dim) visual prior
            M: (B, dim) mediator
            qtype_idx: (B,) question type indices
        Returns:
            (B, dim) fused causal signal
        """
        
        combined = torch.cat([E_z, E_v, M], dim=-1)  # (B, 3*dim)
        
        qtype_idx = qtype_idx.view(-1)
        selected_weights = self.moe_weights[qtype_idx]  # (B, in_dim, out_dim)
        selected_biases = self.moe_biases[qtype_idx]  # (B, out_dim)
        
        # Compute expert gates
        gates = self.gate_network(combined)  # (B, num_qtypes)
        gates_per_expert = gates[torch.arange(len(qtype_idx)), qtype_idx]  # (B,)
        
        # Apply expert transformation
        fused = torch.bmm(combined.unsqueeze(1), selected_weights).squeeze(1)  # (B, out_dim)
        fused = fused + selected_biases
        
        # Weight by gate
        fused = fused * gates_per_expert.unsqueeze(-1)
        
        fused = self.layernorm(fused)
        
        return fused
    
# class QuestionGatedCausalFusion(nn.Module):
#     def __init__(self, in_dim: int, text_dim: int, out_dim: int):
#         super().__init__()
#         # in_dim = E_z(1024) + E_v(512) + M(512) + E_m(512) + E_cl(1024) = 3584
#         self.causal_proj = nn.Linear(in_dim, out_dim)
        
#         # 2. The question generates 5 gates (weights) for the 5 variables
#         self.gate_generator = nn.Sequential(
#             nn.Linear(text_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 5), 
#             nn.Sigmoid()       
#         )
        
#     def forward(self, E_z, E_v, M, E_m, E_cl, Q_emb: torch.Tensor) -> torch.Tensor:
#         # 1. Question generates the scaling weights
#         gates = self.gate_generator(Q_emb) # Shape: (B, 5)
        
#         # 2. Scale each causal variable by its specific gate
#         # We slice gates[:, 0:1] to keep the shape (B, 1) for broadcasting
#         E_z_gated  = E_z  * gates[:, 0:1]
#         E_v_gated  = E_v  * gates[:, 1:2]
#         M_gated    = M    * gates[:, 2:3]
#         E_m_gated  = E_m  * gates[:, 3:4]
#         E_cl_gated = E_cl * gates[:, 4:5]
        
#         # 3. Concatenate the scaled variables
#         combined = torch.cat([E_z_gated, E_v_gated, M_gated, E_m_gated, E_cl_gated], dim=-1)
        
#         # 4. Project back to model dimension (512)
#         fused_vector = self.causal_proj(combined)
#         return fused_vector


# class QuestionGatedCausalFusion(nn.Module):
#     def __init__(self, in_dim: int, text_dim: int, out_dim: int):
#         super().__init__()
#         self.causal_proj = nn.Linear(in_dim, out_dim)
        
#         self.gate_generator = nn.Sequential(
#             nn.Linear(text_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 3),
#             nn.Sigmoid()       
#         )
        
#     def forward(self, E_z, E_v, M, Q_emb: torch.Tensor) -> torch.Tensor:
#         # 1. Question generates the scaling weights
#         gates = self.gate_generator(Q_emb) # Shape: (B, 4)
#         # 2. Scale each causal variable by its specific gate
#         E_z_gated = E_z * gates[:, 0:1]
#         E_v_gated = E_v * gates[:, 1:2]
#         M_gated   = M   * gates[:, 2:3]
        
#         combined = torch.cat([E_z_gated, E_v_gated, M_gated], dim=-1)
        
#         # 4. Project back to model dimension (e.g., 512)
#         fused_vector = self.causal_proj(combined)
#         return fused_vector

###new causal############
class QuestionGatedCausalFusion(nn.Module):
    def __init__(self, text_dim: int, out_dim: int):
        super().__init__()
        # E_z_gated + E_v_gated + M_gated + Interaction_term = 4 blocks
        self.causal_proj = nn.Linear(out_dim * 4, out_dim)
        
        # The question generates 3 competitive gates for the 3 causal variables
        self.gate_generator = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
            
        )
        
    def forward(self, E_z, E_v, M, Q_emb: torch.Tensor) -> torch.Tensor:
        # 1. Question generates the raw gating logits
        gate_logits = self.gate_generator(Q_emb) # Shape: (B, 3)
    
        gates = F.softmax(gate_logits / 0.7, dim=-1)
        gates = 0.8 * gates + (0.2 / 3.0)
        
        # 2. Scale each causal variable by its specific MoE gate
        E_z_gated = E_z * gates[:, 0:1]
        E_v_gated = E_v * gates[:, 1:2]
        M_gated   = M   * gates[:, 2:3]
        
     
        interaction_term = (E_z_gated * M_gated) + (E_v_gated * M_gated)
        
        # 3. Concatenate the scaled variables PLUS the interaction term
        combined = torch.cat([E_z_gated, E_v_gated, M_gated, interaction_term], dim=-1)
        
        # 4. Project back to model dimension
        fused_vector = self.causal_proj(combined)
        return fused_vector


# class QuestionGatedCausalFusion(nn.Module):
#     def __init__(self, text_dim: int, node_dim: int):
#         super().__init__()
        
#         # Interaction term: (E_z * M) + (E_v * M) -> 3 items concatenated
#         self.causal_proj = nn.Linear(node_dim * 4, node_dim)
        
#         # The question generates parameters for FiLM (Scale + Shift) for 3 variables
#         # 3 variables * 2 (gamma, beta) = 6
#         self.gate_generator = nn.Sequential(
#             nn.Linear(text_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, node_dim * 6) # Output size matches node_dim!
#         )
        
#     def forward(self, E_z: torch.Tensor, E_v: torch.Tensor, M: torch.Tensor, Q_emb: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
#         """
#         E_z, E_v, M: (N, Dim) - Node level causal signals
#         Q_emb: (B, Dim) - Graph level question embedding
#         batch_idx: (N,) - Mapping nodes to their parent graphs
#         """
#         N = E_z.size(0)
        
#         # 1. Question generates 6 FiLM parameters (gamma and beta for E_z, E_v, M)
#         film_params = self.gate_generator(Q_emb)  # (B, node_dim * 6)
        
#         # Split into 6 chunks of size (B, node_dim)
#         gammas_betas = film_params.chunk(6, dim=-1)
#         gz, bz, gv, bv, gm, bm = gammas_betas
        
#         # 2. Expand question's FiLM params from (B, Dim) to (N, Dim) to match nodes
#         gz, bz = gz[batch_idx], bz[batch_idx]
#         gv, bv = gv[batch_idx], bv[batch_idx]
#         gm, bm = gm[batch_idx], bm[batch_idx]
        
#         # 3. Apply FiLM modulation to each node-level causal signal
#         E_z_gated = (E_z * gz) + bz
#         E_v_gated = (E_v * gv) + bv
#         M_gated   = (M   * gm) + bm
        
#         # 4. Compute interaction terms (Node-wise element-wise multiplication)
#         interaction_term = (E_z_gated * M_gated) + (E_v_gated * M_gated)
        
#         # 5. Concatenate and project
#         combined = torch.cat([E_z_gated, E_v_gated, M_gated, interaction_term], dim=-1)
#         fused_vector = self.causal_proj(combined) # (N, node_dim)
        
#         return fused_vector

#New

# class QuestionGatedCausalFusion(nn.Module):
#     def __init__(self, in_dim: int, text_dim: int, out_dim: int):
#         super().__init__()
#         self.causal_proj = nn.Linear(in_dim, out_dim)
        
#         self.gate_generator = nn.Sequential(
#             nn.Linear(text_dim, 128),
#             nn.GELU(),
#             nn.Linear(128, 3),
#             nn.Sigmoid()
#         )
#         self.out_dim = out_dim
        
#         # Output normalization
#         self.out_norm = nn.LayerNorm(out_dim)
#         self.output_scale = nn.Parameter(torch.ones(1) * 1.0)
        
#     def forward(self, E_z, E_v, M, Q_emb: torch.Tensor) -> torch.Tensor:
#         # 1. Question generates the scaling weights
#         gates = self.gate_generator(Q_emb)  # (B, 3)
        
#         # 2. Scale each causal variable
#         E_z_gated = E_z * gates[:, 0:1]
#         E_v_gated = E_v * gates[:, 1:2]
#         M_gated   = M   * gates[:, 2:3]
        
#         # 3. Concatenate
#         combined = torch.cat([E_z_gated, E_v_gated, M_gated], dim=-1)
        
#         # 4. Project and normalize
#         fused_vector = self.causal_proj(combined)
#         fused_vector = self.out_norm(fused_vector)
#         fused_vector = fused_vector * self.output_scale
    
#         target_norm = (self.out_dim ** 0.5)  
#         fused_vector = fused_vector * target_norm / (fused_vector.norm(dim=-1, keepdim=True) + 1e-8)
        
#         return fused_vector
    
    import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalCrossAttentionFusion(nn.Module):
    """
    Uses the Question to dynamically attend over the pure causal variables.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        
        # Pre-Norms (Crucial for attention stability)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        # Post-Norm and Scale
        self.out_norm = nn.LayerNorm(dim)
        self.output_scale = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, E_z: torch.Tensor, E_v: torch.Tensor, M: torch.Tensor, Q_emb: torch.Tensor) -> torch.Tensor:
        """
        All inputs are assumed to be shape (B, dim)
        """
        # 1. Prepare Query from the Question
        # Shape: (B, 1, dim)
        Q = self.q_proj(self.norm_q(Q_emb)).unsqueeze(1) 
        
        # 2. Stack Causal Variables to create a 'sequence'
        # Sequence order: [Confounder, Backdoor Prior, Deconfounded Mediator]
        # Shape: (B, 3, dim)
        causal_seq = torch.stack([E_z, E_v, M], dim=1)
        causal_seq = self.norm_kv(causal_seq)
        
        # 3. Project to Keys and Values
        # Shape: (B, 3, dim*2) -> split to two (B, 3, dim)
        KV = self.kv_proj(causal_seq)
        K, V = KV.chunk(2, dim=-1)
        
        # 4. Scaled Dot-Product Attention
        scale = self.dim ** 0.5
        # (B, 1, dim) @ (B, dim, 3) -> (B, 1, 3)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / scale
        attn_weights = F.softmax(attn_logits, dim=-1) 
        
        # 5. Weighted aggregation of the Causal Values
        # (B, 1, 3) @ (B, 3, dim) -> (B, 1, dim)
        fused_vector = torch.bmm(attn_weights, V)
        fused_vector = fused_vector.squeeze(1) # (B, dim)
        
        # 6. Normalize and apply learnable scale
        fused_vector = self.out_norm(fused_vector)
        fused_vector = fused_vector * self.output_scale
        target_norm = self.dim ** 0.5
        fused_vector = fused_vector * target_norm / (fused_vector.norm(dim=-1, keepdim=True) + 1e-8)
        
        return fused_vector
    
    
class CausalGatedFusion(nn.Module):
    """
    Clean causal fusion:
    - Q acts as controller (gates), NOT as feature shortcut
    - No direct Q concatenation
    - Forces all modalities to contribute
    """
    def __init__(self, dim):
        super().__init__()
        
        # modality projections (stabilize)
        self.v_proj = nn.Linear(dim, dim)
        self.m_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        self.q_deconf = nn.Linear(dim, dim)

        # question controller → produces gates
        self.q_controller = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 3)   # 
        )
        self.post_fusion_mlp = nn.Sequential(
        nn.Linear(dim * 4, dim * 4),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(dim * 4, dim * 2),
        nn.GELU(),
        nn.Linear(dim * 2, dim),
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, v, motion, causal, q, q_deconf):
    # def forward(self, v, motion, q):
        """
        v: visual graph (B, D)
        motion: motion (B, D)
        causal: causal (B, D)
        q: question (B, D)
        """

        # project modalities
        v = self.v_proj(v)
        motion = self.m_proj(motion)
        causal = self.c_proj(causal)
        q_deconf    = self.q_deconf(q_deconf)

        modalities = torch.stack([v, motion, causal], dim=1)  # (B, 3, D)
        # modalities = torch.stack([v, motion], dim=1)  # (B, 3, D)

        tau = 0.7
        gates = torch.softmax(self.q_controller(q) / tau, dim=-1)

        # apply gating
        gated_modalities = modalities * gates.unsqueeze(-1)
        fused = gated_modalities.reshape(v.size(0), -1)  # (B, 3D)
        fused = self.post_fusion_mlp(torch.cat([fused, q_deconf], dim=-1))

        return self.out_norm(fused), gates