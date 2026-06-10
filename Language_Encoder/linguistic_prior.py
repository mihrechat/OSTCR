import torch
import torch.nn as nn
import os
import math


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d]
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, L, d]
        """

        L = x.size(1)

        x = x + self.pe[:, :L]

        return self.dropout(x)
    

class PositionalEncodingLearned1D(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.pos_embed = nn.Embedding(
            max_len,
            d_model
        )

        nn.init.trunc_normal_(
            self.pos_embed.weight,
            std=0.02
        )

    def forward(self, x):
        """
        x: [B, L, d]
        """

        B, L, d = x.shape

        idx = torch.arange(
            L,
            device=x.device
        )

        pos = self.pos_embed(idx)  # [L,d]

        x = x + pos.unsqueeze(0)

        return self.dropout(x)


class HierarchicalLinguisticPriorBank(nn.Module):
    def __init__(self, qtype_pt_path=None, qrole_pt_path=None):
        super().__init__()
        if qtype_pt_path is None or qrole_pt_path is None:
            raise FileNotFoundError("############## Please provide both qtype and triplet paths #################")
        
        E_T = torch.load(qtype_pt_path)      # (V_qtype, 1024)
        E_tau = torch.load(qrole_pt_path)    # (V_role, 1024)
        
        self.qtype_embedding = nn.Embedding.from_pretrained(E_T, freeze=True)
        self.triplet_embedding = nn.Embedding.from_pretrained(E_tau, freeze=True, padding_idx=0)
        
        
    def forward(self, qtype_idx: torch.Tensor, triplet_idxs: torch.Tensor, triplet_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            qtype_idx: (B,) tensor of question type indices
            triplet_idxs: (B, N_q) tensor of syntactic triplet indices
            triplet_mask: (B, N_q) boolean tensor where True means valid triplet
        Returns:
            E_cl: (B, N_q + 1, dim) Concatenated linguistic features
            E_cl_mask: (B, N_q + 1) Boolean mask
        """
        
        # =====================================================
        # SHAPE CHECKS & FIXES
        # =====================================================
        # Fix qtype_idx: ensure it's 1D
        if qtype_idx.dim() == 0:
            qtype_idx = qtype_idx.unsqueeze(0)  # () -> (1,)
        elif qtype_idx.dim() == 2 and qtype_idx.shape[1] == 1:
            qtype_idx = qtype_idx.squeeze(1)  # (B,1) -> (B,)
        
        # Fix triplet_idxs: ensure it's 2D
        if triplet_idxs.dim() == 1:
            triplet_idxs = triplet_idxs.unsqueeze(0)  # (N_q,) -> (1, N_q)
        elif triplet_idxs.dim() == 3:
            if triplet_idxs.shape[-1] == 1:
                triplet_idxs = triplet_idxs.squeeze(-1)  # (B, N_q, 1) -> (B, N_q)
            else:
                triplet_idxs = triplet_idxs.view(triplet_idxs.shape[0], -1)  # Flatten last dims
        
        # Fix triplet_mask: ensure same shape as triplet_idxs
        if triplet_mask.dim() == 1:
            triplet_mask = triplet_mask.unsqueeze(0)  # (N_q,) -> (1, N_q)
        elif triplet_mask.dim() == 3:
            triplet_mask = triplet_mask.squeeze(-1)  # (B, N_q, 1) -> (B, N_q)
        
        # =====================================================
        # EMBEDDING
        # =====================================================
        B = qtype_idx.shape[0]
        N_q = triplet_idxs.shape[1]
        device = qtype_idx.device
        
        
        # 1. Macro-level: (B, 1, dim)
        E_T_batch = self.qtype_embedding(qtype_idx)  # (B, dim)
        E_T_batch = E_T_batch.unsqueeze(1)  # (B, 1, dim)
        
        # 2. Micro-level: (B, N_q, dim)
        E_tau_batch = self.triplet_embedding(triplet_idxs)  # (B, N_q, dim)
        
        # Mask out invalid triplets (set to zero)
        triplet_mask_expanded = triplet_mask.unsqueeze(-1)  # (B, N_q, 1)
        E_tau_batch = E_tau_batch * triplet_mask_expanded
        
        # 3. Concatenate along sequence dimension
        E_cl = torch.cat([E_T_batch, E_tau_batch], dim=1)  # (B, N_q + 1, dim)
        
        # 4. Create combined mask
        qtype_mask = torch.ones(B, 1, dtype=torch.bool, device=device)  # (B, 1)
        E_cl_mask = torch.cat([qtype_mask, triplet_mask], dim=1)  # (B, N_q + 1)
        
        assert E_cl.shape == (B, N_q + 1, self.qtype_embedding.embedding_dim), \
            f"E_cl shape {E_cl.shape} vs expected {(B, N_q + 1, self.qtype_embedding.embedding_dim)}"
        assert E_cl_mask.shape == (B, N_q + 1), \
            f"E_cl_mask shape {E_cl_mask.shape} vs expected {(B, N_q + 1)}"
        
        return E_cl, E_cl_mask


class ConceptNetExpectedPriorBank(nn.Module):
    def __init__(self, prior_pt_path="None"):
        super().__init__()
        if prior_pt_path is None:
            raise FileNotFoundError(f"Please provide prior pt path ###### current path: {prior_pt_path} is None or not working")
        if not os.path.exists(prior_pt_path):
            raise FileNotFoundError(f"The given file: {prior_pt_path} doesn't exist or work")
        priors = torch.load(prior_pt_path)
        self.embedding = nn.Embedding.from_pretrained(priors, freeze=True)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(class_ids)
    

class CausalPriorBank(nn.Module):
    def __init__(self, prior_pt_path, num_classes, embedding_dim):
        super().__init__()
        # Load 1D empirical prior
        prior_weight = torch.load(prior_pt_path)        # [num_classes]
        # Expand to 2D
        if prior_weight.dim() == 1:
            prior_weight = prior_weight.unsqueeze(1).repeat(1, embedding_dim)  # [num_classes, embedding_dim]
        self.empr_prior = nn.Embedding.from_pretrained(prior_weight, freeze=True)
        self.prior = nn.Embedding(num_classes, embedding_dim)
    def forward(self, class_ids):
        return self.prior(class_ids) * self.empr_prior(class_ids)
    
 
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLanguageEncoder(nn.Module):
    """
    LAM for VideoQA
    """

    def __init__(
        self,
        in_dim=1024,
        model_dim=512,
        num_heads=8,
        num_transformer_layers=2,
        gru_layers=1,
        pos_flag='learned',
        dropout=0.1
    ):
        super().__init__()

        self.model_dim = model_dim
        
        embed_dim = model_dim
        self.pos_flag = pos_flag
        if pos_flag == 'sincos':

            self.embed_scale = math.sqrt(embed_dim)

            self.pos_encoder = PositionalEncoding(
                embed_dim,
                dropout
            )

        elif pos_flag == 'learned':

            self.embed_scale = 1.0

            self.pos_encoder = PositionalEncodingLearned1D(
                embed_dim,
                dropout
            )

        else:

            self.embed_scale = 1.0
            self.pos_encoder = nn.Identity()


        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )


        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({

                "norm1": nn.LayerNorm(model_dim),

                "attn": nn.MultiheadAttention(
                    embed_dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),

                "drop1": nn.Dropout(dropout),

                "norm2": nn.LayerNorm(model_dim),

                "ffn": nn.Sequential(
                    nn.Linear(model_dim, model_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim * 4, model_dim)
                ),

                "drop2": nn.Dropout(dropout)

            })
            for _ in range(num_transformer_layers)
        ])

        self.bigru = nn.GRU(
            input_size=model_dim,
            hidden_size=model_dim // 2,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True
        )

        self.output_norm = nn.LayerNorm(model_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )


        self.token_score = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1)
        )

    def forward(
        self,
        x,          # [B,L,in_dim]
        mask=None   # [B,L]
    ):

        x = self.input_proj(x)
        
        x = self.pos_encoder(
            self.embed_scale * x
        )

    

        for layer in self.transformer_layers:

            h = layer["norm1"](x)

            attn_out, _ = layer["attn"](
                query=h,
                key=h,
                value=h,
                key_padding_mask=~mask if mask is not None else None
            )

            x = x + layer["drop1"](attn_out)

            h = layer["norm2"](x)

            ffn_out = layer["ffn"](h)

            x = x + layer["drop2"](ffn_out)


        if mask is not None:

            lengths = mask.sum(dim=1).cpu()

            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            packed_out, _ = self.bigru(packed)

            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=mask.size(1)
            )

        else:

            x, _ = self.bigru(x)


        x = self.output_norm(x)

        x = self.output_proj(x)


        token_scores = self.token_score(x).squeeze(-1)

        if mask is not None:

            token_scores = token_scores.masked_fill(
                ~mask,
                -1e9
            )

        token_weights = F.softmax(
            token_scores,
            dim=-1
        )


        global_question = torch.sum(
            x * token_weights.unsqueeze(-1),
            dim=1
        )

        return {
            "token_features": x,                   # [B,L,d]
            "token_weights": token_weights,        # [B,L]
            "global_question": global_question     # [B,d]
        }
