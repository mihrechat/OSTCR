import torch
import torch.nn as nn
import os

class HierarchicalLinguisticPriorBank(nn.Module):
    def __init__(self, qtype_pt_path=None, qrole_pt_path=None):
        super().__init__()
        if qtype_pt_path is None or qrole_pt_path is None:
            raise FileNotFoundError("############## Please provide both qtype and triplet paths #################")
        # Load precomputed expectations and freeze them
        E_T = torch.load(qtype_pt_path)
        E_tau = torch.load(qrole_pt_path)
        
        self.qtype_embedding = nn.Embedding.from_pretrained(E_T, freeze=True)
        self.triplet_embedding = nn.Embedding.from_pretrained(E_tau, freeze=True, padding_idx=0)

    def forward(self, qtype_idx: torch.Tensor, triplet_idxs: torch.Tensor, triplet_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            qtype_idx: (B,) tensor of question type indices
            triplet_idxs: (B, N_q) tensor of syntactic triplet indices
            triplet_mask: (B, N_q) boolean tensor where True means valid triplet
        Returns:
            E_cl: (B, dim) The combined expected linguistic confounder
        """
        # 1. Macro-level expectation E_T (B, dim)
        E_T_batch = self.qtype_embedding(qtype_idx)
        
        # 2. Micro-level expectation E_tau (B, N_q, dim)
        E_tau_batch = self.triplet_embedding(triplet_idxs)
        
        # Average over valid triplets for each question
        # Shape: (B, dim)
        valid_counts = triplet_mask.sum(dim=1, keepdim=True).clamp(min=1)
        E_tau_mean = (E_tau_batch * triplet_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        
        # 3. Final Instance-Specific Confounder
        E_cl = E_T_batch + E_tau_mean
        return E_cl
    
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
    
    
# class CausalPriorBank(nn.Module):
#     def __init__(self, prior_pt_path, num_classes, embedding_dim):
#         super().__init__()
#         prior_weight = torch.load(prior_pt_path)
#         self.empr_prior = nn.Embedding.from_pretrained(prior_weight, freeze=True)
#         self.prior = nn.Embedding(num_classes, embedding_dim)
#     def forward(self, class_ids):
#         #torch.cat([self.prior(class_ids), self.empr_prior(class_ids)], dim=-1)
#         return self.prior(class_ids) * self.empr_prior(class_ids)
#         # return self.prior(class_ids) 
