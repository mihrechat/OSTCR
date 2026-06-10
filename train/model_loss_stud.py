import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from model_loss import compute_object_motion_distillation_loss, compute_object_mse_loss


class BaseLoss:
    def __init__(self, weight):
        self.weight = weight

    def compute(self, cfg, batch, outputs):
        raise NotImplementedError

class VQALoss(BaseLoss):
    def __init__(self, weight):
        super().__init__(weight)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # self.criterion = nn.CrossEntropyLoss()

    def compute(self, cfg, batch, outputs):
        logits = outputs["causal_logits"]
        labels = batch.answers.view(-1)

        loss = self.criterion(logits, labels)
        return loss, {"loss_vqa": loss}
    def entropy_reg(self, cfg, outputs):
        branch_scales = outputs["branch_scales"]
        entropy = -(branch_scales * torch.log(branch_scales + 1e-8)).sum(dim=-1).mean()
        loss = -0.01 * entropy
        return loss, {"loss_branch": loss}
    
    # def compute_aux(self,cfg, batch, outputs):
    #     logits = outputs["causal_logits_aux"]
    #     labels = batch.answers.view(-1)
    #     loss    = self.criterion(logits, labels)
    #     return loss, {"loss_causal": loss}

class KLLoss(BaseLoss):
    def compute(self, cfg, batch, outputs):
        node_feat = outputs["node_feat"]
        device = node_feat.device

        total_kl = torch.zeros(1, device=device)

        for b in range(batch.num_graphs):
            mask = (batch.batch == b)
            
            loss_kl = compute_object_motion_distillation_loss(
                node_proj=batch.node_proj[mask].detach().cpu().numpy(),
                node_feat=node_feat[mask],
                node_obj_ids_flat=batch.node_obj_ids_flat[mask].cpu().numpy(),
                node_kf_list_idx=batch.node_kf_list_idx[mask].cpu().numpy(),
                node_is_keyframe=batch.node_is_keyframe[mask].cpu().numpy(),
                num_keyframes=int(batch.num_keyframes[b].item()),
                tau=cfg.train.tau,
                fallback_temp=cfg.train.fallback_temp,
            )
            total_kl += loss_kl

        avg_kl = total_kl / batch.num_graphs
        return avg_kl, {"loss_kl": avg_kl}
       
       
class MSELoss(BaseLoss):
    def compute(self, cfg, batch, outputs):
        node_feat_proj = outputs["node_feat_proj"]
        device = node_feat_proj.device

        total_mse = torch.zeros(1, device=device)

        for b in range(batch.num_graphs):
            mask = (batch.batch == b)

            loss_mse = compute_object_mse_loss(
                node_proj=batch.node_proj[mask].detach().cpu().numpy(),
                node_feat_proj=node_feat_proj[mask],
                node_kf_list_idx=batch.node_kf_list_idx[mask].cpu().numpy(),
                node_is_keyframe=batch.node_is_keyframe[mask].cpu().numpy(),
                num_keyframes=int(batch.num_keyframes[b].item()),
            )
            total_mse += loss_mse

        avg_mse = total_mse / batch.num_graphs
        return avg_mse, {"loss_mse": avg_mse}



LOSS_REGISTRY = {
    "vqa": VQALoss,
    "kl": KLLoss,
    "mse": MSELoss,
}
# LOSS_REGISTRY = {
#     "vqa": VQALoss,
# }

def build_losses(cfg):
    modules = []
    for name in cfg.train.losses:
        weight = getattr(cfg.train, f"lambda_{name}", 1.0)
        modules.append(LOSS_REGISTRY[name](weight))
    return modules
  
  
def compute_total_loss(cfg, batch, outputs, loss_modules):

    # total_loss = 0.0
    total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
    log_dict = {}

    for module in loss_modules:
        loss, log = module.compute(cfg, batch, outputs)
        loss = loss.reshape(())
        weighted = module.weight * loss

        total_loss += weighted
        log_dict.update(log)
    
    if "aux_logits" in outputs:
        aux_logits = outputs['aux_logits']
        labels = batch.answers.view(-1)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        aux_weight = getattr(cfg.train, "aux_loss", 0.3)
        
        aux_loss = criterion(aux_logits, labels) * aux_weight
        total_loss += aux_loss
        
        log_dict['aux_loss'] = aux_loss.detach()
        
        
    if "gates" in outputs:
    
        import math
        
        gate_scales = outputs["gates"]

        entropy = -(gate_scales * torch.log(gate_scales + 1e-8)).sum(dim=-1)

        target_entropy = math.log(gate_scales.size(-1))  # ≈ log(3)

        gate_loss = ((target_entropy - entropy) ** 2).mean()


        lambda_entropy = getattr(cfg.train, "lambda_gate", 0.005)

        gate_loss = lambda_entropy * gate_loss

        total_loss += gate_loss

        log_dict["loss_gate"] = gate_loss.detach()

    log_dict["loss"] = total_loss
    return log_dict
     
