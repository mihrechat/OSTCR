import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

# ==================================================================
# Loss functions
# ==================================================================
def compute_object_motion_distillation_loss(
    node_proj,           # (N, C) numpy float32 — post-merger teacher
    node_feat,           # (N, D)    torch tensor  — ST-Graph student
    node_obj_ids_flat,   # (N,)      numpy int
    node_kf_list_idx,    # (N,)      numpy int
    node_is_keyframe,    # (N,)      numpy bool
    num_keyframes,       # int
    tau:          float = 0.1,
    fallback_temp:float = 0.07,
) -> torch.Tensor:
    """
    Cross-slot motion distillation:
        teacher = Qwen node_proj similarity across Qwen temporal slot boundaries
        student = ST-Graph node_feat similarity across same slots
        loss    = KL(student || teacher)

    Only keyframe nodes used — non-keyframe nodes have stale Qwen grids
    Only cross-slot pairs used — within-slot shares the same Qwen grid 
    """
    SLOT_SIZE  = 2
    T_grid     = num_keyframes // SLOT_SIZE
    total_loss = torch.zeros(1, device=node_feat.device)
    n_pairs    = 0

    for slot in range(T_grid - 1):

        kf_t    = slot * SLOT_SIZE
        kf_next = (slot + 1) * SLOT_SIZE

        rows_t    = np.where((node_kf_list_idx == kf_t)    & node_is_keyframe)[0]
        rows_next = np.where((node_kf_list_idx == kf_next) & node_is_keyframe)[0]

        if len(rows_t) == 0 or len(rows_next) == 0:
            continue

        ids_t    = node_obj_ids_flat[rows_t]
        ids_next = node_obj_ids_flat[rows_next]

        # deduplicate — one row per tracked_id per slot
        _, u_t    = np.unique(ids_t,    return_index=True)
        _, u_next = np.unique(ids_next, return_index=True)
        rows_t    = rows_t[u_t];    ids_t    = node_obj_ids_flat[rows_t]
        rows_next = rows_next[u_next]; ids_next = node_obj_ids_flat[rows_next]

        shared = np.intersect1d(ids_t, ids_next)
        if len(shared) == 0:
            continue

        rows_t    = rows_t   [np.isin(ids_t,    shared)]
        rows_next = rows_next[np.isin(ids_next, shared)]

        # sort by tracked_id — ensures diagonal = same object
        rows_t    = rows_t   [np.argsort(node_obj_ids_flat[rows_t])]
        rows_next = rows_next[np.argsort(node_obj_ids_flat[rows_next])]

        # teacher — Qwen cross-slot (different feature grids → real signal)
        ft = F.normalize(torch.tensor(node_proj[rows_t],    dtype=torch.float32,
                                      device=node_feat.device), dim=-1)
        fn = F.normalize(torch.tensor(node_proj[rows_next], dtype=torch.float32,
                                      device=node_feat.device), dim=-1)
        teacher_prob = torch.softmax(ft @ fn.T / fallback_temp, dim=-1)

        # student — ST-Graph cross-slot
        gt = F.normalize(node_feat[rows_t],    dim=-1)
        gn = F.normalize(node_feat[rows_next], dim=-1)
        graph_prob = torch.softmax(gt @ gn.T / tau, dim=-1)

        total_loss = total_loss + F.kl_div(
            graph_prob.log(), teacher_prob, reduction="batchmean"
        )
        n_pairs += 1

    return total_loss / max(n_pairs, 1)

# def compute_object_motion_distillation_loss(
#     node_proj,           # (N, C) torch tensor (or numpy, handled below)
#     node_feat,           # (N, D) torch tensor
#     node_obj_ids_flat,   # (N,)   torch tensor
#     node_kf_list_idx,    # (N,)   torch tensor
#     node_is_keyframe,    # (N,)   torch tensor (bool)
#     num_keyframes,       # int
#     tau: float = 0.1,
#     fallback_temp: float = 0.07,
# ) -> torch.Tensor:

#     device = node_feat.device

#     # --- ensure torch tensors ---
#     if not torch.is_tensor(node_proj):
#         node_proj = torch.tensor(node_proj, dtype=torch.float32, device=device)
#     else:
#         node_proj = node_proj.to(device)

#     node_obj_ids_flat = node_obj_ids_flat.to(device)
#     node_kf_list_idx  = node_kf_list_idx.to(device)
#     node_is_keyframe  = node_is_keyframe.to(device)

#     SLOT_SIZE = 2
#     T_grid = num_keyframes // SLOT_SIZE

#     total_loss = torch.tensor(0.0, device=device)
#     n_pairs = 0

#     for slot in range(T_grid - 1):

#         kf_t    = slot * SLOT_SIZE
#         kf_next = (slot + 1) * SLOT_SIZE

#         rows_t = torch.where(
#             (node_kf_list_idx == kf_t) & node_is_keyframe
#         )[0]

#         rows_next = torch.where(
#             (node_kf_list_idx == kf_next) & node_is_keyframe
#         )[0]

#         if rows_t.numel() == 0 or rows_next.numel() == 0:
#             continue

#         ids_t    = node_obj_ids_flat[rows_t]
#         ids_next = node_obj_ids_flat[rows_next]

#         # --- deduplicate (first occurrence per object id) ---
#         unique_ids_t = torch.unique(ids_t)
#         unique_ids_next = torch.unique(ids_next)

#         rows_t = torch.stack([
#             rows_t[(ids_t == uid).nonzero(as_tuple=True)[0][0]]
#             for uid in unique_ids_t
#         ])

#         rows_next = torch.stack([
#             rows_next[(ids_next == uid).nonzero(as_tuple=True)[0][0]]
#             for uid in unique_ids_next
#         ])

#         ids_t    = node_obj_ids_flat[rows_t]
#         ids_next = node_obj_ids_flat[rows_next]

#         # --- intersection of object IDs ---
#         shared = torch.tensor(
#             list(set(ids_t.tolist()) & set(ids_next.tolist())),
#             device=device,
#             dtype=ids_t.dtype
#         )

#         if shared.numel() == 0:
#             continue

#         mask_t    = torch.isin(ids_t, shared)
#         mask_next = torch.isin(ids_next, shared)

#         rows_t    = rows_t[mask_t]
#         rows_next = rows_next[mask_next]

#         ids_t     = node_obj_ids_flat[rows_t]
#         ids_next  = node_obj_ids_flat[rows_next]

#         # --- sort by object id (align diagonal) ---
#         order_t    = torch.argsort(ids_t)
#         order_next = torch.argsort(ids_next)

#         rows_t    = rows_t[order_t]
#         rows_next = rows_next[order_next]

#         # --- teacher (Qwen) ---
#         ft = F.normalize(node_proj[rows_t].to(torch.float32), dim=-1)
#         fn = F.normalize(node_proj[rows_next].to(torch.float32), dim=-1)

#         teacher_prob = torch.softmax((ft @ fn.T) / fallback_temp, dim=-1)

#         # --- student (ST-Graph) ---
#         gt = F.normalize(node_feat[rows_t], dim=-1)
#         gn = F.normalize(node_feat[rows_next], dim=-1)

#         graph_prob = torch.softmax((gt @ gn.T) / tau, dim=-1)

#         # --- KL divergence ---
#         loss = F.kl_div(
#             graph_prob.log(),
#             teacher_prob,
#             reduction="batchmean"
#         )

#         total_loss = total_loss + loss
#         n_pairs += 1

#     if n_pairs == 0:
#         return torch.tensor(0.0, device=device)

#     return (total_loss / n_pairs).reshape(())


def compute_object_mse_loss(
    node_proj,           # (N, 2048) numpy float32 — Qwen post-merger teacher
    node_feat_proj,      # (N, 2048) torch tensor  — ST-Graph projected to Qwen space
    node_kf_list_idx,    # (N,)      numpy int
    node_is_keyframe,    # (N,)      numpy bool
    num_keyframes,       # int
) -> torch.Tensor:
    """
    Per-slot MSE between ST-Graph projected features and Qwen post-merger targets.
    Anchors the graph representation to Qwen's visual feature space.
    Only uses first keyframe per slot to avoid stale within-slot supervision.
    """
    SLOT_SIZE  = 2
    T_grid     = num_keyframes // SLOT_SIZE
    total_loss = torch.zeros(1, device=node_feat_proj.device)
    n_slots    = 0

    for slot in range(T_grid):
        kf_rep = slot * SLOT_SIZE
        rows   = np.where((node_kf_list_idx == kf_rep) & node_is_keyframe)[0]

        if len(rows) == 0:
            continue

        target = torch.tensor(node_proj[rows], dtype=torch.float32,
                              device=node_feat_proj.device)
        # target = node_proj[rows].detach().clone().to(torch.float32)
        pred   = node_feat_proj[rows]

        total_loss = total_loss + F.mse_loss(pred, target)
        n_slots   += 1

    return total_loss / max(n_slots, 1)


# import torch
# import torch.nn.functional as F

# def calc_focal_loss(logits, labels, alpha=1.0, gamma=2):
#     """
#     Focal Loss heavily penalizes hard examples (like 'What' questions)
#     and reduces the loss for easy examples (like 'Who' questions).
#     """
#     # 1. Standard Cross Entropy (Shape: [Batch_Size])
#     ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
#     # 2. Recover the probability of the correct target class
#     pt = torch.exp(-ce_loss)
    
#     # 3. Apply the focal weighting: (1 - pt)^gamma
#     loss = (alpha * ((1 - pt) ** gamma) * ce_loss).mean()
    
#     return loss
# def compute_total_loss(
#     batch,                # PyG batched graph
#     node_feat,            # (N_total, D)     model raw output
#     node_feat_proj,       # (N_total, 2048)  model projected output
#     causal_logits,
#     lambda_kl:   float = 1.0,
#     lambda_mse:  float = 0.5,
#     lambda_vqa:  float = 1.0,  
#     lambda_focal:float = 1.0,
#     tau:         float = 0.1,
#     fallback_temp: float = 0.07,
# ) -> dict:
#     """
#     Compute combined loss over all videos in the batch.
#     Loops over per-video masks using batch.batch bookkeeping tensor.
#     """
#     # total_kl  = torch.zeros(1, device=node_feat.device)
#     # total_mse = torch.zeros(1, device=node_feat.device)
#     total_kl = torch.tensor(0.0, device=node_feat.device)
#     total_mse = torch.tensor(0.0, device=node_feat.device)
#     n_graphs  = batch.num_graphs

#     for b in range(n_graphs):
#         mask = (batch.batch == b)

#         # per-video numpy arrays for indexing
#         node_proj_b         = batch.node_proj[mask].detach().cpu().numpy()
#         node_obj_ids_flat_b = batch.node_obj_ids_flat[mask].cpu().numpy()
#         node_kf_list_idx_b  = batch.node_kf_list_idx[mask].cpu().numpy()
#         node_is_keyframe_b  = batch.node_is_keyframe[mask].cpu().numpy()
#         num_keyframes_b     = int(batch.num_keyframes[b].item())

#         loss_kl = compute_object_motion_distillation_loss(
#             node_proj         = node_proj_b,
#             node_feat         = node_feat[mask],
#             node_obj_ids_flat = node_obj_ids_flat_b,
#             node_kf_list_idx  = node_kf_list_idx_b,
#             node_is_keyframe  = node_is_keyframe_b,
#             num_keyframes     = num_keyframes_b,
#             tau               = tau,
#             fallback_temp     = fallback_temp,
#         )

#         loss_mse = compute_object_mse_loss(
#             node_proj        = node_proj_b,
#             node_feat_proj   = node_feat_proj[mask],
#             node_kf_list_idx = node_kf_list_idx_b,
#             node_is_keyframe = node_is_keyframe_b,
#             num_keyframes    = num_keyframes_b,
#         )

#         total_kl  = total_kl  + loss_kl
#         total_mse = total_mse + loss_mse

#     # avg_kl  = total_kl  / n_graphs
#     avg_mse = (total_mse / batch.num_graphs).squeeze()
#     avg_kl = (total_kl / batch.num_graphs).squeeze()
#     # avg_mse = total_mse / n_graphs
    
#     vqa_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
#     # Squeeze batch.answers to ensure it is strictly 1D: (B,)
#     # labels = batch.answers.squeeze() 
#     labels = batch.answers.view(-1) 
#     loss_vqa = vqa_criterion(causal_logits, labels)
#     labels = batch.answers.view(-1) 
#     # focal_loss = calc_focal_loss(causal_logits, labels)
#     total = (lambda_kl * avg_kl) + (lambda_mse * avg_mse) + (loss_vqa * lambda_vqa)

#     return {
#         "loss":     total,
#         "loss_kl":  avg_kl,
#         "loss_mse": avg_mse,
#         "loss_vqa": loss_vqa
#         # "loss_focal": focal_loss,

#     }