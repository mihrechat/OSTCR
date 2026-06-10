
import sys
import os
# Adds the 'CausalNet' directory to the python path
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from transformers import get_cosine_schedule_with_warmup
import os
import csv
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data, Batch as PyGBatch
from data_class_open_ended import get_args
from build_dataset import MSVDDatasetMotion
from torch.optim.lr_scheduler import ReduceLROnPlateau,  CosineAnnealingLR, LinearLR
from torch.amp import GradScaler, autocast
from models.visualEncoder.msvd_revised_transformer import STGraphTransformerNet
from model_loss_stud import build_losses, compute_total_loss
from build_dataset import stgraph_collate

CUDA_LAUNCH_BLOCKING=1
class EarlyStopping:
    """FIXED: Proper early stopping with patience and best model saving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
        
        
# class EarlyStopping:
#     def __init__(self, patience: int = 10, min_delta: float = 1e-4):
#         self.patience   = patience
#         self.min_delta  = min_delta
#         self.best_loss  = float("inf")
#         self.counter    = 0
#         self.should_stop = False

#     def step(self, val_loss: float) -> bool:
#         if val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter   = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.should_stop = True
#         return self.should_stop

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    best_loss,
    best_acc,
    cfg,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "epoch": epoch,

        "model": model.state_dict(),

        "optimizer": optimizer.state_dict(),

        "scheduler": scheduler.state_dict(),

        "scaler": scaler.state_dict(),

        "ema": ema.state_dict(),

        "best_loss": best_loss,
        "best_acc": best_acc,

        "config": cfg,
    }, path)
    

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    device,
):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model"]
    )

    optimizer.load_state_dict(
        ckpt["optimizer"]
    )

    scheduler.load_state_dict(
        ckpt["scheduler"]
    )

    scaler.load_state_dict(
        ckpt["scaler"]
    )

    if "ema" in ckpt:
        ema.load_state_dict(
            ckpt["ema"]
        )

    start_epoch = ckpt["epoch"] + 1

    best_loss = ckpt["best_loss"]
    best_acc  = ckpt["best_acc"]

    print(
        f"Loaded checkpoint from epoch "
        f"{ckpt['epoch']}"
    )

    return start_epoch, best_loss, best_acc


# def save_checkpoint(
#     path:      str,
#     epoch:     int,
#     model:     torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler,
#     scaler:    GradScaler,
#     best_loss: float,
#     best_acc:  float,
#     cfg,       # ExperimentConfig
# ):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
    
#     # Save the state dictionaries and the config object
#     torch.save({
#         "epoch":          epoch,
#         "model":          model.state_dict(),
#         "optimizer":      optimizer.state_dict(),
#         "scheduler":      scheduler.state_dict(),
#         "scaler":         scaler.state_dict(),
#         "best_loss":      best_loss,
#         "best_acc":       best_acc,
#         "config":         cfg,
#     }, path)


# def load_checkpoint(
#     path:      str,
#     model:     torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler,
#     scaler:    GradScaler,
#     device:    torch.device,
# ) -> tuple:
#     ckpt = torch.load(path, map_location=device, weights_only=False)
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     scheduler.load_state_dict(ckpt["scheduler"])
#     scaler.load_state_dict(ckpt["scaler"])
    
#     return ckpt["epoch"], ckpt["best_loss"], ckpt["best_acc"]


# ==================================================================
# CSV logger
# ==================================================================
class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        # Safely create directory if it doesn't exist
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_kl", "train_mse", "train_vqa",
                "val_loss", "val_kl", "val_mse", "val_vqa",
                "lr", "epoch_time_s", "gpu_mem_mb"
            ])

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row.get("epoch",         ""),
                row.get("train_loss",    ""),
                row.get("train_kl",      ""),
                row.get("train_mse",     ""),
                row.get("train_vqa",     ""),
                row.get("val_loss",      ""),
                row.get("val_kl",        ""),
                row.get("val_mse",       ""),
                row.get("val_vqa",       ""),
                row.get("lr",            ""),
                row.get("epoch_time_s",  ""),
                row.get("gpu_mem_mb",    ""),
            ])

# ==================================================================
# One epoch — train
# ==================================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,  # Passed in but NOT stepped here anymore
    scaler:     GradScaler,
    cfg,        # ExperimentConfig
    loss_modules,
    device:     torch.device,
    epoch:      int,
    writer,
    log,
    ema,
    current_scheduler,
    global_batch_idx,
    warmup_iters
    
) -> dict:

    model.train()
    optimizer.zero_grad(set_to_none=True)   # set_to_none saves memory vs zero_grad()

    total_loss = 0.0
    total_kl = 0.0
    total_mse = 0.0
    total_vqa = 0.0 
    total_entropy = 0.0
    total_loss_entropy = 0.0 
    total_aux = 0.0
    total_gate = 0.0
    n_batches  = len(loader)
    global_step = epoch * n_batches
    total_correct = 0
    total_samples = 0
    nan_count     = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        # ── forward (mixed precision - updated modern syntax) ──────
        with torch.autocast(device_type=device.type, enabled=cfg.system.use_amp, dtype=torch.bfloat16):
            outputs = model(batch)
            
           
            losses = compute_total_loss(
              cfg=cfg,
              batch=batch,
              outputs=outputs,
              loss_modules=loss_modules,
             )
            

            # scale loss for gradient accumulation
            loss = losses["loss"] / cfg.train.grad_accum_steps
        if torch.isnan(loss):
                nan_count += 1
                log.warning(f"NaN loss at step {step}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
      
        # ── Accuracy Calculation ──
        
        causal_logits  = outputs["causal_logits"] 
        labels = batch.answers.view(-1) 
        preds = causal_logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        # ── backward (scaler handles fp16 underflow) ──────────────
        scaler.scale(loss).backward()
        
        # ── optimizer step every grad_accum_steps ─────────────────
        if (step + 1) % cfg.train.grad_accum_steps == 0 or (step + 1) == n_batches:

            # unscale before clipping so clip operates on true gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)

            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            
            if current_scheduler == 'warmup':
                scheduler['warmup'].step()
             
                if global_batch_idx + step >= warmup_iters:
                    current_scheduler = 'plateau'
                    log.info(f"✅ Warmup complete at batch {global_batch_idx + step}! Switching to ReduceLROnPlateau")
            
            
            
        # ── accumulate metrics ────────────────────────────────────
        total_loss += losses["loss"].item()
        total_kl   += losses.get("loss_kl",  torch.tensor(0.0)).item()
        total_mse  += losses.get("loss_mse", torch.tensor(0.0)).item()
        total_vqa  += losses.get("loss_vqa", torch.tensor(0.0)).item()
        total_aux   += losses.get('aux_loss', torch.tensor(0.0)).item()
        total_gate +=  losses.get('loss_gate', torch.tensor(0.0)).item()
        total_entropy += losses.get("entropy", 0.0)
        total_loss_entropy += losses.get("loss_entropy", 0.0)
        
        
         
        # ── per-step tensorboard ──────────────────────────────────
        if writer:
            writer.add_scalar("step/loss",     losses["loss"].item(),     global_step + step)
            if "loss_kl" in losses:
               writer.add_scalar("step/loss_kl",  losses["loss_kl"].item(),  global_step + step)
            if "loss_mse" in losses:
               writer.add_scalar("step/loss_mse", losses["loss_mse"].item(), global_step + step)
            writer.add_scalar("step/loss_vqa", losses["loss_vqa"].item(), global_step + step)
            writer.add_scalar("step/lr", optimizer.param_groups[0]["lr"], global_step + step)
            

    n = max(n_batches, 1)
    if nan_count > 0:
        log.warning(f"Encountered {nan_count} NaN losses during training")
    return {
        "loss":     total_loss / n,
        "loss_kl":  total_kl   / n,
        "loss_mse": total_mse  / n,
        "loss_vqa": total_vqa  / n,
        "entropy": total_entropy / n,
        "loss_entropy": total_loss_entropy / n,
        "loss_aux":   total_aux / n,
        'loss_gate':  total_gate / n,
        "acc": total_correct / max(total_samples, 1), 
    }, current_scheduler, global_batch_idx + n_batches


# ==================================================================
# One epoch — validate
# ==================================================================
@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    cfg,
    loss_modules,
    device: torch.device,
    log,
    ema,
) -> dict:

    ema.module.eval() if ema is not None else model.eval()
    
    total_loss = total_kl = total_mse = total_vqa = total_entropy = total_entropy_loss =  total_aux = total_gate = 0.0
    n_batches  = len(loader)
    total_correct = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.autocast(device_type=device.type, enabled=cfg.system.use_amp, dtype=torch.bfloat16):
            # outputs    = model(batch)
            eval_model = ema.module if ema is not None else model
            outputs = eval_model(batch)
            logits  = outputs["causal_logits"]   # (N, num_classes)
            
            losses = compute_total_loss(
              cfg=cfg,
              batch=batch,
              outputs=outputs,
              loss_modules=loss_modules,
             )
            
        preds = logits.argmax(dim=-1)
        labels = batch.answers.view(-1) # FIXED: view(-1) is safer than squeeze(-1)
        
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        #####    loss ###################
        total_loss += losses["loss"].item()
        total_kl   += losses.get("loss_kl",  torch.tensor(0.0)).item()
        total_mse  += losses.get("loss_mse", torch.tensor(0.0)).item()
        total_vqa  += losses.get("loss_vqa", torch.tensor(0.0)).item()
        total_aux   += losses.get('aux_loss', torch.tensor(0.0)).item()
        total_gate +=  losses.get('loss_gate', torch.tensor(0.0)).item()
        total_entropy += losses.get("entropy", 0.0)
        total_entropy_loss += losses.get("loss_entropy", 0.0)

    n = max(n_batches, 1)
    return {
        "loss":     total_loss / n,
        "loss_kl":  total_kl   / n,
        "loss_mse": total_mse  / n,
        "loss_vqa": total_vqa  / n,
        "entropy": total_entropy / n,
        "loss_entropy": total_entropy_loss / n,
        "loss_aux": total_aux / n,
        'loss_gate':  total_gate / n,
        "acc": total_correct / total_samples if total_samples > 0 else 0.0,
    }



# ==================================================================
# GPU memory helper
# ==================================================================
def gpu_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device) / 1024 ** 2
    return 0.0

def param_groups(model):

        encoder_params = []
        fusion_params = []
        classifier_params = []
        no_decay = []

        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            # ---------------------------------------------
            # no weight decay
            # ---------------------------------------------

            if (
                param.ndim == 1
                or name.endswith(".bias")
                or "norm" in name.lower()
            ):
                no_decay.append(param)
                continue

            # ---------------------------------------------
            # language encoder
            # ---------------------------------------------

            if "language_encoder" in name:
                encoder_params.append(param)

            # ---------------------------------------------
            # classifier
            # ---------------------------------------------

            elif "classifier" in name:
                classifier_params.append(param)

            # ---------------------------------------------
            # fusion / transformers
            # ---------------------------------------------

            else:
                fusion_params.append(param)

        return [
            {
                "params": encoder_params,
                "lr": 5e-5,
                "weight_decay": 0.05,
            },
            {
                "params": fusion_params,
                "lr": cfg.train.fusion_lr,
                "weight_decay": 0.05,
            },
            {
                "params": classifier_params,
                "lr": cfg.train.classifier_lr,
                "weight_decay": 0.05,
            },
            {
                "params": no_decay,
                "lr": cfg.train.no_decay_lr,
                "weight_decay": 0.0,
            }
        ]
# ==================================================================
# Main training function
# ==================================================================
def train(
    model,
    train_ds,
    val_ds,
    cfg,  
):
    # ── reproducibility ───────────────────────────────────────────
    torch.manual_seed(cfg.system.seed)
    np.random.seed(cfg.system.seed)

    # ── device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(device)}")
        print(f"VRAM total  : {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"AMP enabled : {cfg.system.use_amp}")
    print(f"Grad accum  : {cfg.train.grad_accum_steps} steps "
          f"(effective batch = {cfg.train.batch_size * cfg.train.grad_accum_steps})")
    print(f"{'='*60}\n")

    model = model.to(device)
    loss_modules = build_losses(cfg)

    # ── train / val split ─────────────────────────────────────────
    n_train = len(train_ds)
    n_val   = len(val_ds)
    print(f"Train videos : {n_train} | Val videos : {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.train.batch_size,
        shuffle     = True,
        num_workers = cfg.data.num_workers,
        collate_fn  = stgraph_collate, 
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.train.batch_size,
        shuffle     = False,
        num_workers = cfg.data.num_workers,
        collate_fn  = stgraph_collate,
        pin_memory  = device.type == "cuda",
        drop_last=True,
    )

    # ── optimizer ──────────────��──────────────────────────────────
    
    optimizer = torch.optim.AdamW(
    param_groups(model),
    lr=cfg.train.lr,
    betas=(0.9, 0.98),
    eps=1e-8
)
    from timm.utils import ModelEmaV3

    ema = ModelEmaV3(
        model,
        decay=0.999
    )
  

    # ── scheduler — ReduceLROnPlateau  ─────────────────────
    
    
    warmup_iters = len(train_loader) * cfg.train.warmup_epochs
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_iters
)
    # =========================================================
    # Plateau scheduler
    # =========================================================

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          # monitor validation accuracy
        factor=0.5,          # reduce LR by half
        patience=2,          # wait 2 epochs
        threshold=0.002,     # minimum improvement
        min_lr=1e-6,
    )
    
    scheduler = {
        "warmup": warmup_scheduler,
        "plateau": plateau_scheduler
    }
    current_scheduler = "warmup"
    
    # total_steps = (
    # len(train_loader)
    # * cfg.train.epochs)

    # warmup_steps = int(
    #     cfg.train.warmup * total_steps)

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    
    # ── AMP scaler ────────────────────────────────────────────────
    scaler = GradScaler(device.type, enabled=cfg.system.use_amp)
    
    # ── logging ───────────────────────────────────────────────────
    os.makedirs(cfg.system.save_dir, exist_ok=True)
    
    csv_logger = CSVLogger(os.path.join(cfg.system.save_dir, "metrics.csv"))

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(cfg.system.save_dir, "train.log")),
        ]
    )
    log = logging.getLogger(__name__)

    best_loss  = float("inf")
    best_acc   = 0.0
    start_epoch = 0
    global_batch_idx = 0 

    # ── resume from checkpoint ────────────────────────────────────
    if (cfg.system.resume_from and os.path.exists(cfg.system.resume_from)):
        start_epoch, best_loss, best_acc = load_checkpoint(cfg.system.resume_from,model,optimizer,scheduler[current_scheduler],scaler,ema,device)
        if start_epoch > warmup_iters // len(train_loader):
            current_scheduler = "plateau"
        log.info(f"Resumed from {cfg.system.resume_from} — epoch {start_epoch}, best_loss={best_loss:.6f}, best_acc={best_acc:.4f}")
       
    # ── training loop ─────────────────────────────────────────────
    log.info(f"Starting training — {cfg.train.epochs} epochs")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        # ── train ──────────────────────────────────────────────────
        # train_metrics = train_one_epoch(
        #     model     = model,
        #     loader    = train_loader,
        #     optimizer = optimizer,
        #     scheduler = scheduler,
        #     scaler    = scaler,
        #     cfg       = cfg,
        #     loss_modules = loss_modules,
        #     device    = device,
        #     epoch     = epoch,
        #     writer    = None,
        #     log       = log,
        #     ema=ema
        # )
        train_metrics, current_scheduler, global_batch_idx = train_one_epoch(
            model     = model,
            loader    = train_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            scaler    = scaler,
            cfg       = cfg,
            loss_modules = loss_modules,
            device    = device,
            epoch     = epoch,
            writer    = None,
            log       = log,
            ema=ema,
            current_scheduler = current_scheduler,
            global_batch_idx = global_batch_idx,
            warmup_iters = warmup_iters
        )

        # ── validate ───────────────────────────────────────────────
        val_metrics = validate_one_epoch(
            model  = model,
            loader = val_loader,
            cfg    = cfg,
            loss_modules = loss_modules,
            device = device,
            log    = log,
            ema=ema
        )
        
        if current_scheduler == "plateau":
           scheduler['plateau'].step(val_metrics['acc'])
        else:
            log.info(f"Warmup in progress... ({global_batch_idx}/{warmup_iters} batches)")

        epoch_time = time.time() - t0
        mem_mb     = gpu_mem_mb(device)
        lr_now     = optimizer.param_groups[1]["lr"]

        # ── logging ────────────────────────────────────────────────
        log.info(
            f"Epoch {epoch+1:03d}/{cfg.train.epochs} | "
            f"train={train_metrics['loss']:.4f} "
            # f"(vqa={train_metrics['loss_vqa']:.4f} kl={train_metrics['loss_kl']:.4f} mse={train_metrics['loss_mse']:.4f} acc={train_metrics['acc']:.4f}) | "
            f"(vqa={train_metrics['loss_vqa']:.4f} acc={train_metrics['acc']:.4f}) | "
            # f"(vqa={train_metrics['loss_vqa']:.4f} acc={train_metrics['acc']:.4f} entropy_loss={train_metrics["loss_entropy"]:.4f} entr={train_metrics['entropy']:.4f}) | "
            f"val={val_metrics['loss']:.4f} "
            # f"(vqa={val_metrics['loss_vqa']:.4f} kl={val_metrics['loss_kl']:.4f} mse={val_metrics['loss_mse']:.4f} acc={val_metrics['acc']:.4f}) | "
             f"(vqa={val_metrics['loss_vqa']:.4f}  acc={val_metrics['acc']:.4f}) | "
            # f"(vqa={val_metrics['loss_vqa']:.4f} acc={val_metrics['acc']:.4f} entropy_loss={val_metrics["loss_entropy"]:.4f}, entr={val_metrics['entropy']:.4f}) | "
            f"lr={lr_now:.2e} | {epoch_time:.1f}s | mem={mem_mb:.0f}MB"
        )

        csv_logger.log({
            "epoch":        epoch + 1,
            "train_loss":   f"{train_metrics['loss']:.6f}",
            "train_vqa":    f"{train_metrics['loss_vqa']:.6f}",
            # "train_aux":    f"{train_metrics['loss_aux']:.6f}",
            # 'train_gate':   f"{train_metrics['loss_gate']:.6f}",
            # "train_kl":     f"{train_metrics['loss_kl']:.6f}",
            # "train_mse":    f"{train_metrics['loss_mse']:.6f}",
            
            # "entorpy":  f"{train_metrics['entropy']:.6f}",
            # "entropy_loss": f"{train_metrics['loss_entropy']:.6f}",
            
            "val_loss":     f"{val_metrics['loss']:.6f}",
            "val_vqa":      f"{val_metrics['loss_vqa']:.6f}",
            # "val_aux":      f"{val_metrics['loss_aux']:.6f}",
            # 'val_gate':    f"{val_metrics['loss_gate']:.6f}",
            # "val_kl":       f"{val_metrics['loss_kl']:.6f}",
            # "val_mse":      f"{val_metrics['loss_mse']:.6f}",
            # "entropy":   f"{val_metrics['entropy']:.6f}",
            # "entr_loss": f"{val_metrics['loss_entropy']:.6f}",
            "lr":           f"{lr_now:.6e}",
            "epoch_time_s": f"{epoch_time:.1f}",
            "gpu_mem_mb":   f"{mem_mb:.0f}",
        })

        # ── save best checkpoint of vqa loss ───────────────────────
        if val_metrics["loss_vqa"] < best_loss:
            best_loss = val_metrics["loss_vqa"]
        #     save_checkpoint(
        #         path      = os.path.join(cfg.system.save_dir, "best.pt"),
        #         epoch     = epoch,
        #         model     = model,
        #         optimizer = optimizer,
        #         scheduler = current_scheduler,
        #         scaler    = scaler,
        #         best_loss = best_loss,
        #         best_acc  = best_acc,
        #         cfg       = cfg,
        #     )
            log.info(f"  ↳ New best loss: {best_loss:.6f} — saved best.pt")
            
        # ── save best checkpoint of acc ────────────────────────────
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(
                path      = os.path.join(cfg.system.save_dir, "best_acc.pt"),
                epoch     = epoch,
                model     = model,
                optimizer = optimizer,
                scheduler = scheduler[current_scheduler],
                scaler    = scaler,
                ema       =ema,
                best_loss = best_loss,
                best_acc  = best_acc,  
                cfg       = cfg,
            )
            
            log.info(f"  ↳ New best acc: {best_acc:.4f} — saved best_acc.pt")
        
        # ── periodic checkpoint ────────────────────────────────────
        # if (epoch + 1) % cfg.train.save_every == 0:
        #     save_checkpoint(
        #         path      = os.path.join(cfg.system.save_dir, f"epoch_{epoch+1:03d}_{best_loss:.5f}.pt"),
        #         epoch     = epoch,
        #         model     = model,
        #         optimizer = optimizer,
        #         scheduler = scheduler,
        #         scaler    = scaler,
        #         best_loss = best_loss,
        #         best_acc  = best_acc,
        #         cfg       = cfg,
        #     )
        
        # if early_stopping(val_metrics["acc"]):
        #     log.info(f"Early stopping triggered at epoch {epoch+1}")
        #     break

        # always save latest for resume
        save_checkpoint(
            path      = os.path.join(cfg.system.save_dir, "latest.pt"),
            epoch     = epoch,
            model     = model,
            optimizer = optimizer,
            scheduler = scheduler[current_scheduler],
            scaler    = scaler,
            ema       = ema,
            best_loss = best_loss,
            best_acc  = best_acc,
            cfg       = cfg,
        )

    log.info(f"\nTraining complete. Best val loss: {best_loss:.6f}")
    log.info(f"Checkpoints saved to: {cfg.system.save_dir}")
    log.info(f"Metrics CSV:          {os.path.join(cfg.system.save_dir, 'metrics.csv')}")

    return best_loss
   
   
if __name__ == "__main__":
    

    cfg = get_args()

    # save resolved config next to checkpoints
    os.makedirs(cfg.system.save_dir, exist_ok=True)
    cfg.save_json(os.path.join(cfg.system.save_dir, "config.json"))

    print(f"Backbone : {cfg.model.backbone} | raw={cfg.model.raw_dim} proj={cfg.model.proj_dim}")
    print(f"Config saved to {cfg.system.save_dir}/config.json")
    # from build_dataset import build_dataloader
    # dataset = build_dataloader(cfg, split='train')
    # for batch in dataset:
    #     print(batch["node_raw"].shape)
    #     print(batch["question_mask"].shape)
    #     print(batch["question_emb"].shape)
    #     break
      
    
    print(f"-================= Training Model----------------------------")
   
    train_ds = MSVDDatasetMotion(cfg, split="train")
    val_ds   = MSVDDatasetMotion(cfg, split="val")
    model   = STGraphTransformerNet(cfg)
    train(model, train_ds, val_ds, cfg)
    