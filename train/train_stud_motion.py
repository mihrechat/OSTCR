
import sys
import os
# Adds the 'CausalNet' directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
# from data_class_stud import get_args
from data_class_multi_choice import get_args
from build_dataset import STUDTrafficDatasetMotion
# from models.visualEncoder.causal_STCR_stud_motion import STGraphTransformerNet
from VisualEncoder.stud_traffic import STGraphTransformerNet
from build_dataset import stgraph_collate
from model_loss_stud import build_losses, compute_total_loss

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def save_checkpoint(
    path:      str,
    epoch:     int,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    best_loss: float,
    best_acc:  float,
    cfg,       # ExperimentConfig
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the state dictionaries and the config object
    torch.save({
        "epoch":          epoch,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "scaler":         scaler.state_dict(),
        "best_loss":      best_loss,
        "best_acc":       best_acc,
        "config":         cfg,
    }, path)


def load_checkpoint(
    path:      str,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler:    GradScaler,
    device:    torch.device,
) -> tuple:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    
    return ckpt["epoch"], ckpt["best_loss"], ckpt["best_acc"]


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
) -> dict:

    model.train()
    optimizer.zero_grad(set_to_none=True)   # set_to_none saves memory vs zero_grad()

    total_loss = total_kl = total_mse = total_vqa = 0.0
    n_batches  = len(loader)
    global_step = epoch * n_batches
    total_correct = 0
    total_samples = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        # ── forward (mixed precision - updated modern syntax) ──────
        with torch.autocast(device_type=device.type, enabled=cfg.system.use_amp, dtype=torch.bfloat16):
            outputs = model(batch)
            causal_logits  = outputs["causal_logits"] 
           
            losses = compute_total_loss(
              cfg=cfg,
              batch=batch,
              outputs=outputs,
              loss_modules=loss_modules,
             )

            # scale loss for gradient accumulation
            loss = losses["loss"] / cfg.train.grad_accum_steps
      
        # ── Accuracy Calculation ──
        labels = batch.answers.view(-1) # FIXED: view(-1) is safer than squeeze(-1)
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

            # The scaler updates the optimizer. NO SCHEDULER STEP HERE!
            scaler.step(optimizer)
            scaler.update()
                
            optimizer.zero_grad(set_to_none=True)
        
        # ── accumulate metrics ────────────────────────────────────
        total_loss += losses["loss"].item()
        total_kl   += losses.get("loss_kl",  torch.tensor(0.0)).item()
        total_mse  += losses.get("loss_mse", torch.tensor(0.0)).item()
        total_vqa  += losses.get("loss_vqa", torch.tensor(0.0)).item()

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
    return {
        "loss":     total_loss / n,
        "loss_kl":  total_kl   / n,
        "loss_mse": total_mse  / n,
        "loss_vqa": total_vqa  / n,
        "acc": total_correct / total_samples if total_samples > 0 else 0.0,
    }


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
) -> dict:

    model.eval()
    total_loss = total_kl = total_mse = total_vqa = 0.0
    n_batches  = len(loader)
    total_correct = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.autocast(device_type=device.type, enabled=cfg.system.use_amp, dtype=torch.bfloat16):
            outputs    = model(batch)
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

    n = max(n_batches, 1)
    return {
        "loss":     total_loss / n,
        "loss_kl":  total_kl   / n,
        "loss_mse": total_mse  / n,
        "loss_vqa": total_vqa  / n,
        "acc": total_correct / total_samples if total_samples > 0 else 0.0,
    }


# ==================================================================
# GPU memory helper
# ==================================================================
def gpu_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device) / 1024 ** 2
    return 0.0

# ==================================================================
# Main training function
# ==================================================================
def train(
    model,
    train_dataset,
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
    
    n_val   = max(1, int(len(train_dataset) * cfg.data.val_split))
    n_train = len(train_dataset) - n_val
    train_ds, val_ds = random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.system.seed),
    )

    # ── train / val split ─────────────────────────────────────────
    n_train = len(train_ds)
    n_val   = len(val_ds)
    print(f"Train videos : {n_train} | Val videos : {n_val}")
    loss_modules = build_losses(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.train.batch_size,
        shuffle     = True,
        num_workers = cfg.data.num_workers,
        collate_fn  = stgraph_collate, 
        pin_memory  = device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.train.batch_size,
        shuffle     = False,
        num_workers = cfg.data.num_workers,
        collate_fn  = stgraph_collate,
        pin_memory  = device.type == "cuda",
    )

    # ── optimizer ──────────────��──────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.train.lr,
        weight_decay = cfg.train.weight_decay,
        betas        = (0.9, 0.999), 
    )

    # ── scheduler — ReduceLROnPlateau (FIXED) ─────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',          # We want validation accuracy to maximize
        factor=0.5,          # Cut LR in half when plateaued
        patience=cfg.train.patience,          # Wait 2 epochs before cutting
        min_lr=1e-6,         # Absolute bottom floor for learning rate
    )

    # ── AMP scaler ────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.system.use_amp)

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

    # ── resume from checkpoint ────────────────────────────────────
    if cfg.system.resume_from and os.path.exists(cfg.system.resume_from):
        start_epoch, best_loss, best_acc = load_checkpoint(
            cfg.system.resume_from, model, optimizer, scheduler, scaler, device
        )
        start_epoch += 1
        log.info(f"Resumed from {cfg.system.resume_from} — epoch {start_epoch}, best_loss={best_loss:.6f}, best_acc={best_acc:.4f}")
        # scheduler.patience = 5
    # ── training loop ─────────────────────────────────────────────
    log.info(f"Starting training — {cfg.train.epochs} epochs")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        # ── train ──────────────────────────────────────────────────
        train_metrics = train_one_epoch(
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
        )

        # ── validate ───────────────────────────────────────────────
        val_metrics = validate_one_epoch(
            model        = model,
            loader       = val_loader,
            cfg          = cfg,
            loss_modules = loss_modules,
            device       = device,
            log          = log,
        )

        # ── STEP THE SCHEDULER HERE ────────────────────────────
        # ReduceLROnPlateau must step using the validation metric!
        scheduler.step(val_metrics["acc"])

        epoch_time = time.time() - t0
        mem_mb     = gpu_mem_mb(device)
        lr_now     = optimizer.param_groups[0]["lr"]

        # ── logging ────────────────────────────────────────────────
        log.info(
            f"Epoch {epoch+1:03d}/{cfg.train.epochs} | "
            f"train={train_metrics['loss']:.4f} "
            f"(vqa={train_metrics['loss_vqa']:.4f} kl={train_metrics['loss_kl']:.4f} mse={train_metrics['loss_mse']:.4f} acc={train_metrics['acc']:.4f}) | "
            f"val={val_metrics['loss']:.4f} "
            f"(vqa={val_metrics['loss_vqa']:.4f} kl={val_metrics['loss_kl']:.4f} mse={val_metrics['loss_mse']:.4f} acc={val_metrics['acc']:.4f}) | "
            f"lr={lr_now:.2e} | {epoch_time:.1f}s | mem={mem_mb:.0f}MB"
        )

        csv_logger.log({
            "epoch":        epoch + 1,
            "train_loss":   f"{train_metrics['loss']:.6f}",
            "train_vqa":    f"{train_metrics['loss_vqa']:.6f}",
            "train_kl":     f"{train_metrics['loss_kl']:.6f}",
            "train_mse":    f"{train_metrics['loss_mse']:.6f}",
            "val_loss":     f"{val_metrics['loss']:.6f}",
            "val_vqa":      f"{val_metrics['loss_vqa']:.6f}",
            "val_kl":       f"{val_metrics['loss_kl']:.6f}",
            "val_mse":      f"{val_metrics['loss_mse']:.6f}",
            "lr":           f"{lr_now:.6e}",
            "epoch_time_s": f"{epoch_time:.1f}",
            "gpu_mem_mb":   f"{mem_mb:.0f}",
        })

        # ── save best checkpoint of vqa loss ───────────────────────
        if val_metrics["loss_vqa"] < best_loss:
            best_loss = val_metrics["loss_vqa"]
            save_checkpoint(
                path      = os.path.join(cfg.system.save_dir, "best.pt"),
                epoch     = epoch,
                model     = model,
                optimizer = optimizer,
                scheduler = scheduler,
                scaler    = scaler,
                best_loss = best_loss,
                best_acc  = best_acc,
                cfg       = cfg,
            )
            log.info(f"  ↳ New best loss: {best_loss:.6f} — saved best.pt")
            
        # ── save best checkpoint of acc ────────────────────────────
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(
                path      = os.path.join(cfg.system.save_dir, "best_acc.pt"),
                epoch     = epoch,
                model     = model,
                optimizer = optimizer,
                scheduler = scheduler,
                scaler    = scaler,
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

        # always save latest for resume
        save_checkpoint(
            path      = os.path.join(cfg.system.save_dir, "latest.pt"),
            epoch     = epoch,
            model     = model,
            optimizer = optimizer,
            scheduler = scheduler,
            scaler    = scaler,
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
    # dataset = build_dataloader(cfg)
    # for batch in dataset:
    #     print(batch["motion_feat"].shape)
    #     print(batch["answers"])
    #     break
    
    # print(f"-================= Training Model----------------------------")
   
    train_ds = STUDTrafficDatasetMotion(cfg, split="train")
    model   = STGraphTransformerNet(cfg)
    train(model, train_ds, cfg)
    
