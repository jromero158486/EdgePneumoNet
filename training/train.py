"""
MobileNetV2 training binary for RSNA Pneumonia Detection

This script implements:
1. Class imbalance handling via 'pos_weight' in BCEWithLogitsLoss
2. Optional custom Focal Loss as an alternative
3. Training loop with Early Stopping and TensorBoard logging
"""

import argparse, time, torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from models.mobilenetv2 import MobileNetV2Binary
from data.rsna import RSNAPneumonia
from training.utils import (
    AverageMeter, EarlyStopper, save_checkpoint, log_epoch
)
from eval.metrics import auc_score, f1_binary, expected_calibration_error
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ------------------ Focal Loss ------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', weight=self.weight
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)  # pt = p_t
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * BCE
        return loss.mean()

# ------------------ CLI Args ------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze", action="store_true", help="freeze convolutional layers")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints/")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--use_focal", action="store_true", help="use Focal Loss instead of BCE")
    return ap.parse_args()

# ------------------ Data Loaders ------------------
def make_loaders(batch_size=16):
    train_ds = RSNAPneumonia(split="train")
    val_ds   = RSNAPneumonia(split="val")

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    dl_val   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                          num_workers=0, pin_memory=True)
    return dl_train, dl_val

# ------------------ Evaluation ------------------
def evaluate(model, loader, loss_fn, device):
    model.eval()

    all_logits, all_labels = [], []
    loss_meter = AverageMeter()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)

            logits = model(x)
            loss = loss_fn(logits, y)
            loss_meter.update(loss.item(), len(x))

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    # Concatenate all predictions and compute metrics once
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    print("VAL distribution:", torch.bincount(labels.to(torch.int64)))

    try:
        auc = auc_score(logits, labels)
    except ValueError:
        auc = float("nan")  # fallback in case one class is missing

    f1  = f1_binary(logits, labels)
    ece = expected_calibration_error(logits, labels)

    return {
        "loss": loss_meter.avg,
        "auc":  auc,
        "f1":   f1,
        "ece":  ece
    }

# ------------------ Training Loop ------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    meters = {"loss": AverageMeter()}

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        meters["loss"].update(loss.item(), len(x))

    return {"loss": meters["loss"].avg}

# ------------------ Main Training Script ------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir="runs/edgepneumonet")

    # 1. Data
    dl_train, dl_val = make_loaders(args.batch)

    # 2. Model
    model = MobileNetV2Binary(
        pretrained=True,
        freeze_features=args.freeze,
        dropout=0.3
    ).to(device)

    # 3. Class imbalance correction (pos_weight)
    targets = torch.cat([y for _, y in dl_train])
    pos_weight = (targets == 0).sum() / (targets == 1).sum()
    pos_weight_tensor = torch.tensor(pos_weight).to(device)

    # 4. Loss Function
    if args.use_focal:
        print("Using Focal Loss (gamma=2.0)")
        loss_fn = FocalLoss(gamma=2.0)
    else:
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight:.2f}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    early_stop = EarlyStopper(patience=5, delta=0.001)

    # 6. Training loop
    ckpt_dir = Path(args.ckpt_dir)
    best_auc = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, dl_train, loss_fn, optimizer, device)
        val_metrics   = evaluate(model, dl_val, loss_fn, device)

        scheduler.step(val_metrics["auc"])
        log_epoch(writer, epoch, "train", train_metrics)
        log_epoch(writer, epoch, "val", val_metrics)

        elapsed = time.time() - t0
        print(f"[{epoch:02d}/{args.epochs}] train_loss={train_metrics['loss']:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}  val_f1={val_metrics['f1']:.4f}  "
              f"val_ece={val_metrics['ece']:.4f}  time={elapsed:.1f}s")

        is_best = val_metrics["auc"] > best_auc
        best_auc = max(best_auc, val_metrics["auc"])
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_auc": best_auc,
        }, ckpt_dir, is_best)

        if early_stop(val_metrics["auc"]):
            print("↳ Early stopping: no improvement in AUC")
            break

    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    main()