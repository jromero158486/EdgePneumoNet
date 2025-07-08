import math, time, torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from eval.metrics import auc_score, f1_binary, expected_calibration_error


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / (self.count + 1e-8)


class EarlyStopper:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = -math.inf
        self.counter = 0

    def __call__(self, metric):
        if metric > self.best + self.delta:
            self.best = metric
            self.counter = 0
            return False  # keep training
        self.counter += 1
        return self.counter >= self.patience   # True => stop


def save_checkpoint(state: dict, ckpt_dir: Path, is_best: bool):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / "last.pth")
    if is_best:
        torch.save(state, ckpt_dir / "best.pth")

def log_epoch(writer, epoch: int, tag: str, meters: dict):
    for k, v in meters.items():
        scalar = v.avg if hasattr(v, "avg") else v
        writer.add_scalar(f"{tag}/{k}", scalar, epoch)
