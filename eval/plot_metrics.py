import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import calibration_curve

from data.rsna import RSNAPneumonia
from torch.utils.data import DataLoader
from models.mobilenetv2 import MobileNetV2Binary

# ----------------------------- CLI -----------------------------
parser = argparse.ArgumentParser(description="EdgePneumoNet validation metrics")
parser.add_argument("--batch",  type=int, default=32,  help="Batch size")
parser.add_argument("--ckpt",   type=str, required=True, help="Path to checkpoint (.pth)")
parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# ----------------------- Dataset & Model -----------------------
val_ds = RSNAPneumonia(split="val")
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

model = MobileNetV2Binary(pretrained=False).to(device)
state = torch.load(args.ckpt, map_location=device, weights_only=False)
model.load_state_dict(state["state_dict"])
model.eval()

# ------------------------ Inference ----------------------------
logits, labels = [], []
with torch.no_grad():
    for x, y in val_dl:
        x = x.to(device)
        out = model(x)
        logits.append(out.cpu())
        labels.append(y.cpu())

logits  = torch.cat(logits).squeeze()
labels  = torch.cat(labels)
probs   = torch.sigmoid(logits).numpy()
y_true  = labels.numpy()

# --------------------------- ROC -------------------------------
fpr, tpr, thresholds = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                estimator_name=f"AUC = {roc_auc:.3f}").plot()
plt.title("ROC Curve – Validation")
plt.savefig("results/roc_curve.png", dpi=200)
plt.close()

# ----------------- Calibration (Reliability) -------------------
prob_true, prob_pred = calibration_curve(
    y_true, probs, n_bins=15, strategy="uniform"
)

# Trim to equal length (sklearn can return 1–2 fewer preds than bins)
min_len = min(len(prob_true), len(prob_pred))
prob_true, prob_pred = prob_true[:min_len], prob_pred[:min_len]

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray',
         label='Perfect calibration')
plt.xlabel('Mean Confidence')
plt.ylabel('Empirical Positive Rate')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True)
plt.savefig("results/calibration_curve.png", dpi=200)
plt.close()

# -------------------- Confusion Matrix -------------------------
j_index        = tpr - fpr
optimal_thr    = thresholds[np.argmax(j_index)]
y_pred         = (probs >= optimal_thr).astype(int)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    cm, display_labels=["Non-Pneumonia", "Pneumonia"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (thr = {optimal_thr:.2f})")
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.close()

print("Plots saved to the 'results/' folder.")