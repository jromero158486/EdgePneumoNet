# eval/mc_dropout.py
import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from data.rsna import RSNAPneumonia
from models.mobilenetv2 import MobileNetV2Binary
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

# ------------------------ CLI ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--samples", type=int, default=30, help="MC samples")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")


model = MobileNetV2Binary(pretrained=False).to(device)
state = torch.load(args.ckpt, map_location=device, weights_only=False)
model.load_state_dict(state["state_dict"])
model.eval()
enable_dropout(model) 


val_ds = RSNAPneumonia(split="val")
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

# ------------------------ MC Inference ------------------------
all_preds = []

with torch.no_grad():
    for _ in tqdm(range(args.samples), desc="MC samples"):
        preds = []
        for x, _ in val_dl:
            x = x.to(device)
            out = torch.sigmoid(model(x)).squeeze()
            preds.append(out.cpu().numpy())
        all_preds.append(np.concatenate(preds))

all_preds = np.stack(all_preds)  # shape: [T, N]
mean_probs = all_preds.mean(axis=0)
std_probs  = all_preds.std(axis=0)  # epistemic uncertainty


os.makedirs("results/mc_dropout", exist_ok=True)
np.save("results/mc_dropout/probs_mean.npy", mean_probs)
np.save("results/mc_dropout/probs_std.npy", std_probs)
print("Save: probs_mean.npy y probs_std.npy")


plt.hist(std_probs, bins=40, color="skyblue", edgecolor="k")
plt.title("Distribución de incertidumbre (std) – MC Dropout")
plt.xlabel("Incertidumbre Epistemológica")
plt.ylabel("Número de muestras")
plt.grid(True)
plt.savefig("results/mc_dropout/uncertainty_histogram.png", dpi=200)
print("Histograma guardado en: results/mc_dropout/uncertainty_histogram.png")
