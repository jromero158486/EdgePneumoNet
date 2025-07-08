import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

def auc_score(logits, y_true):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return roc_auc_score(y_true.cpu().numpy(), probs)

def f1_binary(logits, y_true, thresh=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs >= thresh).astype(int)
    return f1_score(y_true.cpu().numpy(), preds)

def expected_calibration_error(logits, y_true, n_bins=15):
    """
    Vectorised ECE for binary classification. Lower = better calibrated.
    """
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = y_true.cpu().numpy()
    bins = np.linspace(0, 1, n_bins + 1)
    inds = np.digitize(probs, bins, right=True) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if mask.sum() == 0:
            continue
        p_avg = probs[mask].mean()
        acc = y_true[mask].mean()
        ece += np.abs(p_avg - acc) * mask.mean()
    return ece