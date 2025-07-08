# Quantization-Aware Training (QAT) for EdgePneumoNet
# ----------------------------------------------------
# Fine-tunes the best FP32 checkpoint and exports an int8 TorchScript model
# ready for deployment on Raspberry Pi / Jetson Nano. Compatible with legacy
# torchvision versions that use ReLU6.
#
# Usage:
#   python -m training.qat_train \
#          --ckpt checkpoints/best.pth \
#          --epochs 3 --batch 64 --lr 1e-5 --device cpu
# ----------------------------------------------------

import argparse, time, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.ao.quantization import (
    get_default_qat_qconfig, prepare_qat, convert, fuse_modules
)
from torchvision.ops.misc import Conv2dNormActivation
from models.mobilenetv2 import MobileNetV2Binary
from data.rsna import RSNAPneumonia
from training.utils import AverageMeter
from tqdm import tqdm
from pathlib import Path

# ---------------- Command Line Arguments ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',   required=True, help="Path to FP32 model checkpoint")
parser.add_argument('--epochs', type=int, default=3, help="Number of QAT fine-tuning epochs")
parser.add_argument('--batch',  type=int, default=64, help="Batch size")
parser.add_argument('--lr',     type=float, default=1e-5, help="Learning rate")
parser.add_argument('--device', type=str, default='cpu', help="Device: cpu or cuda")
parser.add_argument('--export', type=str, default='edgepneumonet_int8.pt', help="Output path for quantized model")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Replace ReLU6 with ReLU
print('Replacing ReLU6 with ReLU...')
def swap_relu6(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            swap_relu6(child)

# Fuse Conv-BN-ReLU blocks
print('Fusing Conv-BN-ReLU layers...')
def fuse_mobilenetv2(model):
    for block in model.backbone.features:
        if not hasattr(block, 'conv'):
            continue
        for layer in block.conv:
            if isinstance(layer, Conv2dNormActivation):
                fuse_modules(layer, ['0', '1', '2'], inplace=True)
            if isinstance(layer, nn.Sequential) and len(layer) >= 2:
                names = [str(i) for i, m in enumerate(layer)
                         if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
                if len(names) == 2:
                    fuse_modules(layer, names, inplace=True)

# Load and prepare FP32 model
print('Loading FP32 model...')
model_fp32 = MobileNetV2Binary(pretrained=False).to(device)
state = torch.load(args.ckpt, map_location=device, weights_only=False)
model_fp32.load_state_dict(state['state_dict'])
swap_relu6(model_fp32)  # Replace ReLU6 → ReLU

# Set model to eval mode for proper fusion
model_fp32.eval()
fuse_mobilenetv2(model_fp32)

# Set to train after fake quant is added
model_fp32.train()

# Apply QAT configuration
print('Preparing QAT (backend = qnnpack)...')
model_fp32.qconfig = get_default_qat_qconfig('qnnpack')
prepare_qat(model_fp32, inplace=True)
model_fp32.train()

# Prepare DataLoader
train_ds = RSNAPneumonia(split='train', use_png=True)
train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                      num_workers=0, pin_memory=True)

# Loss function with class imbalance correction
labels_tensor = torch.tensor(train_ds.df.Target.values)
pos_weight = (labels_tensor == 0).sum() / (labels_tensor == 1).sum()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device).float())
optimizer = torch.optim.AdamW(model_fp32.parameters(), lr=args.lr)

# QAT fine-tuning loop
for epoch in range(1, args.epochs + 1):
    meter, t0 = AverageMeter(), time.time()
    for x, y in tqdm(train_dl, desc=f'QAT {epoch}/{args.epochs}'):
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        loss = loss_fn(model_fp32(x), y)
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), len(x))
    print(f'Epoch {epoch}: loss = {meter.avg:.4f}, time = {time.time() - t0:.1f}s')

# Convert to INT8 and export
print('Converting to INT8...')
model_int8 = convert(model_fp32.eval().cpu(), inplace=False)
torch.jit.save(torch.jit.script(model_int8), args.export)
size_mb = round(Path(args.export).stat().st_size / 1e6, 2)
print(f'Quantized model saved to {args.export} (size ~ {size_mb} MB)')