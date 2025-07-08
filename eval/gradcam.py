# eval/gradcam.py
import argparse, torch, numpy as np, cv2, os
import matplotlib.pyplot as plt
from torchvision import transforms
from models.mobilenetv2 import MobileNetV2Binary
from data.rsna import RSNAPneumonia
from torch.utils.data import DataLoader

# ----------------------------- GradCAM -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred = torch.sigmoid(output).squeeze()
        pred.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        return heatmap, pred.item()

# ----------------------------- Main Script -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")


model = MobileNetV2Binary(pretrained=False).to(device)
state = torch.load(args.ckpt, map_location=device, weights_only=False)
model.load_state_dict(state["state_dict"])
model.eval()


dataset = RSNAPneumonia(split="val")
x, y = dataset[args.index]
input_tensor = x.unsqueeze(0).to(device)


target_layer = model.backbone.features[-2]
gradcam = GradCAM(model, target_layer)


heatmap, prob = gradcam.generate(input_tensor)
img = np.transpose(x.numpy(), (1, 2, 0))      # [H,W,C]
img = (img - img.min()) / (img.max() - img.min() + 1e-7)


heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay = 0.4 * heatmap / 255 + 0.6 * img


os.makedirs("results/gradcam", exist_ok=True)
plt.imsave(f"results/gradcam/sample_{args.index}_prob{prob:.2f}.png", overlay)
print(f"Grad-CAM save as: results/gradcam/sample_{args.index}_prob{prob:.2f}.png")