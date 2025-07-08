import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Binary(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2, freeze_features=False):
        super().__init__()

        # Base Model
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        #self.backbone = models.mobilenet_v2(pretrained=pretrained)

        # Freeze if it is required
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)  # output: logit
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)  

if __name__ == "__main__":
    model = MobileNetV2Binary(pretrained=True, freeze_features=True)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)  # ➝ torch.Size([4])