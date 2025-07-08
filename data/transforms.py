import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224  # Lado que espera MobileNetV2

# ---------- ENTRENAMIENTO ----------
train_tfms = A.Compose([
    A.RandomResizedCrop(                       # ① → size
        size=(IMG_SIZE, IMG_SIZE),
        scale=(0.7, 1.0),
        ratio=(0.9, 1.1),
        p=1.0
    ),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])

# ---------- VALIDACIÓN / TEST ----------
val_tfms = A.Compose([
    A.Resize(                                  # ② → height / width
        height=IMG_SIZE,
        width=IMG_SIZE,
        interpolation=1,   # (opcional) LINEAR
        p=1.0
    ),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])