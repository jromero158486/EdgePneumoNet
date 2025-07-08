# EdgePneumoNet 🫁📱  
**Uncertainty-Aware Pneumonia Detection on Edge Devices with Visual Explainability**

EdgePneumoNet is a lightweight, modular framework for detecting pneumonia from chest X-rays using deep learning. It is optimized for deployment on resource-constrained edge devices (e.g., Raspberry Pi, Jetson Nano) with real-time performance, calibrated uncertainty estimates, and interpretable visual explanations.

---

## Motivation

Pneumonia is a leading cause of mortality, especially in low-resource settings. While deep learning can assist in diagnosis, real-world deployment is hindered by:
- Heavy models incompatible with edge devices.
- Lack of uncertainty quantification.
- Poor interpretability for clinicians.

EdgePneumoNet addresses these challenges by combining:
- Efficient CNNs (e.g., MobileNetV2, EfficientNet-Lite)
- Quantization-aware training (QAT)
- Monte Carlo Dropout for uncertainty estimation
- Grad-CAM for visual explanations

---

## Features

✅ Lightweight models for real-time inference  
✅ Uncertainty-aware predictions (ECE ≤ 0.03)  
✅ Visual interpretability with Grad-CAM  
✅ Quantized inference with ONNX/QAT  
✅ Deployable on Raspberry Pi 4 and Jetson Nano  
✅ Modular training/evaluation pipeline  

---

## 🗂 Project Structure
EdgePneumoNet/
├── config/ # YAML configuration files
├── data/ # Dataset loading & preprocessing (RSNA Pneumonia)
├── models/ # Lightweight CNNs & dropout layers
├── training/ # Training, loss functions, callbacks
├── eval/ # Inference, uncertainty, Grad-CAM
├── experiments/ # Saved weights, logs, metrics
├── utils/ # Helpers (metrics, logging, calibration)
├── requirements.txt
└── README.md

## Dataset

We use the **RSNA Pneumonia Detection Challenge** dataset.  
To download and prepare the data:

```bash
python data/prepare_rsna.py --path /path/to/dataset

## Clone repository

git clone https://github.com/your-username/EdgePneumoNet.git
cd EdgePneumoNet

pip install -r requirements.txt

python training/train.py --config config/train_mobilenet.yaml

python eval/mc_inference.py --model checkpoints/mobilenet_qat.pth

python eval/visualize_gradcam.py --image /path/to/image.png

## Performance

| Metric                           | Value                 |
| -------------------------------- | --------------------- |
| AUC (RSNA)                       | 0.822 ± 0.004         |
| F1 Score                         | 0.78                  |
| Expected Calibration Error (ECE) | 0.0258                |
| Inference Speed                  | ≥ 10 FPS (Jetson/RPi) |
| Model Size (QAT)                 | \~4× smaller          |

