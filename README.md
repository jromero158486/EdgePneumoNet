# EdgePneumoNet 🫁📱  
**Uncertainty-Aware Pneumonia Detection on Edge Devices with Visual Explainability**

EdgePneumoNet is a lightweight, modular framework for detecting pneumonia in chest X-rays. It is optimized for real-time inference on resource-constrained edge devices (e.g., Raspberry Pi 4, Jetson Nano) and provides calibrated uncertainty estimates together with interpretable Grad-CAM heat-maps.

---

## Motivation
Pneumonia remains a leading cause of mortality in low-resource settings. Deep-learning systems can help, yet typical approaches are:

- Too computationally heavy for edge hardware.  
- Lacking reliable uncertainty quantification.  
- Hard to interpret for clinicians.

**EdgePneumoNet** overcomes these hurdles via:

- **MobileNetV2** backbone—small and fast.  
- **Quantization-Aware Training (QAT)** for 8-bit deployment.  
- **Monte-Carlo Dropout** to estimate predictive uncertainty.  
- **Grad-CAM** for visual explanations aligned with radiologist intuition.

---

## Key Features

| ✔️ | Feature |
|----|---------|
| Lightweight MobileNetV2 (≤ 5 MB quantized) |
| Real-time inference ≥ 10 FPS on Raspberry Pi 4 / Jetson Nano |
| Calibrated predictions (ECE ≤ 0.03) |
| Pixel-level uncertainty via MC-Dropout |
| Explainability with Grad-CAM |
| End-to-end training / eval pipeline |

---

## Project Structure

EdgePneumoNet/
├── config/ # YAML configuration files
├── data/ # Dataset loading & preprocessing (RSNA Pneumonia)
├── models/ # MobileNetV2 with dropout layers
├── training/ # Training scripts, loss functions, callbacks
├── eval/ # Inference, uncertainty, Grad-CAM
├── utils/ # Metrics, logging, calibration helpers
├── requirements.txt
└── README.md

## Dataset

We employ the **RSNA Pneumonia Detection Challenge** dataset.

```bash
python data/prepare_rsna.py --path /path/to/rsna

## Quick Start

# Clone repository
git clone https://github.com/your-username/EdgePneumoNet.git
cd EdgePneumoNet

# Install dependencies
pip install -r requirements.txt

# Train MobileNetV2
python training/train.py --config config/train_mobilenet.yaml

# Run MC-Dropout inference
python eval/mc_inference.py --model checkpoints/mobilenet_qat.pth

# Visualize Grad-CAM on a single image
python eval/visualize_gradcam.py --image /path/to/image.png

## Performance (RSNA Validation)

| Metric                           | Value             |
| -------------------------------- | ----------------- |
| AUC                              | **0.822 ± 0.004** |
| F1 Score                         | **0.78**          |
| Expected Calibration Error (ECE) | **0.0258**        |
| Inference Speed                  | **≥ 10 FPS**      |
| Model Size (int8 QAT)            | **≈ 4× smaller**  |

