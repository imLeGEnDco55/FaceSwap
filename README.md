# FaceSwap Standalone ⚡

Ultra-fast, lightweight, pure-Python face swapping application powered by **InsightFace**, **Hyperswap (256x256)**, **INSwapper (128x128)**, **GPEN (512x512)** face restoration, and **ONNX CUDA Execution Provider**.

Designed by **imLeGEnDco +FlowCode Department**.

---

## ✨ Features
- **Zero ComfyUI Dependencies:** Standalone Gradio app in pure Python.
- **A100 / L4 / T4 CUDA Acceleration:** Uses `CUDAExecutionProvider` for sub-second face swaps on NVIDIA GPUs.
- **Multiple Models:** Supports `hyperswap_1b_256.onnx`, `hyperswap_1a_256.onnx`, `hyperswap_1c_256.onnx`, and `inswapper_128.onnx`.
- **Optional Face Restoration:** High-resolution face sharpening with `GPEN-BFR-512.onnx`.
- **Lossless PNG Export:** Automatically exports clean `.png` files with proper extensions for easy downloading.

---

## 🚀 Quickstart on Google Colab

```bash
# 1. Clone repo
git clone https://github.com/imLeGEnDco55/FaceSwap
cd FaceSwap

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch App
python app.py
```
