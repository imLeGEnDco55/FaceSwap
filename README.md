# FaceSwap ⚡ (INSwapper 128 + GPEN 512)

Ultra-fast, lightweight, pure-Python face swapping application powered by **InsightFace (INSwapper 128)**, **GPEN (512x512)** face restoration, and **ONNX CUDA Execution Provider**.

Designed by **imLeGEnDco +FlowCode Department**.

---

## ✨ Features
- **Zero ComfyUI Dependencies:** Minimalist Gradio app in pure Python.
- **A100 / L4 / T4 CUDA Acceleration:** Powered by `CUDAExecutionProvider` for sub-second face swaps.
- **Native Paste-Back:** Exact face placement powered by InsightFace native alignment.
- **High-Resolution Face Restoration:** Sharpen faces up to 512x512 with `GPEN-BFR-512.onnx`.
- **Lossless PNG Export:** Automatically exports `.png` files for easy downloading.

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
