# FaceSwap ⚡ (Image & Video · INSwapper 128 + GPEN 512)

Ultra-fast, lightweight, pure-Python face swapping application for **Images and Videos**, powered by **InsightFace (INSwapper 128)**, **GPEN (512x512)** face restoration, and **ONNX CUDA Execution Provider**.

Designed by **imLeGEnDco +FlowCode Department**.

---

## ✨ Features
- **Zero ComfyUI Dependencies:** Standalone Gradio app in pure Python.
- **Image & Video Support:** Swap faces on static photos or full video clips.
- **Audio Preservation:** Video output retains the original audio track seamlessly.
- **A100 / L4 / T4 CUDA Acceleration:** Sub-second image swaps and ~10ms per frame video processing.
- **Native Paste-Back:** Pixel-perfect face placement powered by InsightFace.
- **Lossless PNG Export:** Images download natively with `.png` extension.

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
