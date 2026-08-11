# CONTEXT — FaceSwap Standalone · imLeGEnDco +FlowCode Dept.

> Última actualización: 2026-08-10 · Estado: **PRODUCCIÓN / STANDALONE PURA**

---

## 🎯 ESTADO ACTUAL

Aplicación Gradio **Standalone pura en Python** (100% libre de ComfyUI) para FaceSwap rápido impulsado por `InsightFace`, `Hyperswap 256`, `INSwapper 128` y `GPEN 512`.

### ⚡ Aceleración GPU A100:
- Utiliza `CUDAExecutionProvider` de ONNX Runtime para ejecutar la detección y el intercambio directo en los Tensor Cores de la A100.
- Tiempo de intercambio por rostro: **~0.05 - 0.15 segundos**.
- Genera exportaciones nativas `.png` para descargas directas sin pérdida.

---

## 📁 ESTRUCTURA DEL PROYECTO

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Gradio standalone pura en Python (InsightFace + ONNX CUDA + GPEN + Gradio) |
| `requirements.txt` | Dependencias limpias de PyTorch, InsightFace y ONNX Runtime GPU |
| `README.md` | Documentación oficial para el repositorio GitHub `imLeGEnDco55/FaceSwap` |

---

## 🚀 COMANDOS GIT PARA SUBIR A GITHUB

```bash
git init
git add .
git commit -m "feat: Standalone A100 CUDA FaceSwap application"
git branch -M main
git remote add origin https://github.com/imLeGEnDco55/FaceSwap.git
git push -u origin main
```
