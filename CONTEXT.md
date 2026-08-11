# CONTEXT — FaceSwap Standalone (Imagen & Video) · imLeGEnDco +FlowCode Dept.

> Última actualización: 2026-08-10 · Estado: **PRODUCCIÓN / IMAGEN & VIDEO READY**

---

## 🎯 ESTADO ACTUAL

Aplicación Gradio **Standalone pura en Python** para intercambio de rostros tanto en **Imágenes** como en **Videos** usando `INSwapper 128` (con paste-back nativo) y `GPEN-BFR-512` en GPU CUDA.

### ⚡ Características Principales:
1. **Pestaña 📷 Image FaceSwap:** Intercambio ultra-rápido de rostro en imágenes en **~1 segundo** con exportación PNG sin pérdida.
2. **Pestaña 🎬 Video FaceSwap:** Intercambio fotograma a fotograma en GPU A100 (~10-15 ms por frame), con preservación automática de la pista de audio original mediante FFmpeg.

---

## 📁 ESTRUCTURA Y ARCHIVOS

| Archivo | Descripción |
|---|---|
| `app.py` | App Gradio standalone completa con soporte Dual: Imagen y Video |
| `requirements.txt` | Dependencias limpias (`torch`, `insightface`, `onnxruntime-gpu==1.19.2`, `imageio`, `imageio-ffmpeg`) |
| `README.md` | Documentación oficial del repositorio |

---

## 🚀 INSTRUCCIONES PARA COLAB

```bash
# 1. Clonar
!git clone https://github.com/imLeGEnDco55/FaceSwap
%cd FaceSwap

# 2. Instalar dependencias GPU
!pip install -r requirements.txt

# 3. Lanzar
!python app.py
```
