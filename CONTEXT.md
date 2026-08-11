# CONTEXT — FaceSwap Standalone (INSwapper 128 + GPEN 512) · imLeGEnDco +FlowCode Dept.

> Última actualización: 2026-08-10 · Estado: **PRODUCCIÓN / SIMPLIFICADA AL MÁXIMO**

---

## 🎯 ESTADO ACTUAL

Aplicación Gradio ultra-simplificada para intercambio de rostros usando **`inswapper_128.onnx`** (con paste-back nativo de InsightFace) y **`GPEN-BFR-512.onnx`** para la restauración de nitidez facial en GPU A100 / L4 / T4.

### ⚙️ Interfaz y Parámetros:
- **`Source (Face)`**: Imagen con el rostro a transplantar.
- **`Target (Body)`**: Imagen objetivo.
- **`Target Face Index`**: Índice del rostro a reemplazar (0 = rostro más grande).
- **`Face Restore Strength (GPEN 512)`**: Slider de restauración facial (0.0 a 1.0, por defecto 0.7).

### ⚡ Solución para Colab CUDA 12.x:
- `requirements.txt` fijado con **`onnxruntime-gpu==1.19.2`** para resolver la compatibilidad de librerías dinámicas CUDA 12 (`libcublasLt.so`).
- Paste-back nativo de InsightFace para posicionamiento exacto sin distorsión de coordenadas.

---

## 📁 ARCHIVOS

| Archivo | Descripción |
|---|---|
| `app.py` | App Gradio minimalista (INSwapper 128 + GPEN 512 + CUDA ONNX) |
| `requirements.txt` | Dependencias limpias con `onnxruntime-gpu==1.19.2` |
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
