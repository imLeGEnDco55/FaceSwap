"""
FaceSwap Standalone (A100 GPU Accelerated) — imLeGEnDco +FlowCode Dept.

Pure Python + InsightFace + ONNX Runtime (CUDAExecutionProvider) implementation
without any ComfyUI dependencies.

GitHub: https://github.com/imLeGEnDco55/FaceSwap
"""

import os
import sys
import glob
import math
import ctypes
import tempfile
import types
from typing import Optional, List, Tuple
import cv2
import numpy as np

# ── Pre-load CUDA & cuDNN libraries for ONNX Runtime ─────────────────────────
try:
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    site_dir = os.path.dirname(torch_dir)
    nvidia_dir = os.path.join(site_dir, "nvidia")
    if os.path.exists(nvidia_dir):
        for lib_path in glob.glob(os.path.join(nvidia_dir, "*", "lib", "*.so*")):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
except Exception as exc:
    print(f"[faceswap] CUDA library preload notice: {exc}")

import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

# ── Transparent `spaces` mock for Colab ──────────────────────────────────────
try:
    import spaces
except Exception:
    spaces = types.ModuleType("spaces")

    class _GPUNoop:
        def __init__(self, fn=None, **kw):
            self._fn = fn

        def __call__(self, *args, **kwargs):
            if self._fn is not None:
                return self._fn(*args, **kwargs)
            fn = args[0] if args else kwargs.get("fn")
            return fn if callable(fn) else (lambda f: f)

    spaces.GPU = _GPUNoop
    sys.modules["spaces"] = spaces

import gradio as gr
import insightface
from insightface.app import FaceAnalysis

# ── Hardware & Execution Provider Setup ──────────────────────────────────────
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
print(f"[faceswap] GPU CUDA Status: {'⚡ CUDA ACTIVE (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else '⚠️ CPU MODE'}", flush=True)

# ── Download Models & Store Exact File Paths ─────────────────────────────────
MODEL_PATHS = {}

print("[faceswap] Downloading required ONNX models ...", flush=True)
MODEL_PATHS["inswapper_128.onnx"] = hf_hub_download(
    repo_id="ezioruan/inswapper_128.onnx", filename="inswapper_128.onnx", local_dir="models/swapper"
)
MODEL_PATHS["GPEN-BFR-512.onnx"] = hf_hub_download(
    repo_id="martintomov/comfy", filename="facerestore_models/GPEN-BFR-512.onnx", local_dir="models/restorer"
)

MODEL_PATHS["hyperswap_1a_256.onnx"] = hf_hub_download(
    repo_id="facefusion/models-3.3.0", filename="hyperswap_1a_256.onnx", local_dir="models/swapper"
)
MODEL_PATHS["hyperswap_1b_256.onnx"] = hf_hub_download(
    repo_id="facefusion/models-3.3.0", filename="hyperswap_1b_256.onnx", local_dir="models/swapper"
)
MODEL_PATHS["hyperswap_1c_256.onnx"] = hf_hub_download(
    repo_id="facefusion/models-3.3.0", filename="hyperswap_1c_256.onnx", local_dir="models/swapper"
)

hf_hub_download(repo_id="MonsterMMORPG/tools", filename="1k3d68.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="MonsterMMORPG/tools", filename="2d106det.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="maze/faceX", filename="det_10g.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="typhoon01/aux_models", filename="genderage.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="maze/faceX", filename="w600k_r50.onnx", local_dir="models/insightface/models/buffalo_l")

# ── Model Cache ───────────────────────────────────────────────────────────────
_face_app = None
_sessions = {}

def get_face_app():
    global _face_app
    if _face_app is None:
        print("[faceswap] Loading InsightFace Buffalo_L on CUDA ...", flush=True)
        app = FaceAnalysis(name="buffalo_l", root="models/insightface", providers=PROVIDERS)
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
    return _face_app

def get_onnx_session(model_path: str):
    global _sessions
    if model_path not in _sessions:
        print(f"[faceswap] Loading ONNX Session: {model_path} ...", flush=True)
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _sessions[model_path] = ort.InferenceSession(model_path, providers=PROVIDERS, session_options=opts)
    return _sessions[model_path]

# ── Helper Functions for Alignment & Blending ───────────────────────────────
STD_LANDMARKS_256 = np.array([
    [84.87, 105.94],
    [171.13, 105.94],
    [128.00, 146.66],
    [96.95, 188.64],
    [159.05, 188.64]
], dtype=np.float32)

def create_oval_mask(crop_size=256):
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    center = (crop_size // 2, crop_size // 2)
    axes = (int(crop_size * 0.35), int(crop_size * 0.4))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return np.clip(mask, 0, 1)

def paste_back_hyperswap(target_img, swapped_face_256, M):
    h, w = target_img.shape[:2]
    IM = cv2.invertAffineTransform(M)

    mask_256 = create_oval_mask(256)
    mask_3c = np.stack([mask_256] * 3, axis=2)

    swapped_norm = swapped_face_256.astype(np.float32) / 255.0

    warped_face = cv2.warpAffine(
        swapped_norm, IM, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.5
    )
    warped_mask = cv2.warpAffine(
        mask_3c, IM, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )

    warped_face = np.nan_to_num(np.clip(warped_face, 0, 1), nan=0.5)
    warped_mask = np.nan_to_num(np.clip(warped_mask, 0, 1), nan=0.0)
    warped_mask = cv2.GaussianBlur(warped_mask, (3, 3), 0)

    target_float = target_img.astype(np.float32) / 255.0
    result_float = target_float * (1.0 - warped_mask) + warped_face * warped_mask
    return (result_float * 255.0).clip(0, 255).astype(np.uint8)

def restore_gpen(face_img_512, strength=0.7):
    if strength <= 0:
        return face_img_512
    gpen_path = MODEL_PATHS.get("GPEN-BFR-512.onnx")
    session = get_onnx_session(gpen_path)
    img = cv2.resize(face_img_512, (512, 512))
    img_t = img[:, :, ::-1].astype(np.float32) / 255.0
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_t})[0][0]
    output = (output.transpose(1, 2, 0)[:, :, ::-1] + 1.0) / 2.0 * 255.0
    restored = np.clip(output, 0, 255).astype(np.uint8)

    if strength < 1.0:
        restored = cv2.addWeighted(restored, strength, face_img_512, 1.0 - strength, 0)
    return restored

# ── Main Face Swap Logic ──────────────────────────────────────────────────────
@spaces.GPU()
def process_faceswap(
    source_path: str,
    target_path: str,
    target_index: int = 0,
    swap_model_name: str = "hyperswap_1b_256.onnx",
    face_restore_model: str = "none",
    restore_strength: float = 0.7
):
    if not source_path or not target_path:
        raise gr.Error("Please upload both Source (Face) and Target (Body) images.")

    app = get_face_app()

    source_bgr = cv2.imread(source_path)
    target_bgr = cv2.imread(target_path)

    if source_bgr is None or target_bgr is None:
        raise gr.Error("Error loading input image files.")

    source_faces = app.get(source_bgr)
    if not source_faces:
        raise gr.Error("No face detected in the Source image.")

    source_faces = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    source_face = source_faces[0]

    target_faces = app.get(target_bgr)
    if not target_faces:
        raise gr.Error("No faces detected in the Target image.")

    target_faces = sorted(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    
    if target_index >= len(target_faces):
        target_index = 0
    target_face = target_faces[target_index]

    model_path = MODEL_PATHS.get(swap_model_name)
    if not model_path or not os.path.exists(model_path):
        raise gr.Error(f"Model file {swap_model_name} not found.")

    session = get_onnx_session(model_path)

    if "hyperswap" in swap_model_name:
        source_emb = source_face.normed_embedding.reshape(1, -1).astype(np.float32)
        landmarks = target_face.kps.astype(np.float32)

        M, _ = cv2.estimateAffinePartial2D(landmarks, STD_LANDMARKS_256)
        crop = cv2.warpAffine(target_bgr, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        crop_input = crop[:, :, ::-1].astype(np.float32) / 255.0
        crop_input = (crop_input - 0.5) / 0.5
        crop_input = crop_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        output = session.run(None, {'source': source_emb, 'target': crop_input})[0][0]
        output = np.nan_to_num(output, nan=0.0, posinf=255.0, neginf=0.0)
        if output.min() < 0.0 or output.max() <= 1.5:
            output = ((output + 1.0) / 2.0 * 255.0)
        
        swapped_256 = output.transpose(1, 2, 0)[:, :, ::-1].clip(0, 255).astype(np.uint8)

        if face_restore_model == "GPEN-BFR-512.onnx":
            swapped_512 = cv2.resize(swapped_256, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            restored_512 = restore_gpen(swapped_512, strength=restore_strength)
            swapped_256 = cv2.resize(restored_512, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        result_bgr = paste_back_hyperswap(target_bgr, swapped_256, M)

    else:
        swapper = insightface.model_zoo.get_model(model_path, providers=PROVIDERS)
        result_bgr = swapper.get(target_bgr, target_face, source_face, paste_back=True)
        if face_restore_model == "GPEN-BFR-512.onnx":
            landmarks = target_face.kps.astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(landmarks, STD_LANDMARKS_256)
            crop = cv2.warpAffine(result_bgr, M, (512, 512), flags=cv2.INTER_CUBIC)
            restored_512 = restore_gpen(crop, strength=restore_strength)
            restored_256 = cv2.resize(restored_512, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            result_bgr = paste_back_hyperswap(result_bgr, restored_256, M)

    # Save to explicit PNG file for clean downloads
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(result_rgb)
    tmp = tempfile.NamedTemporaryFile(suffix="_faceswap.png", delete=False)
    pil_out.save(tmp.name, format="PNG", compress_level=1)
    return tmp.name

# ── Gradio Blocks Interface ──────────────────────────────────────────────────
title_html = """
<div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 10px;">
    <h2 style="color: #6366f1; font-weight: 700; margin-bottom: 4px;">⚡ FaceSwap Standalone (A100 GPU Accelerated)</h2>
    <p style="color: #666; font-size: 0.95rem; margin: 0;">Ultra-fast pure Python face swapping powered by InsightFace & ONNX CUDA Execution Provider.</p>
</div>
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as app:
    gr.HTML(title_html)
    
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Group():
                    source_image = gr.Image(label="Source (Face)", type="filepath")
                    swap_model = gr.Dropdown(
                        choices=["hyperswap_1b_256.onnx", "hyperswap_1a_256.onnx", "hyperswap_1c_256.onnx", "inswapper_128.onnx"],
                        value="hyperswap_1b_256.onnx",
                        label="Swap Model"
                    )
                    face_restore_model = gr.Dropdown(
                        choices=["none", "GPEN-BFR-512.onnx"], 
                        value="none", 
                        label="Face Restore Model"
                    )
                    restore_strength = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.05, 
                        value=0.7, 
                        label="Face Restore Strength"
                    )

                with gr.Group():
                    target_image = gr.Image(label="Target (Body)", type="filepath")
                    target_index = gr.Dropdown(
                        choices=[0, 1, 2, 3, 4], 
                        value=0, 
                        label="Target Face Index"
                    )
                    gr.Markdown("Index 0 = Largest Face. Choose 1, 2, 3, etc. for other target faces.")
                    generate_btn = gr.Button("⚡ Swap Face!", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="Swapped Result", format="png")

    generate_btn.click(
        fn=process_faceswap,
        inputs=[source_image, target_image, target_index, swap_model, face_restore_model, restore_strength],
        outputs=[output_image]
    )

if __name__ == "__main__":
    app.launch(share=True)
