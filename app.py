"""
FaceSwap Standalone (Image & Video · A100 GPU Accelerated) — imLeGEnDco +FlowCode Dept.

Ultra-fast, pure Python face swapping app for Images and Videos powered by InsightFace,
INSwapper 128 (built-in paste back) and GPEN-BFR 512 face restoration.

GitHub: https://github.com/imLeGEnDco55/FaceSwap
"""

import os
import sys
import glob
import ctypes
import subprocess
import tempfile
import types
import cv2
import numpy as np
import imageio
import imageio_ffmpeg

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

# ── Download Models ───────────────────────────────────────────────────────────
print("[faceswap] Downloading inswapper_128.onnx & GPEN-BFR-512.onnx ...", flush=True)
INSWAPPER_PATH = hf_hub_download(
    repo_id="ezioruan/inswapper_128.onnx", filename="inswapper_128.onnx", local_dir="models/swapper"
)
GPEN_PATH = hf_hub_download(
    repo_id="martintomov/comfy", filename="facerestore_models/GPEN-BFR-512.onnx", local_dir="models/restorer"
)

hf_hub_download(repo_id="MonsterMMORPG/tools", filename="1k3d68.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="MonsterMMORPG/tools", filename="2d106det.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="maze/faceX", filename="det_10g.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="typhoon01/aux_models", filename="genderage.onnx", local_dir="models/insightface/models/buffalo_l")
hf_hub_download(repo_id="maze/faceX", filename="w600k_r50.onnx", local_dir="models/insightface/models/buffalo_l")

# ── Model Cache ───────────────────────────────────────────────────────────────
_face_app = None
_swapper = None
_gpen_session = None

def get_face_app():
    global _face_app
    if _face_app is None:
        print("[faceswap] Loading InsightFace Buffalo_L ...", flush=True)
        app = FaceAnalysis(name="buffalo_l", root="models/insightface", providers=PROVIDERS)
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
    return _face_app

def get_swapper():
    global _swapper
    if _swapper is None:
        print("[faceswap] Loading INSwapper 128 ...", flush=True)
        _swapper = insightface.model_zoo.get_model(INSWAPPER_PATH, providers=PROVIDERS)
    return _swapper

def get_gpen_session():
    global _gpen_session
    if _gpen_session is None:
        print("[faceswap] Loading GPEN-BFR-512 ONNX ...", flush=True)
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _gpen_session = ort.InferenceSession(GPEN_PATH, providers=PROVIDERS, session_options=opts)
    return _gpen_session

# ── Face Restoration & Blending Helpers ─────────────────────────────────────
STD_LANDMARKS_512 = np.array([
    [169.74, 211.88],
    [342.26, 211.88],
    [256.00, 293.32],
    [193.90, 377.28],
    [318.10, 377.28]
], dtype=np.float32)

def create_oval_mask(size=512):
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    axes = (int(size * 0.35), int(size * 0.4))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return np.clip(mask, 0, 1)

def apply_gpen_restore(target_img, target_face, strength=0.7):
    if strength <= 0:
        return target_img

    landmarks = target_face.kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(landmarks, STD_LANDMARKS_512)
    crop_512 = cv2.warpAffine(target_img, M, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    session = get_gpen_session()
    img_t = crop_512[:, :, ::-1].astype(np.float32) / 255.0
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_t})[0][0]
    output = (output.transpose(1, 2, 0)[:, :, ::-1] + 1.0) / 2.0 * 255.0
    restored_512 = np.clip(output, 0, 255).astype(np.uint8)

    if strength < 1.0:
        restored_512 = cv2.addWeighted(restored_512, strength, crop_512, 1.0 - strength, 0)

    # Paste restored 512 face back onto full resolution image
    h, w = target_img.shape[:2]
    IM = cv2.invertAffineTransform(M)
    mask_512 = create_oval_mask(512)
    mask_3c = np.stack([mask_512] * 3, axis=2)

    restored_norm = restored_512.astype(np.float32) / 255.0

    warped_face = cv2.warpAffine(restored_norm, IM, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0.5)
    warped_mask = cv2.warpAffine(mask_3c, IM, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    warped_face = np.nan_to_num(np.clip(warped_face, 0, 1), nan=0.5)
    warped_mask = np.nan_to_num(np.clip(warped_mask, 0, 1), nan=0.0)
    warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 0)

    target_float = target_img.astype(np.float32) / 255.0
    result_float = target_float * (1.0 - warped_mask) + warped_face * warped_mask
    return (result_float * 255.0).clip(0, 255).astype(np.uint8)

def _mux_audio(silent_video_path, source_video_path):
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", silent_video_path,
        "-i", source_video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return silent_video_path
    return output_path

# ── Image Face Swap ──────────────────────────────────────────────────────────
@spaces.GPU()
def process_image_faceswap(
    source_path: str,
    target_path: str,
    target_index: int = 0,
    restore_strength: float = 0.7
):
    if not source_path or not target_path:
        raise gr.Error("Upload both Source (Face) and Target (Body) images.")

    app = get_face_app()
    swapper = get_swapper()

    source_bgr = cv2.imread(source_path)
    target_bgr = cv2.imread(target_path)

    if source_bgr is None or target_bgr is None:
        raise gr.Error("Error reading input images.")

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

    swapped_bgr = swapper.get(target_bgr, target_face, source_face, paste_back=True)

    if restore_strength > 0:
        swapped_bgr = apply_gpen_restore(swapped_bgr, target_face, strength=restore_strength)

    result_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)
    pil_out = Image.fromarray(result_rgb)
    tmp = tempfile.NamedTemporaryFile(suffix="_faceswap.png", delete=False)
    pil_out.save(tmp.name, format="PNG", compress_level=1)
    return tmp.name

# ── Video Face Swap ──────────────────────────────────────────────────────────
@spaces.GPU()
def process_video_faceswap(
    source_path: str,
    video_path: str,
    target_index: int = 0,
    restore_strength: float = 0.7,
    progress=gr.Progress(track_tqdm=True)
):
    if not source_path or not video_path:
        raise gr.Error("Upload both Source (Face) image and Target Video.")

    app = get_face_app()
    swapper = get_swapper()

    source_bgr = cv2.imread(source_path)
    if source_bgr is None:
        raise gr.Error("Error reading Source face image.")

    source_faces = app.get(source_bgr)
    if not source_faces:
        raise gr.Error("No face detected in the Source image.")

    source_faces = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    source_face = source_faces[0]

    print("[faceswap] Reading video and prefetching frames into System RAM ...", flush=True)
    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 24) or 24

    # Prefetch all frames to System RAM for maximum speed
    raw_frames = [frame for frame in reader]
    reader.close()
    total_frames = len(raw_frames)

    silent_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(silent_path, fps=fps, codec="libx264", quality=8, macro_block_size=None)

    print(f"[faceswap] Processing {total_frames} video frames on CUDA ...", flush=True)
    try:
        for i, frame in enumerate(progress.tqdm(raw_frames, desc="Swapping video faces")):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            target_faces = app.get(frame_bgr)

            if target_faces:
                target_faces = sorted(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                idx = target_index if target_index < len(target_faces) else 0
                target_face = target_faces[idx]

                swapped_bgr = swapper.get(frame_bgr, target_face, source_face, paste_back=True)
                if restore_strength > 0:
                    swapped_bgr = apply_gpen_restore(swapped_bgr, target_face, strength=restore_strength)
                
                frame_out = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_out = frame

            writer.append_data(frame_out)
    finally:
        writer.close()

    return _mux_audio(silent_path, video_path)

# ── Gradio UI ────────────────────────────────────────────────────────────────
title_html = """
<div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 10px;">
    <h2 style="color: #6366f1; font-weight: 700; margin-bottom: 4px;">⚡ FaceSwap (Image & Video · A100 GPU Accelerated)</h2>
    <p style="color: #666; font-size: 0.95rem; margin: 0;">Fast & direct face swapping powered by InsightFace & GPEN 512</p>
</div>
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as app:
    gr.HTML(title_html)
    
    with gr.Tabs():
        with gr.Tab("📷 Image FaceSwap"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Group():
                            source_img = gr.Image(label="Source (Face)", type="filepath")
                            restore_str_img = gr.Slider(
                                minimum=0.0, 
                                maximum=1.0, 
                                step=0.05, 
                                value=0.7, 
                                label="Face Restore Strength (GPEN 512)"
                            )

                        with gr.Group():
                            target_img = gr.Image(label="Target (Body)", type="filepath")
                            target_idx_img = gr.Dropdown(
                                choices=[0, 1, 2, 3, 4], 
                                value=0, 
                                label="Target Face Index"
                            )
                            gr.Markdown("Index 0 = Largest Face. Choose 1, 2, 3, etc. for other target faces.")
                            btn_img = gr.Button("⚡ Swap Image Face!", variant="primary", size="lg")

                with gr.Column():
                    output_img = gr.Image(label="Swapped Result", format="png")

            btn_img.click(
                fn=process_image_faceswap,
                inputs=[source_img, target_img, target_idx_img, restore_str_img],
                outputs=[output_img]
            )

        with gr.Tab("🎬 Video FaceSwap"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Group():
                            source_vid_face = gr.Image(label="Source (Face)", type="filepath")
                            restore_str_vid = gr.Slider(
                                minimum=0.0, 
                                maximum=1.0, 
                                step=0.05, 
                                value=0.7, 
                                label="Face Restore Strength (GPEN 512)"
                            )

                        with gr.Group():
                            target_vid = gr.Video(label="Target Video")
                            target_idx_vid = gr.Dropdown(
                                choices=[0, 1, 2, 3, 4], 
                                value=0, 
                                label="Target Face Index"
                            )
                            gr.Markdown("Index 0 = Largest Face. Choose 1, 2, 3, etc. for other target faces.")
                            btn_vid = gr.Button("⚡ Swap Video Face!", variant="primary", size="lg")

                with gr.Column():
                    output_vid = gr.Video(label="Swapped Video Result")

            btn_vid.click(
                fn=process_video_faceswap,
                inputs=[source_vid_face, target_vid, target_idx_vid, restore_str_vid],
                outputs=[output_vid]
            )

if __name__ == "__main__":
    app.launch(share=True)
