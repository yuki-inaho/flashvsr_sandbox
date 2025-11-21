#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified FlashVSR v1.1 inference test script
For basic functionality testing with limited GPU memory
"""
import os
import sys
from pathlib import Path

# Add FlashVSR to Python path
WORKSPACE_ROOT = Path(__file__).parent.parent
FLASHVSR_DIR = WORKSPACE_ROOT / "FlashVSR"
sys.path.insert(0, str(FLASHVSR_DIR))
sys.path.insert(0, str(FLASHVSR_DIR / "examples" / "WanVSR"))

import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

print("Imports successful. Attempting to load FlashVSR components...")

try:
    from diffsynth import ModelManager, FlashVSRTinyPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder
    print("✓ FlashVSR components imported successfully")
except ImportError as e:
    print(f"✗ Failed to import FlashVSR components: {e}")
    print("\nPlease ensure:")
    print("1. FlashVSR repository is cloned in the workspace")
    print("2. All dependencies are installed (uv sync)")
    print("3. Block-Sparse Attention is installed")
    sys.exit(1)


def tensor2video(frames):
    """Convert tensor to video frames."""
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    """Convert PIL image to tensor in range [-1, 1]."""
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def save_video(frames, save_path, fps=30, quality=5):
    """Save frames as video."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    """Compute scaled and target dimensions."""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    """Upscale and center crop image."""
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def largest_8n1_leq(n):
    """Compute largest 8n-3 <= n."""
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def prepare_input_video(video_path: str, scale: float = 4, max_frames: int = None,
                        dtype=torch.bfloat16, device='cuda'):
    """Prepare input video tensor."""
    rdr = imageio.get_reader(video_path)
    first = Image.fromarray(rdr.get_data(0)).convert('RGB')
    w0, h0 = first.size

    meta = {}
    try:
        meta = rdr.get_meta_data()
    except Exception:
        pass
    fps_val = meta.get('fps', 30)
    fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

    # Count total frames
    total = 0
    try:
        nf = meta.get('nframes', None)
        if isinstance(nf, int) and nf > 0:
            total = nf
        else:
            total = rdr.count_frames()
    except Exception:
        # Fallback: count by reading
        n = 0
        try:
            while True:
                rdr.get_data(n)
                n += 1
        except Exception:
            total = n

    if total <= 0:
        rdr.close()
        raise RuntimeError(f"Cannot read frames from {video_path}")

    # Limit frames for testing
    if max_frames is not None and max_frames > 0:
        total = min(total, max_frames)
        print(f"[Test Mode] Limiting to first {total} frames")

    print(f"[{os.path.basename(video_path)}] Original Resolution: {w0}x{h0} | "
          f"Original Frames: {total} | FPS: {fps}")

    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    print(f"[{os.path.basename(video_path)}] Scaled (x{scale:.2f}): {sW}x{sH} -> "
          f"Target (128-multiple): {tW}x{tH}")

    # Pad frames
    idx = list(range(total)) + [total - 1] * 4
    F = largest_8n1_leq(len(idx))
    if F == 0:
        rdr.close()
        raise RuntimeError(f"Not enough frames after padding. Got {len(idx)}.")
    idx = idx[:F]
    print(f"[{os.path.basename(video_path)}] Target Frames (8n-3): {F - 4}")

    frames = []
    try:
        for i in tqdm(idx, desc="Loading frames"):
            img = Image.fromarray(rdr.get_data(i)).convert('RGB')
            img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
    finally:
        try:
            rdr.close()
        except Exception:
            pass

    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
    return vid, tH, tW, F, fps


def init_pipeline(model_dir: Path):
    """Initialize FlashVSR pipeline."""
    print("\n" + "=" * 50)
    print("Initializing FlashVSR Pipeline")
    print("=" * 50)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}\n"
                                "Please run: bash scripts/download_model_v1.1.sh")

    # Check required model files
    required_files = [
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ]
    for filename in required_files:
        if not (model_dir / filename).exists():
            raise FileNotFoundError(f"Required model file not found: {filename}")

    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Change to model directory for relative paths
    original_dir = os.getcwd()
    os.chdir(model_dir.parent)

    try:
        # Initialize ModelManager
        mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        mm.load_models([
            str(model_dir / "diffusion_pytorch_model_streaming_dmd.safetensors"),
        ])

        # Create pipeline
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")

        # Load LQ projection
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1
        ).to("cuda", dtype=torch.bfloat16)

        LQ_proj_in_path = model_dir / "LQ_proj_in.ckpt"
        if LQ_proj_in_path.exists():
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(str(LQ_proj_in_path), map_location="cpu"), strict=True
            )
        pipe.denoising_model().LQ_proj_in.to('cuda')

        # Load TCDecoder
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(
            new_channels=multi_scale_channels, new_latent_channels=16 + 768
        )
        mis = pipe.TCDecoder.load_state_dict(
            torch.load(str(model_dir / "TCDecoder.ckpt")), strict=False
        )
        print(f"TCDecoder load info: {mis}")

        # Setup pipeline
        pipe.to('cuda')
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv()
        pipe.load_models_to_device(["dit", "vae"])

        print("✓ Pipeline initialized successfully")
        return pipe

    finally:
        os.chdir(original_dir)


def main():
    """Main inference test."""
    print("\n" + "=" * 50)
    print("FlashVSR v1.1 Inference Test")
    print("=" * 50 + "\n")

    # Configuration
    RESULT_ROOT = WORKSPACE_ROOT / "results"
    MODEL_DIR = FLASHVSR_DIR / "examples" / "WanVSR" / "FlashVSR-v1.1"
    INPUTS_DIR = FLASHVSR_DIR / "examples" / "WanVSR" / "inputs"

    # Test parameters
    SEED = 0
    SCALE = 4.0
    DTYPE = torch.bfloat16
    DEVICE = 'cuda'
    SPARSE_RATIO = 2.0  # 1.5 or 2.0 (2.0 = more stable)
    LOCAL_RANGE = 11    # 9 or 11 (11 = more stable)
    MAX_FRAMES = 16     # Limit frames for testing (set to None for full video)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    # Find sample video (use smallest for testing)
    if not INPUTS_DIR.exists():
        print(f"✗ Inputs directory not found: {INPUTS_DIR}")
        sys.exit(1)

    videos = sorted(INPUTS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_size)
    if not videos:
        print(f"✗ No sample videos found in {INPUTS_DIR}")
        sys.exit(1)

    # Use smallest video for testing
    test_video = videos[0]
    print(f"Test video: {test_video.name} ({test_video.stat().st_size / 1024**2:.2f} MB)\n")

    # Initialize pipeline
    try:
        pipe = init_pipeline(MODEL_DIR)
    except Exception as e:
        print(f"\n✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Prepare input
    print("\n" + "=" * 50)
    print("Preparing Input")
    print("=" * 50)

    try:
        LQ, th, tw, F, fps = prepare_input_video(
            str(test_video), scale=SCALE, max_frames=MAX_FRAMES,
            dtype=DTYPE, device=DEVICE
        )
    except Exception as e:
        print(f"\n✗ Failed to prepare input: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run inference
    print("\n" + "=" * 50)
    print("Running Inference")
    print("=" * 50)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    try:
        print(f"Input shape: {LQ.shape}")
        print(f"Target resolution: {tw}x{th}")
        print(f"Frames: {F}")
        print(f"sparse_ratio: {SPARSE_RATIO}")
        print(f"local_range: {LOCAL_RANGE}\n")

        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=SEED,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=SPARSE_RATIO * 768 * 1280 / (th * tw),
            kv_ratio=3.0,
            local_range=LOCAL_RANGE,
            color_fix=True,
        )

        print("✓ Inference completed successfully")

    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save result
    print("\n" + "=" * 50)
    print("Saving Result")
    print("=" * 50)

    try:
        RESULT_ROOT.mkdir(exist_ok=True)
        video_frames = tensor2video(video)
        output_path = RESULT_ROOT / f"test_output_{test_video.stem}_seed{SEED}.mp4"
        save_video(video_frames, str(output_path), fps=fps, quality=6)
        print(f"\n✓ Result saved to: {output_path}")
        print(f"  Output frames: {len(video_frames)}")
        print(f"  Resolution: {tw}x{th}")
        print(f"  FPS: {fps}")

    except Exception as e:
        print(f"\n✗ Failed to save result: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Memory report
    print("\n" + "=" * 50)
    print("GPU Memory Report")
    print("=" * 50)
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

    print("\n" + "=" * 50)
    print("✓ Test completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
