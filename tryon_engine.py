"""
CatVTON Try-On Engine — Official Repository Pipeline
Direct implementation based on the official Zheng-Chong/CatVTON repository.

Auto-detects environment:
  - COLAB (High GPU): All models stay on GPU, xformers, 768x1024, 50 steps → FAST
  - LAPTOP (Low/No GPU): CPU offloading, 576x768, 30 steps → Regular
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import numpy as np

# NumPy 1.x / 2.x Compatibility Shim (fixes Detectron2 & CatVTON np.int and np.float removals)
try:
    _ = np.int
except (AttributeError, Exception):
    np.int = int
try:
    _ = np.float
except (AttributeError, Exception):
    np.float = float
try:
    _ = np.bool
except (AttributeError, Exception):
    np.bool = bool

from PIL import Image, ImageFilter

# Ensure CatVTON package is in system path
CATVTON_DIR = os.path.join(os.path.dirname(__file__), "CatVTON")
if CATVTON_DIR not in sys.path:
    sys.path.insert(0, CATVTON_DIR)

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import resize_and_crop, resize_and_padding, repaint_result

_auto_masker = None
_vton_pipeline = None
_mask_processor = None
_repo_path = None
_perf_profile = None  # 'colab' or 'laptop'

# ═══════════════════════════════════════════════════════════════
#  Performance Profiles — auto-detected or device-specific
# ═══════════════════════════════════════════════════════════════
PROFILES = {
    'colab': {
        'width': 768,
        'height': 1024,
        'num_inference_steps': 50,      # 50 steps is regular baseline
        'guidance_scale': 2.5,
        'blur_factor': 9,               # Official CatVTON blur factor
        'keep_models_on_gpu': True,     # Pinned on GPU — no PCIe transfer lag
        'use_xformers': True,           # Memory-efficient attention
        'gradient_checkpointing': False,
        'vae_slicing': False,
        'simple_mode': False,
    },
    'laptop': {
        'width': 576,
        'height': 768,
        'num_inference_steps': 50,      # 50 regular steps
        'guidance_scale': 2.5,
        'blur_factor': 9,
        'keep_models_on_gpu': False,    # Offload to CPU to save VRAM
        'use_xformers': False,
        'gradient_checkpointing': True,
        'vae_slicing': True,
        'simple_mode': False,
    },
    'mobile': {
        'width': 512,
        'height': 680,
        'num_inference_steps': 50,      # 50 regular steps, ultra-fast on compact grid
        'guidance_scale': 2.5,
        'blur_factor': 7,
        'keep_models_on_gpu': True,     # GPU pinned
        'use_xformers': True,
        'gradient_checkpointing': False,
        'vae_slicing': False,
        'simple_mode': True,            # Simple fast processing: direct feathering & fast save
    }
}


def _detect_environment():
    """Auto-detect whether we're on Colab (high GPU) or laptop."""
    global _perf_profile

    forced = os.environ.get('TRYON_PROFILE', '').lower()
    if forced in ('colab', 'laptop', 'mobile'):
        _perf_profile = forced
        print(f"[Engine] Profile forced via TRYON_PROFILE={forced}")
        return

    is_colab = os.environ.get('COLAB_RELEASE_TAG') is not None or \
               os.path.exists('/content')

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device_name = torch.cuda.get_device_name(0)

        # High GPU Colab setup: enable cuDNN parallel acceleration
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if is_colab or vram_gb >= 12:
            _perf_profile = 'colab'
            print(f"[Engine] ⚡ HIGH-PERFORMANCE GPU MODE — {device_name} ({vram_gb:.1f}GB VRAM)")
            print(f"[Engine]   Parallel cuDNN: ON | Resolution: 768×1024 | Steps: 50 | GPU Pinned")
        else:
            _perf_profile = 'laptop'
            print(f"[Engine] 🔋 REGULAR MODE — {device_name} ({vram_gb:.1f}GB VRAM)")
            print(f"[Engine]   Resolution: 576×768 | Steps: 50 | CPU offloading enabled")
    else:
        _perf_profile = 'laptop'
        print(f"[Engine] 🔋 REGULAR MODE — CPU only (no CUDA)")
        print(f"[Engine]   Resolution: 576×768 | Steps: 50")


def _get_profile(is_mobile=False):
    """Get current performance profile settings."""
    if is_mobile:
        return PROFILES['mobile']
    if _perf_profile is None:
        _detect_environment()
    return PROFILES[_perf_profile]


def _lazy_load_models():
    """Lazily load official CatVTON models and processors onto GPU."""
    global _auto_masker, _vton_pipeline, _mask_processor, _repo_path
    if _vton_pipeline is not None and _auto_masker is not None:
        return

    profile = _get_profile()

    print("=" * 60)
    print(f"Loading Official CatVTON Engine [{_perf_profile.upper()} mode]")
    print("=" * 60)

    # 1. Download/verify model checkpoint
    print("[1/3] Checking CatVTON weights...")
    _repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enable parallel PyTorch settings for high GPU
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 2. Load AutoMasker (DensePose + SCHP)
    print("[2/3] Loading AutoMasker (DensePose + SCHP)...")
    _auto_masker = AutoMasker(
        densepose_ckpt=os.path.join(_repo_path, "DensePose"),
        schp_ckpt=os.path.join(_repo_path, "SCHP"),
        device=device
    )

    if not profile['keep_models_on_gpu']:
        _auto_masker.densepose_processor.predictor.model.to('cpu')
        _auto_masker.schp_processor_atr.model.to('cpu')
        _auto_masker.schp_processor_lip.model.to('cpu')
        torch.cuda.empty_cache()
        print("  [Laptop] Masker models offloaded to CPU")
    else:
        print("  [Colab] Masker models pinned to GPU (Parallel streams ready) ⚡")

    # 3. Load CatVTON Diffusion Inpainting Pipeline
    print("[3/3] Loading CatVTON Diffusion Pipeline (mix-48k-1024 high-res)...")
    _vton_pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=_repo_path,
        attn_ckpt_version="mix",        # Official 1024x768 48k checkpoint
        weight_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device=device,
        use_tf32=True,
        keep_models_on_gpu=profile['keep_models_on_gpu'],
        skip_safety_check=True
    )

    # Apply performance optimizations based on profile
    if profile['gradient_checkpointing']:
        _vton_pipeline.unet.enable_gradient_checkpointing()
        print("  [Laptop] Gradient checkpointing enabled")

    if profile['vae_slicing']:
        _vton_pipeline.vae.enable_slicing()
        print("  [Laptop] VAE slicing enabled")

    if profile['use_xformers']:
        try:
            _vton_pipeline.unet.enable_xformers_memory_efficient_attention()
            print("  [Colab] xFormers memory-efficient attention enabled ⚡")
        except Exception:
            print("  [Colab] xFormers not available, using default attention")

    _mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True
    )

    print("=" * 60)
    print(f"Official CatVTON Engine Ready [{_perf_profile.upper()}]")
    print("=" * 60)

def process_tryon(target_path, source_path, output_path, cloth_type='upper', is_mobile=False):
    """
    Execute virtual try-on using the clean official CatVTON repository pipeline.
    
    Performance Profiles:
      - Colab GPU (Desktop): 768×1024, 50 regular steps, GPU pinned, cuDNN parallel acceleration
      - Colab GPU (Mobile):  512×680, 50 regular steps, simple fast processing (<15s)
      - Laptop (CPU/Low GPU): 576×768, 50 regular steps, CPU offloading
    
    Args:
        target_path: Path to target person image
        source_path: Path to garment image
        output_path: Path to save the synthesized result
        cloth_type: Clothing category ('upper', 'lower', 'overall', 'inner', 'outer')
        is_mobile: If True, uses lightweight mobile resolution and simple fast blending
    Returns:
        bool: True if generation succeeded, False otherwise
    """
    _lazy_load_models()
    profile = _get_profile(is_mobile=is_mobile)

    if cloth_type not in ['upper', 'lower', 'overall', 'inner', 'outer']:
        cloth_type = 'upper'

    try:
        t_start = time.time()

        # Load original inputs
        person_image_orig = Image.open(target_path).convert("RGB")
        cloth_image_orig = Image.open(source_path).convert("RGB")
        orig_w, orig_h = person_image_orig.size

        width = profile['width']
        height = profile['height']
        mode_label = "MOBILE-FAST" if is_mobile else _perf_profile.upper()

        # Resize and prepare inputs using official CatVTON utilities
        person_image = resize_and_crop(person_image_orig, (width, height))
        cloth_image = resize_and_padding(cloth_image_orig, (width, height))

        # ── Step 1: Agnostic Masking via DensePose & SCHP ──
        print(f"\n[CatVTON 1/3] Generating agnostic mask for '{cloth_type}' ({width}×{height}) [{mode_label}]...")
        t1 = time.time()

        if not profile['keep_models_on_gpu']:
            # Laptop: move masker models to GPU temporarily
            _auto_masker.densepose_processor.predictor.model.to('cuda')
            _auto_masker.schp_processor_atr.model.to('cuda')
            _auto_masker.schp_processor_lip.model.to('cuda')

        mask_dict = _auto_masker(person_image, mask_type=cloth_type)
        mask = mask_dict['mask']

        if not profile['keep_models_on_gpu']:
            # Laptop: offload back to CPU
            _auto_masker.densepose_processor.predictor.model.to('cpu')
            _auto_masker.schp_processor_atr.model.to('cpu')
            _auto_masker.schp_processor_lip.model.to('cpu')
            torch.cuda.empty_cache()

        print(f"  Mask generated in {time.time() - t1:.1f}s")

        # Apply official mask boundary blur
        mask = _mask_processor.blur(mask, blur_factor=profile['blur_factor'])

        # ── Step 2: Official Diffusion Inference ──
        steps = profile['num_inference_steps']
        print(f"\n[CatVTON 2/3] Running diffusion pipeline ({steps} steps, {width}×{height}) [{mode_label}]...")
        t2 = time.time()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = torch.Generator(device=device).manual_seed(42)
        result_image = _vton_pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=profile['guidance_scale'],
            height=height,
            width=width,
            generator=generator
        )[0]

        print(f"  Diffusion completed in {time.time() - t2:.1f}s")

        # ── Step 3: Seamless Repainting (Authentic Background Preservation) ──
        simple_mode = profile.get('simple_mode', False)
        print(f"\n[CatVTON 3/3] Blending synthesized garment ({'Simple/Fast' if simple_mode else 'Studio/Crisp Feathered'})...")
        mask_full = mask.resize((orig_w, orig_h), Image.LANCZOS)
        result_full = result_image.resize((orig_w, orig_h), Image.LANCZOS)
        final_result = repaint_result(result_full, person_image_orig, mask_full, simple_mode=simple_mode)

        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if simple_mode:
            final_result.save(output_path, quality=88, optimize=True)
        else:
            final_result.save(output_path, quality=95)

        total_time = time.time() - t_start
        print(f"✅ Done in {total_time:.1f}s [{mode_label}] → {output_path}")
        return True

    except Exception as e:
        print(f"Error during CatVTON try-on: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_dual_tryon(target_path, top_path, bottom_path, output_path, is_mobile=False):
    """
    Execute chained virtual try-on fitting BOTH top and bottom garments onto person.
    Pass 1: Person + Top Garment ('upper') -> intermediate look.
    Pass 2: Intermediate look + Bottom Garment ('lower') -> complete full outfit!
    
    Args:
        target_path: Path to target person image
        top_path: Path to upper garment image (shirt, top, jacket)
        bottom_path: Path to lower garment image (pants, jeans, skirt)
        output_path: Path to save the combined outfit result
        is_mobile: If True, uses fast mobile processing profile
    Returns:
        bool: True if generation succeeded, False otherwise
    """
    temp_dir = os.path.dirname(os.path.abspath(output_path))
    inter_filename = f"inter_{os.path.basename(output_path)}"
    inter_path = os.path.join(temp_dir, inter_filename)

    mode_label = "MOBILE-FAST" if is_mobile else "DESKTOP"
    print("\n" + "=" * 60)
    print(f"DUAL GARMENT TRY-ON [{mode_label}] — PASS 1/2: Fitting Upper Garment (Top/Shirt)")
    print("=" * 60)
    ok_top = process_tryon(target_path, top_path, inter_path, cloth_type='upper', is_mobile=is_mobile)
    if not ok_top:
        print("Pass 1 (Top) failed!")
        return False

    print("\n" + "=" * 60)
    print(f"DUAL GARMENT TRY-ON [{mode_label}] — PASS 2/2: Fitting Lower Garment (Bottom/Pants)")
    print("=" * 60)
    ok_bottom = process_tryon(inter_path, bottom_path, output_path, cloth_type='lower', is_mobile=is_mobile)

    # Clean up intermediate file
    try:
        if os.path.exists(inter_path):
            os.remove(inter_path)
    except Exception:
        pass

    return ok_bottom
