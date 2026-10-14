"""
CatVTON Try-On Engine — Official Repository Pipeline
Direct implementation based on the official Zheng-Chong/CatVTON repository.
Eliminates all custom bounding-box crop/paste hacks and color shifts,
rendering authentic garments while preserving 100% of the original photo.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

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

def _lazy_load_models():
    """Lazily load official CatVTON models and processors onto GPU."""
    global _auto_masker, _vton_pipeline, _mask_processor, _repo_path
    if _vton_pipeline is not None and _auto_masker is not None:
        return

    print("=" * 60)
    print("Loading Official CatVTON Engine (Direct from Repository)")
    print("=" * 60)

    # 1. Download/verify model checkpoint
    print("[1/3] Checking CatVTON weights...")
    _repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

    # 2. Load AutoMasker (DensePose + SCHP)
    print("[2/3] Loading AutoMasker (DensePose + SCHP)...")
    _auto_masker = AutoMasker(
        densepose_ckpt=os.path.join(_repo_path, "DensePose"),
        schp_ckpt=os.path.join(_repo_path, "SCHP"),
        device='cuda'
    )
    # Offload to CPU initially to conserve VRAM
    _auto_masker.densepose_processor.predictor.model.to('cpu')
    _auto_masker.schp_processor_atr.model.to('cpu')
    _auto_masker.schp_processor_lip.model.to('cpu')
    torch.cuda.empty_cache()

    # 3. Load CatVTON Diffusion Inpainting Pipeline
    print("[3/3] Loading CatVTON Diffusion Pipeline...")
    _vton_pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=_repo_path,
        attn_ckpt_version="vitonhd",
        weight_dtype=torch.float16,
        device='cuda',
        use_tf32=True,
        skip_safety_check=True
    )
    _vton_pipeline.unet.enable_gradient_checkpointing()
    _vton_pipeline.vae.enable_slicing()

    _mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True
    )

    print("=" * 60)
    print("Official CatVTON Engine Ready")
    print("=" * 60)

def process_tryon(target_path, source_path, output_path, cloth_type='upper'):
    """
    Execute virtual try-on using the clean official CatVTON repository pipeline.
    
    Args:
        target_path: Path to target person image
        source_path: Path to garment image
        output_path: Path to save the synthesized result
        cloth_type: Clothing category ('upper', 'lower', 'overall', 'inner', 'outer')
    Returns:
        bool: True if generation succeeded, False otherwise
    """
    _lazy_load_models()

    if cloth_type not in ['upper', 'lower', 'overall', 'inner', 'outer']:
        cloth_type = 'upper'

    try:
        # Load original inputs
        person_image_orig = Image.open(target_path).convert("RGB")
        cloth_image_orig = Image.open(source_path).convert("RGB")
        orig_w, orig_h = person_image_orig.size

        # Optimal inference resolution (Divisible by 64, preserving 3:4 aspect ratio)
        width, height = 576, 768

        # Resize and prepare inputs using official CatVTON utilities
        person_image = resize_and_crop(person_image_orig, (width, height))
        cloth_image = resize_and_padding(cloth_image_orig, (width, height))

        # ── Step 1: Agnostic Masking via DensePose & SCHP ──
        print(f"\n[CatVTON 1/3] Generating agnostic mask for category: '{cloth_type}'...")
        _auto_masker.densepose_processor.predictor.model.to('cuda')
        _auto_masker.schp_processor_atr.model.to('cuda')
        _auto_masker.schp_processor_lip.model.to('cuda')

        mask_dict = _auto_masker(person_image, mask_type=cloth_type)
        mask = mask_dict['mask']

        _auto_masker.densepose_processor.predictor.model.to('cpu')
        _auto_masker.schp_processor_atr.model.to('cpu')
        _auto_masker.schp_processor_lip.model.to('cpu')
        torch.cuda.empty_cache()

        # Apply official mask boundary blur
        mask = _mask_processor.blur(mask, blur_factor=9)

        # ── Step 2: Official Diffusion Inference ──
        print("\n[CatVTON 2/3] Running CatVTON diffusion pipeline (35 steps)...")
        generator = torch.Generator(device='cuda').manual_seed(42)
        result_image = _vton_pipeline(
            image=person_image,
            condition_image=cloth_image,
            mask=mask,
            num_inference_steps=35,
            guidance_scale=2.5,
            height=height,
            width=width,
            generator=generator
        )[0]

        # ── Step 3: Seamless Repainting (Authentic Background Preservation) ──
        print("\n[CatVTON 3/3] Blending synthesized garment to preserve authentic photo...")
        mask_full = mask.resize((orig_w, orig_h), Image.BILINEAR)
        result_full = result_image.resize((orig_w, orig_h), Image.LANCZOS)
        final_result = repaint_result(result_full, person_image_orig, mask_full)

        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        final_result.save(output_path, quality=95)
        print(f"Success! Output saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error during CatVTON try-on: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_dual_tryon(target_path, top_path, bottom_path, output_path):
    """
    Execute chained virtual try-on fitting BOTH top and bottom garments onto person.
    Pass 1: Person + Top Garment ('upper') -> intermediate look.
    Pass 2: Intermediate look + Bottom Garment ('lower') -> complete full outfit!
    
    Args:
        target_path: Path to target person image
        top_path: Path to upper garment image (shirt, top, jacket)
        bottom_path: Path to lower garment image (pants, jeans, skirt)
        output_path: Path to save the combined outfit result
    Returns:
        bool: True if generation succeeded, False otherwise
    """
    temp_dir = os.path.dirname(os.path.abspath(output_path))
    inter_filename = f"inter_{os.path.basename(output_path)}"
    inter_path = os.path.join(temp_dir, inter_filename)

    print("\n" + "=" * 60)
    print("DUAL GARMENT TRY-ON — PASS 1/2: Fitting Upper Garment (Top/Shirt)")
    print("=" * 60)
    ok_top = process_tryon(target_path, top_path, inter_path, cloth_type='upper')
    if not ok_top:
        print("Pass 1 (Top) failed!")
        return False

    print("\n" + "=" * 60)
    print("DUAL GARMENT TRY-ON — PASS 2/2: Fitting Lower Garment (Bottom/Pants)")
    print("=" * 60)
    ok_bottom = process_tryon(inter_path, bottom_path, output_path, cloth_type='lower')

    # Clean up intermediate file
    try:
        if os.path.exists(inter_path):
            os.remove(inter_path)
    except Exception:
        pass

    return ok_bottom

