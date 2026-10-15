import os
from PIL import Image
from typing import Union
import numpy as np
import cv2
from diffusers.image_processor import VaeImageProcessor
import torch

from model.SCHP import SCHP  # type: ignore
from model.DensePose import DensePose  # type: ignore

DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}

ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}

LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PROTECT_BODY_PARTS = {
    'upper': ['Left-leg', 'Right-leg'],
    'lower': ['Right-arm', 'Left-arm', 'Face'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg'],
    'outer': ['Left-leg', 'Right-leg'],
}
PROTECT_CLOTH_PARTS = {
    'upper': {
        'ATR': ['Skirt', 'Pants'],
        'LIP': ['Skirt', 'Pants']
    },
    'lower': {
        'ATR': ['Upper-clothes'],
        'LIP': ['Upper-clothes', 'Coat']
    },
    'overall': {
        'ATR': [],
        'LIP': []
    },
    'inner': {
        'ATR': ['Dress', 'Coat', 'Skirt', 'Pants'],
        'LIP': ['Dress', 'Coat', 'Skirt', 'Pants', 'Jumpsuits']
    },
    'outer': {
        'ATR': ['Dress', 'Pants', 'Skirt'],
        'LIP': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Jumpsuits']
    }
}
MASK_CLOTH_PARTS = {
    'upper': ['Upper-clothes', 'Coat', 'Dress', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}
MASK_DENSE_PARTS = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}
    
schp_public_protect_parts = ['Hat', 'Hair', 'Sunglasses', 'Left-shoe', 'Right-shoe', 'Bag', 'Glove', 'Scarf']
schp_protect_parts = {
    'upper': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits'],  
    'lower': ['Left-arm', 'Right-arm', 'Upper-clothes', 'Coat'],
    'overall': [],
    'inner': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Coat'],
    'outer': ['Left-leg', 'Right-leg', 'Skirt', 'Pants', 'Jumpsuits', 'Upper-clothes']
}
schp_mask_parts = {
    'upper': ['Upper-clothes', 'Dress', 'Coat', 'Jumpsuits'],
    'lower': ['Pants', 'Skirt', 'Dress', 'Jumpsuits', 'socks'],
    'overall': ['Upper-clothes', 'Dress', 'Pants', 'Skirt', 'Coat', 'Jumpsuits', 'socks'],
    'inner': ['Upper-clothes'],
    'outer': ['Coat',]
}

dense_mask_parts = {
    'upper': ['torso', 'big arms', 'forearms'],
    'lower': ['thighs', 'legs'],
    'overall': ['torso', 'thighs', 'legs', 'big arms', 'forearms'],
    'inner': ['torso'],
    'outer': ['torso', 'big arms', 'forearms']
}

def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))

def part_mask_of(part: Union[str, list],
                 parse: np.ndarray, mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask

def hull_mask(mask_area: np.ndarray):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask
    

class AutoMasker:
    def __init__(
        self, 
        densepose_ckpt='./Models/DensePose', 
        schp_ckpt='./Models/SCHP', 
        device='cuda'):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        self.device = device
        self.densepose_processor = DensePose(densepose_ckpt, device)
        self.schp_processor_atr = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908301523-atr.pth'), device=device)
        self.schp_processor_lip = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908261155-lip.pth'), device=device)
        
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def process_densepose(self, image_or_path):
        return self.densepose_processor(image_or_path, resize=1024)

    def process_schp_lip(self, image_or_path):
        return self.schp_processor_lip(image_or_path)

    def process_schp_atr(self, image_or_path):
        return self.schp_processor_atr(image_or_path)
        
    def preprocess_image(self, image_or_path):
        # Parallel GPU execution for DensePose + SCHP (ATR & LIP)
        if torch.cuda.is_available() and getattr(self, 'device', 'cpu') != 'cpu':
            from concurrent.futures import ThreadPoolExecutor
            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    f_dp = executor.submit(self.densepose_processor, image_or_path, resize=1024)
                    f_atr = executor.submit(self.schp_processor_atr, image_or_path)
                    f_lip = executor.submit(self.schp_processor_lip, image_or_path)
                    return {
                        'densepose': f_dp.result(),
                        'schp_atr': f_atr.result(),
                        'schp_lip': f_lip.result()
                    }
            except Exception as e:
                print(f"[AutoMasker] Parallel execution notice: {e}, falling back to sequential")
        return {
            'densepose': self.densepose_processor(image_or_path, resize=1024),
            'schp_atr': self.schp_processor_atr(image_or_path),
            'schp_lip': self.schp_processor_lip(image_or_path)
        }
    
    @staticmethod
    def cloth_agnostic_mask(
        densepose_mask: Image.Image,
        schp_lip_mask: Image.Image,
        schp_atr_mask: Image.Image,
        part: str='overall',
        **kwargs
    ):
        assert part in ['upper', 'lower', 'overall', 'inner', 'outer'], f"part should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {part}"
        w, h = densepose_mask.size
        
        densepose_mask = np.array(densepose_mask)
        schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)
        
        # 1. Background Mask: identify genuine background pixels from both SCHP segmentations
        background_area = (part_mask_of(['Background'], schp_lip_mask, LIP_MAPPING) & 
                           part_mask_of(['Background'], schp_atr_mask, ATR_MAPPING))
        
        # 2. Strong Protection: Hands, Face, Accessories, Feet
        hand_kernel = np.ones((max(3, max(w, h) // 180), max(3, max(w, h) // 180)), np.uint8)
        hands_dense = part_mask_of(['hands', 'feet'], densepose_mask, DENSE_INDEX_MAP)
        hands_protect_area = cv2.dilate(hands_dense.astype(np.uint8), hand_kernel, iterations=2).astype(bool)
        
        face_protect_area = (part_mask_of('Face', schp_lip_mask, LIP_MAPPING) | 
                             part_mask_of('Face', schp_atr_mask, ATR_MAPPING))
        face_protect_area = cv2.dilate(face_protect_area.astype(np.uint8), hand_kernel, iterations=1).astype(bool)
        strong_protect_area = hands_protect_area | face_protect_area
        
        # 3. Weak Protection: Hair, Irrelevant Clothes, Body Parts (Legs for upper, Arms for lower)
        body_protect_area = (part_mask_of(PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING) | 
                             part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING))
        hair_protect_area = (part_mask_of(['Hair'], schp_lip_mask, LIP_MAPPING) | 
                             part_mask_of(['Hair'], schp_atr_mask, ATR_MAPPING))
        cloth_protect_area = (part_mask_of(PROTECT_CLOTH_PARTS[part]['LIP'], schp_lip_mask, LIP_MAPPING) | 
                              part_mask_of(PROTECT_CLOTH_PARTS[part]['ATR'], schp_atr_mask, ATR_MAPPING))
        accessory_parts = ['Hat', 'Glove', 'Sunglasses', 'Bag', 'Left-shoe', 'Right-shoe', 'Scarf', 'Socks']
        accessory_protect_area = (part_mask_of(accessory_parts, schp_lip_mask, LIP_MAPPING) | 
                                  part_mask_of(accessory_parts, schp_atr_mask, ATR_MAPPING))
        
        weak_protect_area = (body_protect_area | cloth_protect_area | hair_protect_area | 
                             strong_protect_area | accessory_protect_area)
        
        # 4. Garment Area: Exact existing garment detected on the person
        strong_mask_area = (part_mask_of(MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING) | 
                            part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING))
        
        # DensePose target anatomy
        mask_dense_area = part_mask_of(MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP)
        
        # 5. Body Contour Preservation (Prevents Ballooning / Oversized Fit)
        # Apply convex hull ONLY to the torso / garment core, NOT bridging arms to torso through air!
        torso_dense = part_mask_of(['torso'], densepose_mask, DENSE_INDEX_MAP)
        torso_core = (torso_dense | strong_mask_area) & (~weak_protect_area) & (~background_area)
        if np.any(torso_core):
            torso_hull = hull_mask((torso_core.astype(np.uint8) * 255)) // 255
            # Strictly exclude background so the hull never expands into empty air between arms & waist
            torso_hull = torso_hull & (~background_area) & (~weak_protect_area)
        else:
            torso_hull = torso_core
        
        # Combine: Torso core + existing garment + anatomical arms
        mask_area = torso_hull | strong_mask_area | mask_dense_area
        
        # Enforce hard boundaries: NO background, NO protected anatomy
        mask_area = mask_area & (~weak_protect_area) & (~strong_protect_area) & (~background_area)
        
        # Morphological close to bridge small internal wrinkles inside the garment
        close_kernel_size = max(max(w, h) // 100, 3)
        if close_kernel_size % 2 == 0:
            close_kernel_size += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        mask_area = cv2.morphologyEx(mask_area.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
        
        # Subtle 1px boundary dilation for clean anti-aliased edge
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)
        
        # STRICT FINAL ENFORCEMENT: Mask must NEVER touch background outside the person's body!
        mask_area = mask_area & (~background_area) & (~strong_protect_area)
        
        return Image.fromarray((mask_area * 255).astype(np.uint8))
        
    def __call__(
        self,
        image: Union[str, Image.Image],
        mask_type: str = "upper",
    ):
        assert mask_type in ['upper', 'lower', 'overall', 'inner', 'outer'], f"mask_type should be one of ['upper', 'lower', 'overall', 'inner', 'outer'], but got {mask_type}"
        preprocess_results = self.preprocess_image(image)
        mask = self.cloth_agnostic_mask(
            preprocess_results['densepose'], 
            preprocess_results['schp_lip'], 
            preprocess_results['schp_atr'], 
            part=mask_type,
        )
        return {
            'mask': mask,
            'densepose': preprocess_results['densepose'],
            'schp_lip': preprocess_results['schp_lip'],
            'schp_atr': preprocess_results['schp_atr']
        }


if __name__ == '__main__':
    pass
