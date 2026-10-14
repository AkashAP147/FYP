import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import os

print("Loading SegFormer for Shirts...")
_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

def detect_shirt(image_path, output_path):
    print(f"Loading image: {image_path}")
    if not os.path.exists(image_path):
        return False

    img_cv = cv2.imread(image_path)
    height, width = img_cv.shape[:2]
    
    print("Running SegFormer AI for Perfect Shirts...")
    image_pil = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = _model(**inputs)
        
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    seg_map = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Class 4 is Upper-clothes
    upper_mask = (seg_map == 4)
    
    # Create the beautiful, unified aesthetic overlay
    color_seg = np.zeros_like(img_cv)
    color_seg[upper_mask] = [255, 255, 0] # Cyan (BGR) for Shirts
    
    # Blend the highlight onto the original image
    final_image = img_cv.copy()
    blended = cv2.addWeighted(img_cv, 0.4, color_seg, 0.6, 0)
    final_image[upper_mask] = blended[upper_mask]
    
    cv2.imwrite(output_path, final_image)
    print(f"Success! Final image saved to: {output_path}")
    return True
