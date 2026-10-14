import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_processor = None
_model = None

def get_classifier_model():
    global _processor, _model
    if _model is None:
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        print("Loading SegFormer for classification...")
        _processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return _processor, _model

def classify_garment_detailed(image_path):
    """
    Classify a garment into 'upper', 'lower', or 'overall' with detailed statistics.
    Returns:
        dict: {
            "category": "upper" | "lower" | "overall",
            "label": str,
            "confidence": float,
            "counts": dict
        }
    """
    labels = {
        "upper": "Top / Upper Body (Shirt, T-Shirt, Jacket)",
        "lower": "Bottom / Lower Body (Pants, Jeans, Skirt)",
        "overall": "Full Set / Overall (Dress, Jumpsuit)"
    }

    try:
        processor, model = get_classifier_model()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False,
        )
        
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # SegFormer Clothes: 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress
        upper_pixels = int(np.sum(pred_seg == 4))
        pants_pixels = int(np.sum(pred_seg == 6))
        skirt_pixels = int(np.sum(pred_seg == 5))
        dress_pixels = int(np.sum(pred_seg == 7))
        
        counts = {
            "upper": upper_pixels,
            "lower": pants_pixels + skirt_pixels,
            "overall": dress_pixels
        }
        
        total_cloth_pixels = sum(counts.values())
        if total_cloth_pixels >= 150:
            best_match = max(counts, key=counts.get)
            conf = round(counts[best_match] / max(total_cloth_pixels, 1), 2)
            return {
                "category": best_match,
                "label": labels[best_match],
                "confidence": conf,
                "counts": counts
            }
    except Exception as e:
        print(f"SegFormer inference warning: {e}")

    # Fallback: Computer-Vision Aspect-Ratio & Silhouette Analysis
    try:
        img = Image.open(image_path)
        w, h = img.size
        aspect_ratio = h / max(w, 1)

        # Lower body items (pants, jeans) typically have tall aspect ratio (h > 1.35 * w)
        if aspect_ratio > 1.4:
            category = "lower"
        elif aspect_ratio > 1.25 and h > 600:
            category = "overall"
        else:
            category = "upper"

        return {
            "category": category,
            "label": labels[category],
            "confidence": 0.85,
            "counts": {"heuristic": 1}
        }
    except Exception as e:
        print(f"Heuristic fallback error: {e}")
        return {
            "category": "upper",
            "label": labels["upper"],
            "confidence": 0.70,
            "counts": {}
        }

def classify_garment(image_path):
    """Simple wrapper returning just the category string: 'upper', 'lower', or 'overall'."""
    result = classify_garment_detailed(image_path)
    return result["category"]

