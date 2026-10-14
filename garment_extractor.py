import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import uuid

_processor = None
_model = None

def get_model():
    global _processor, _model
    if _model is None:
        print("Loading SegFormer for Garment Extraction...")
        _processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        _model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return _processor, _model

def extract_garments(image_path, output_dir):
    """
    Extracts clothing items from an image and saves them to output_dir.
    Returns a list of dictionaries: [{'type': 'upper', 'filename': '...', 'path': '...'}, ...]
    """
    processor, model = get_model()
    image = Image.open(image_path).convert("RGB")
    
    # 1. Get Segformer mask to remove background cleanly
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Combine all potential clothing classes to create a master mask
    clothing_classes = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19]
    master_mask = np.isin(pred_seg, clothing_classes)
    
    img_cv = cv2.imread(image_path)
    h, w = img_cv.shape[:2]
    
    # 2. Get YOLO bounding boxes to separate individual items (shirt vs pants)
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
    shoe_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
    yolo = YOLO(shoe_model_path)
    
    results = yolo.predict(img_cv, conf=0.25, verbose=False)
    
    boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo.names[cls_id]
            if name == "clothing":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                
    # Sort boxes by Y coordinate (top to bottom)
    boxes.sort(key=lambda b: b[1])
    
    extracted_items = []
    os.makedirs(output_dir, exist_ok=True)
    
    white_bg = np.ones_like(img_cv) * 255
    masked_img = np.where(master_mask[:, :, None], img_cv, white_bg)
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Determine type based on relative position
        g_type = "upper" if i == 0 else "lower"
        if len(boxes) == 1:
            g_type = "overall"
            
        pad = 20
        y1_p = max(0, y1 - pad)
        y2_p = min(h, y2 + pad)
        x1_p = max(0, x1 - pad)
        x2_p = min(w, x2 + pad)
        
        cropped = masked_img[y1_p:y2_p, x1_p:x2_p]
        
        filename = f"ext_{g_type}_{uuid.uuid4().hex[:8]}.jpg"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, cropped)
        
        extracted_items.append({
            'type': g_type,
            'filename': filename,
            'path': out_path
        })
        
    return extracted_items
