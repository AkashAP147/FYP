import cv2
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import os

def recognize_clothing(image_path, output_path):
    print(f"Loading image: {image_path}")
    if not os.path.exists(image_path):
        print("Error: Image not found!")
        return
        
    print("Loading SegFormer Semantic Clothing model...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    
    image = Image.open(image_path).convert("RGB")
    
    print("Analyzing clothing pixels...")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    labels = {
        0: "Background", 1: "Hat", 2: "Hair", 3: "Glove", 4: "Sunglasses", 
        5: "Upper-clothes (Shirt/T-shirt)", 6: "Dress", 7: "Coat", 8: "Socks", 
        9: "Pants/Jeans", 10: "Jumpsuits", 11: "Scarf", 12: "Skirt", 13: "Face", 
        14: "Left-arm", 15: "Right-arm", 16: "Left-leg", 17: "Right-leg", 
        18: "Left-shoe", 19: "Right-shoe"
    }
    
    colors = {
        1: [255, 255, 0],   # Hat (Cyan)
        3: [255, 165, 0],   # Glove (Orange)
        4: [128, 0, 128],   # Sunglasses (Purple)
        5: [0, 255, 255],   # Upper-clothes (Yellow)
        6: [255, 20, 147],  # Dress (Pink)
        7: [255, 0, 255],   # Coat (Magenta)
        8: [0, 128, 128],   # Socks (Teal)
        9: [255, 0, 0],     # Pants/Jeans (Blue)
        10: [0, 255, 127],  # Jumpsuits (Spring Green)
        11: [255, 69, 0],   # Scarf (Red-Orange)
        12: [0, 165, 255],  # Skirt (Orange)
        18: [0, 255, 0],    # Shoes (Green)
        19: [0, 255, 0],    
    }
    
    valid_classes = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19]

    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    valid_mask = np.isin(pred_seg, valid_classes)
    
    for c in valid_classes:
        if c in colors:
            bgr_color = colors[c][::-1] 
            color_seg[pred_seg == c] = bgr_color
            
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()
    
    blended = cv2.addWeighted(img_cv, 0.4, color_seg, 0.6, 0)
    overlay[valid_mask] = blended[valid_mask]
    
    detected_classes = np.unique(pred_seg)
    y_offset = 40
    for c in detected_classes:
        if c not in valid_classes: continue
            
        label = labels[c]
        color_bgr = colors.get(c, [200, 200, 200])[::-1] 
        
        cv2.rectangle(overlay, (15, y_offset - 25), (400, y_offset + 10), (0, 0, 0), -1)
        cv2.putText(overlay, f"{label} Detected", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        y_offset += 45
        
    cv2.imwrite(output_path, overlay)
    print(f"Success! Analyzed clothing saved to: {output_path}")

if __name__ == "__main__":
    input_image = "IMG_20260725_131929.jpg"
    output_image = "clothing_output.jpg"
    recognize_clothing(input_image, output_image)
