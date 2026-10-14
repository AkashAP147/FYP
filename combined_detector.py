import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import os

def detect_all(image_path, output_path):
    print(f"Loading image: {image_path}")
    if not os.path.exists(image_path):
        return False

    img_cv = cv2.imread(image_path)
    height, width = img_cv.shape[:2]

    # 1. Segformer for "Cover Everything" Organic Clothing Masks
    print("Loading SegFormer for Clothing...")
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    
    image_pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=image_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    seg_map = upsampled_logits.argmax(dim=1)[0].numpy()

    # 2. YOLO Object Detection for Shoes (Since Segformer misses them)
    print("Loading YOLO Object Detection for Shoes...")
    shoe_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
    shoe_model = YOLO(shoe_model_path)
    shoe_results = shoe_model.predict(img_cv, conf=0.25)
    
    shoe_mask = np.zeros((height, width), dtype=bool)
    shoe_exclusion_mask = np.zeros((height, width), dtype=bool) # Used to erase pants
    
    image_area = height * width
    
    if len(shoe_results) > 0:
        for box in shoe_results[0].boxes:
            class_name = shoe_model.names[int(box.cls[0])].lower()
            if "shoe" in class_name or "foot" in class_name:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                
                # Filter out tiny background shoes
                if box_area > image_area * 0.005:
                    center = ((x1+x2)//2, (y1+y2)//2)
                    axes = ((x2-x1)//2, (y2-y1)//2)
                    
                    # Draw green oval for aesthetic
                    temp_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.ellipse(temp_mask, center, axes, 0, 0, 360, 255, -1)
                    temp_mask = cv2.GaussianBlur(temp_mask, (15, 15), 0)
                    shoe_mask[temp_mask > 50] = True
                    
                    # Exclusion mask (a bit larger to ensure pants are completely erased from shoes)
                    cv2.ellipse(shoe_exclusion_mask, center, (axes[0]+15, axes[1]+15), 0, 0, 360, True, -1)

    # 3. Create Semantic Color Overlays
    color_seg = np.zeros_like(img_cv)
    final_mask = np.zeros((height, width), dtype=bool)
    
    # Pants (Magenta) - Classes 9 (Pants), 12 (Skirt)
    pants_mask = (seg_map == 9) | (seg_map == 12)
    
    # ERASE shoes from pants! (As requested: "recognize shoes by yolo and ignore it")
    pants_mask[shoe_exclusion_mask] = False
    
    color_seg[pants_mask] = [255, 0, 255] # Magenta
    final_mask[pants_mask] = True

    # Shirts (Cyan) - Classes 5 (Upper-clothes), 6 (Dress), 7 (Coat)
    shirt_mask = (seg_map == 5) | (seg_map == 6) | (seg_map == 7)
    
    # Since Shirts are drawn AFTER pants, they will properly overlay the pants (Z-index physics)
    color_seg[shirt_mask] = [255, 255, 0] # Cyan
    final_mask[shirt_mask] = True
    
    # Shoes (Green)
    color_seg[shoe_mask] = [0, 255, 0] # Green
    final_mask[shoe_mask] = True
    
    # Blend the highlight onto the original image
    final_image = img_cv.copy()
    blended = cv2.addWeighted(img_cv, 0.4, color_seg, 0.6, 0)
    final_image[final_mask] = blended[final_mask]
    
    cv2.imwrite(output_path, final_image)
    print(f"Success! Final image saved to: {output_path}")
    return True

if __name__ == "__main__":
    detect_all("static/uploads/3b66ef69_IMG_20260611_142807.jpg", "combined_output.jpg")
