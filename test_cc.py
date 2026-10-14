import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

image_path = "IMG_20260409_104924.jpg" 
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    
logits = outputs.logits.cpu()
upsampled_logits = nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

# Combine all clothing classes
clothing_classes = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19]
clothing_mask = np.isin(pred_seg, clothing_classes).astype(np.uint8)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clothing_mask, connectivity=8)

print(f"Found {num_labels - 1} clothing items.")

img_cv = cv2.imread(image_path)
h, w = img_cv.shape[:2]

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < (h * w * 0.005): # 0.5% threshold
        continue
        
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    bw = stats[i, cv2.CC_STAT_WIDTH]
    bh = stats[i, cv2.CC_STAT_HEIGHT]
    
    # What was the dominant class in this component?
    comp_mask = (labels == i)
    comp_classes = pred_seg[comp_mask]
    dominant_class = np.bincount(comp_classes).argmax()
    
    print(f"Item {i}: Area={area}, BBox=({x},{y},{bw},{bh}), Dominant Class={dominant_class}")
