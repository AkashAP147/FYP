import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

print("Loading SegFormer for Garment Extraction...")
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

unique, counts = np.unique(pred_seg, return_counts=True)
total_pixels = pred_seg.shape[0] * pred_seg.shape[1]

labels = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Glove", 4: "Sunglasses", 
    5: "Upper-clothes (Shirt/T-shirt)", 6: "Dress", 7: "Coat", 8: "Socks", 
    9: "Pants/Jeans", 10: "Jumpsuits", 11: "Scarf", 12: "Skirt", 13: "Face", 
    14: "Left-arm", 15: "Right-arm", 16: "Left-leg", 17: "Right-leg", 
    18: "Left-shoe", 19: "Right-shoe"
}

print(f"Total pixels: {total_pixels}")
print("Detected classes:")
for u, c in zip(unique, counts):
    label = labels.get(u, f"Unknown ({u})")
    percentage = (c / total_pixels) * 100
    print(f"Class {u} ({label}): {c} pixels ({percentage:.2f}%)")
