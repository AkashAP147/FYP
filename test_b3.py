from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b3_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b3_clothes")

image = Image.open('IMG_20260725_131929.jpg').convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits.cpu()
upsampled_logits = nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False,
)
pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
print("Detected:", np.unique(pred_seg))
