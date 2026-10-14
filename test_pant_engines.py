import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from rembg import remove, new_session
from PIL import Image
import os

img_path = r"d:\TRY_ON_MODEL\static\uploads\6b82792f_IMG_20260611_114038.jpg" # Using a best guess of original image name
# If that specific file doesn't exist, try to find the newest image
if not os.path.exists(img_path):
    uploads = [f for f in os.listdir(r"d:\TRY_ON_MODEL\static\uploads") if "114038" in f]
    if uploads:
        img_path = os.path.join(r"d:\TRY_ON_MODEL\static\uploads", uploads[0])

if not os.path.exists(img_path):
    print("Could not find image containing 114038 in uploads")
    exit()

img_cv = cv2.imread(img_path)
height, width = img_cv.shape[:2]

# 1. Segformer
print("Running Segformer...")
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
image_pil = Image.open(img_path).convert("RGB")
inputs = processor(images=image_pil, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
upsampled_logits = torch.nn.functional.interpolate(outputs.logits.cpu(), size=(height, width), mode="bilinear", align_corners=False)
seg_map = upsampled_logits.argmax(dim=1)[0].numpy()
sf_mask = (seg_map == 9) | (seg_map == 12) # Pants/Skirt
sf_img = img_cv.copy()
color_seg = np.zeros_like(img_cv)
color_seg[sf_mask] = [255, 0, 255] # Magenta
sf_img[sf_mask] = cv2.addWeighted(img_cv, 0.4, color_seg, 0.6, 0)[sf_mask]
cv2.imwrite("test_pant_segformer.jpg", sf_img)

# 2. U2Net
print("Running U2Net Cloth Seg...")
session = new_session("u2net_cloth_seg")
mask_pil = remove(image_pil, session=session, only_mask=True)
mask_np = np.array(mask_pil)
u2net_mask = mask_np[height:2*height, :, 0] > 128
u2net_img = img_cv.copy()
color_seg = np.zeros_like(img_cv)
color_seg[u2net_mask] = [255, 0, 255] # Magenta
u2net_img[u2net_mask] = cv2.addWeighted(img_cv, 0.4, color_seg, 0.6, 0)[u2net_mask]
cv2.imwrite("test_pant_u2net.jpg", u2net_img)

# 3. DeepFashion2
print("Running DeepFashion2...")
seg_model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="deepfashion2_yolov8s-seg.pt")
seg_model = YOLO(seg_model_path)
seg_results = seg_model.predict(img_cv, conf=0.25, retina_masks=True)

df2_mask = np.zeros((height, width), dtype=bool)
if seg_results[0].masks is not None:
    masks = seg_results[0].masks.data.cpu().numpy()
    for i, box in enumerate(seg_results[0].boxes):
        class_name = seg_model.names[int(box.cls[0])].lower()
        if "trousers" in class_name or "short" in class_name or "skirt" in class_name:
            m = masks[i]
            if m.shape != (height, width):
                m = cv2.resize(m, (width, height))
            df2_mask[m > 0.5] = True
df2_img = img_cv.copy()
color_seg = np.zeros_like(img_cv)
color_seg[df2_mask] = [255, 0, 255] # Magenta
df2_img[df2_mask] = cv2.addWeighted(img_cv, 0.4, color_seg, 0.6, 0)[df2_mask]
cv2.imwrite("test_pant_df2.jpg", df2_img)

print("Finished testing pants!")
