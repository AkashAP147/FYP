import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

print("Downloading YOLOv8s DeepFashion2 Segmentation weights...")
model_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="deepfashion2_yolov8s-seg.pt")

print("Loading model...")
model = YOLO(model_path)

print("Running segmentation...")
results = model.predict('IMG_20260725_131929.jpg', conf=0.25)

# We want only the masks, no bounding boxes
annotated = results[0].plot(boxes=False, labels=False, line_width=2)
cv2.imwrite("test_seg.jpg", annotated)
print("Done!")
