from ultralytics import YOLO
from huggingface_hub import hf_hub_download

shoe_model_path = hf_hub_download(repo_id="kesimeg/yolov8n-clothing-detection", filename="best.pt")
shoe_model = YOLO(shoe_model_path)

print("YOLO Classes:", shoe_model.names)

results = shoe_model.predict("IMG_20260409_104924.jpg", conf=0.25)
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        name = shoe_model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"Found: {name} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")
