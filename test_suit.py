from ultralytics import YOLO
import cv2

try:
    model = YOLO("yolov8n-pose.pt")
    res = model("suit-1.webp")
    if res[0].keypoints is not None:
        print("Keypoints detected:", res[0].keypoints.shape)
    else:
        print("No keypoints detected.")
except Exception as e:
    print(f"Error: {e}")
