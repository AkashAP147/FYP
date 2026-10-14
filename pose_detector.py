import cv2
from ultralytics import YOLO
import os

def detect_pose(image_path, output_path):
    if not os.path.exists(image_path):
        return False

    # Load the YOLOv8 Nano Pose model (Extremely lightweight, ~6MB)
    # YOLO is fundamentally an object detector, so it handles varying sizes and scales 
    # much better than MediaPipe for static images, leading to higher accuracy.
    model = YOLO('yolov8n-pose.pt') 

    # Run inference on the image
    results = model(image_path)
    
    # Plot the results (draws the skeleton automatically)
    # line_width=4 ensures the lines are thick and highlighted
    # labels=False hides the "person" text box so we only see the skeleton
    annotated_image = results[0].plot(line_width=4, labels=False) 
    
    # Save the result
    cv2.imwrite(output_path, annotated_image)
    return True

if __name__ == "__main__":
    input_image = "IMG_20260409_104924.jpg"
    output_image = "pose_output.jpg"
    detect_pose(input_image, output_image)
