import os
from garment_extractor import extract_garments

# Use an existing image from the directory
image_path = "IMG_20260409_104924.jpg" 
output_dir = "static/outputs/test_extract"
os.makedirs(output_dir, exist_ok=True)

try:
    print(f"Extracting garments from {image_path}...")
    items = extract_garments(image_path, output_dir)
    print("Success! Items extracted:")
    for item in items:
        print(item)
except Exception as e:
    print("Error:", str(e))
