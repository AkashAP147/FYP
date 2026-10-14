import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image

# Initialize the cloth segmentation session
session = new_session("u2net_cloth_seg")

# Load the image
img_path = r"d:\TRY_ON_MODEL\static\uploads\3b66ef69_IMG_20260611_142807.jpg"
img = Image.open(img_path)

# rembg with return_mask=True returns a 1-channel mask Image
mask = remove(img, session=session, return_mask=True)

# Save the mask to see what it looks like
mask.save("test_rembg_mask.jpg")
print("Done!")
