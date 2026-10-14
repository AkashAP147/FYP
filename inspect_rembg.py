import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image

print("Loading u2net_cloth_seg session...")
session = new_session("u2net_cloth_seg")

img_path = "static/uploads/3b66ef69_IMG_20260611_142807.jpg"
print(f"Loading image {img_path}")
img = Image.open(img_path)

print("Running rembg...")
# We use only_mask=True which is the correct parameter in newer rembg versions
mask = remove(img, session=session, only_mask=True)

mask_np = np.array(mask)
print("Mask shape:", mask_np.shape)
print("Unique values in mask:", np.unique(mask_np))

# If it's a grayscale mask with specific values, let's save distinct layers
if len(mask_np.shape) == 2 or mask_np.shape[2] == 1:
    print("Grayscale mask detected.")
    cv2.imwrite("test_rembg_raw.png", mask_np)
elif len(mask_np.shape) == 3:
    print("RGB mask detected.")
    cv2.imwrite("test_rembg_r.png", mask_np[:, :, 0])
    cv2.imwrite("test_rembg_g.png", mask_np[:, :, 1])
    cv2.imwrite("test_rembg_b.png", mask_np[:, :, 2])

print("Done!")
