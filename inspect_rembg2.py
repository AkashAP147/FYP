import cv2
import numpy as np

r = cv2.imread("test_rembg_r.png", cv2.IMREAD_GRAYSCALE)
g = cv2.imread("test_rembg_g.png", cv2.IMREAD_GRAYSCALE)
b = cv2.imread("test_rembg_b.png", cv2.IMREAD_GRAYSCALE)

print("Red channel unique:", np.unique(r))
print("Green channel unique:", np.unique(g))
print("Blue channel unique:", np.unique(b))
