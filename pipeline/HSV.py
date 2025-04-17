import cv2
import numpy as np

def preprocess_images(images, V_FIXED = 200):
    fixed_images = []
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_fixed = hsv.copy()
        hsv_fixed[:, :, 2] = (hsv[:, :, 2] / hsv[:, :, 2].max()) * V_FIXED
        hsv_fixed[:, :, 1] = hsv_fixed[:, :, 1] * (hsv_fixed[:, :, 2] / hsv[:, :, 2].max())
        hsv_fixed[:, :, 1] = np.clip(hsv_fixed[:, :, 1], 0, 255)

        fixed_image = cv2.cvtColor(hsv_fixed, cv2.COLOR_HSV2BGR)
        fixed_images.append(fixed_image)
    return fixed_images
