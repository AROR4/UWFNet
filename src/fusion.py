import cv2
import numpy as np

def compute_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    return entropy

def fuse_images(img1, img2):
    e1 = compute_entropy(img1)
    e2 = compute_entropy(img2)
    fused = (e1 * img1 + e2 * img2) / (e1 + e2)
    return fused.astype(np.uint8)
