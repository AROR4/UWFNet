import cv2
import numpy as np

def image_decomposition(image, alpha=0.5):
    """
    Decompose the image into foreground and background components.
    """
    # Normalize image to [0, 1]
    normalized_image = image.astype(np.float32) / 255.0

    # Compute separation critical value k
    max_intensity = np.max(normalized_image)
    k = alpha * normalized_image / max_intensity

    # Compute foreground and background
    foreground = (1 - k) * normalized_image
    background = k * normalized_image

    # Scale back to [0, 255]
    foreground = np.clip(foreground * 255, 0, 255).astype(np.uint8)
    background = np.clip(background * 255, 0, 255).astype(np.uint8)

    return foreground, background