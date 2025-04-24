import cv2
import numpy as np

def enhance_image_colors(image, saturation_scale=1.5):
    
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    h, s, v = cv2.split(img_hsv)


    black_mask = v < 30
    white_gray_mask = (v > 200) & (s < 20)
    exclude_mask = black_mask | white_gray_mask

    s = s * saturation_scale
    s = np.clip(s, 0, 255)

    original_s = img_hsv[..., 1]
    s[exclude_mask] = original_s[exclude_mask]

   
    enhanced_hsv = cv2.merge([h, s, v]).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return enhanced_bgr
