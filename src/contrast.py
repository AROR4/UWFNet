import cv2
import numpy as np

def percentile_maximum_contrast_enhancement(image):
  
    enhanced_channels = []
    for channel in cv2.split(image):
        channel = channel.astype(np.float32)

        v1 = np.percentile(channel, 0.1)
        v2 = np.percentile(channel, 99.5)

     
        channel[channel < v1] = v1
        channel[channel > v2] = v2

        enhanced_channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
        enhanced_channels.append(enhanced_channel)

   
    enhanced_image = cv2.merge(enhanced_channels)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    return enhanced_image