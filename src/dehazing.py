import cv2
import numpy as np

def estimate_atmospheric_light(image):
    """
    Estimate the atmospheric light (A) using regional analysis.
    
    Parameters:
        image (numpy.ndarray): Input background image.
    
    Returns:
        numpy.ndarray: Estimated atmospheric light value (3-element array for RGB).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    half_h, half_w = h // 2, w // 2
    
    quadrants = {
        "ITL": image[:half_h, :half_w],       # Top-left
        "ITR": image[:half_h, half_w:],      # Top-right
        "IBL": image[half_h:, :half_w],     # Bottom-left
        "IBR": image[half_h:, half_w:]      # Bottom-right
    }
    
    scores = {}
    for name, quadrant in quadrants.items():
        # Compute mean and standard deviation for each channel
        means = [np.mean(quadrant[:, :, c]) for c in range(3)]  # R, G, B channels
        stds = [np.std(quadrant[:, :, c]) for c in range(3)]    # R, G, B channels
        
        # Calculate the score for the region
        score = sum(means[c] - stds[c] for c in range(3))
        scores[name] = score
    
    # Select the quadrant with the highest score
    best_quadrant_name = max(scores, key=scores.get)
    best_quadrant = quadrants[best_quadrant_name]
    
    # Find the pixel closest to white (255, 255, 255) in the selected quadrant
    min_distance = float('inf')
    atmospheric_light = None
    
    for i in range(best_quadrant.shape[0]):
        for j in range(best_quadrant.shape[1]):
            pixel = best_quadrant[i, j]
            distance = np.linalg.norm(pixel - np.array([255, 255, 255]))
            if distance < min_distance:
                min_distance = distance
                atmospheric_light = pixel
    
    return atmospheric_light

def estimate_transmission_map(image, patch_size=15):
    """
    Estimate the transmission map using block-based minima.
    
    Parameters:
        image (numpy.ndarray): Input background image.
        patch_size (int): Size of the block for transmission map estimation.
    
    Returns:
        numpy.ndarray: Estimated transmission map.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    t_map = np.zeros((h, w), dtype=np.float32)

    # Segment the image into blocks of size `patch_size`
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            block = gray[i:i+patch_size, j:j+patch_size]
            t_map[i:i+patch_size, j:j+patch_size] = 1 - np.min(block) / np.max(gray)

    return t_map

def optimize_transmission_map(image, t_map, A, lambda_=10, r=16, eps=1e-3):
    """
    Optimize the transmission map using guided filtering.
    
    Parameters:
        image (numpy.ndarray): Input background image.
        t_map (numpy.ndarray): Initial transmission map.
        A (float): Estimated atmospheric light.
        lambda_ (float): Regularization parameter.
        r (int): Radius for guided filtering.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        numpy.ndarray: Optimized transmission map.
    """
    refined_t_map = guided_filter(image, t_map, radius=r, epsilon=eps)
    return refined_t_map

def guided_filter(image, p, radius=5, epsilon=1e-3):
    """
    Applies a guided filter to refine the input image.
    
    Parameters:
        image (numpy.ndarray): The guidance image (grayscale or color).
        p (numpy.ndarray): The input image to be filtered (grayscale).
        radius (int): Radius of the guided filter window.
        epsilon (float): Regularization parameter to avoid division by zero.
    
    Returns:
        numpy.ndarray: The filtered image.
    """
    # Convert inputs to float32
    image = image.astype(np.float32)
    p = p.astype(np.float32)

    # Convert image to grayscale if it's a 3-channel image
    if len(image.shape) == 3:  # Check if image is RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute mean of image and p within the window
    mean_image = cv2.boxFilter(image, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))

    # Compute correlation terms
    mean_ip = cv2.boxFilter(image * p, -1, (radius, radius))
    corr_ip = mean_ip - mean_image * mean_p

    # Compute variance of the image
    mean_ii = cv2.boxFilter(image * image, -1, (radius, radius))
    var_image = mean_ii - mean_image * mean_image

    # Compute the coefficients a and b
    a = corr_ip / (var_image + epsilon)
    b = mean_p - a * mean_image

    # Compute mean of a and b
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # Compute the output
    q = mean_a * image + mean_b
    return q


# Step 6: Multilayer Transmission Map Estimated Dehazing
def multilayer_transmission_map_dehazing(image, A, t_map):
    """
    Perform dehazing using the atmospheric scattering model.
    
    Parameters:
        image (numpy.ndarray): Input background image.
        A (numpy.ndarray): Estimated atmospheric light (3-element array for RGB).
        t_map (numpy.ndarray): Estimated transmission map.
    
    Returns:
        numpy.ndarray: Dehazed image.
    """
    J = np.zeros_like(image, dtype=np.float32)
    for c in range(3):  # RGB channels
        J[:, :, c] = (image[:, :, c] - A[c] * (1 - t_map)) / t_map
        J[:, :, c][t_map == 0] = 0  # Avoid division by zero
    
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J
