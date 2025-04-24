import cv2
import numpy as np
from whitebal import white_balance
from decomposition import image_decomposition
from contrast import percentile_maximum_contrast_enhancement
from dehazing import multilayer_transmission_map_dehazing, optimize_transmission_map ,estimate_transmission_map ,estimate_atmospheric_light
from fusion import fuse_images
from hsi import enhance_image_colors
from skimage.filters import sobel
from skimage import color

import cv2
import numpy as np

def edge_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(magnitude)

def average_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gy, gx = np.gradient(gray.astype(np.float64))
    ag = np.mean(np.sqrt(gx**2 + gy**2))
    return ag 

def information_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    return entropy  

def colorfulness_contrast_fog_density(image):
    R, G, B = image[:,:,2], image[:,:,1], image[:,:,0]
    rg = R - G
    yb = 0.5 * (R + G) - B

    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    contrast = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    ccf = (colorfulness * 0.6 + contrast * 0.4)
    return ccf 

def evaluate_image(image):
    ei = edge_intensity(image)
    ag = average_gradient(image)
    ie = information_entropy(image)
    ccf = colorfulness_contrast_fog_density(image)

    print("===== Image Evaluation Metrics =====")
    print(f"Edge Intensity (EI):                 {ei:.2f} ")
    print(f"Average Gradient (AG):              {ag:.2f} ")
    print(f"Information Entropy (IE):           {ie:.2f} ")
    print(f"Colorfulness Contrast Fog Index:    {ccf:.2f} ")

def compute_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-6))  # Avoid log(0)
    return entropy

def compute_saturation_mean(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return np.mean(saturation)

def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def compute_white_balance_channels(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)
    return np.mean(a), np.mean(b)

def compute_sobel_edge_strength(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edge_strength = sobel(gray)
    return np.mean(edge_strength) * 255  

def compare_images(original, enhanced):
    print("===== Image Comparison Metrics =====")
    print(f"Color Saturation Mean:\t\tOriginal = {compute_saturation_mean(original):.2f}, Enhanced = {compute_saturation_mean(enhanced):.2f}")
    print(f"Contrast (Std Dev):\t\tOriginal = {compute_contrast(original):.2f}, Enhanced = {compute_contrast(enhanced):.2f}")
    print(f"Entropy:\t\t\tOriginal = {compute_entropy(original):.2f}, Enhanced = {compute_entropy(enhanced):.2f}")
    
    a1, b1 = compute_white_balance_channels(original)
    a2, b2 = compute_white_balance_channels(enhanced)
    print(f"White Balance (A, B):\t\tOriginal = (A={a1:.1f}, B={b1:.1f}), Enhanced = (A={a2:.1f}, B={b2:.1f})")

    print(f"Edge Sharpness (Sobel):\t\tOriginal = {compute_sobel_edge_strength(original):.2f}, Enhanced = {compute_sobel_edge_strength(enhanced):.2f}")

def pcfb_pipeline(input_image_path, output_image_path):
    input_image = cv2.imread(input_image_path)

    balanced =white_balance(input_image)
    cv2.imshow('balanced', balanced)

    foreground, background = image_decomposition(balanced)
    cv2.imshow('foreground', foreground)
    cv2.imshow('background', background)


    enhanced_foreground = percentile_maximum_contrast_enhancement(foreground)
    cv2.imshow('enhanced_foreground', enhanced_foreground)

    A = estimate_atmospheric_light(background)
    t_map = estimate_transmission_map(background)
    refined_t_map = optimize_transmission_map(background, t_map, A)
    dehazed_background = multilayer_transmission_map_dehazing(background,A, refined_t_map)
    cv2.imshow('dehazed_background', dehazed_background)

    fused_image=fuse_images(enhanced_foreground,dehazed_background)

    final_image=enhance_image_colors(fused_image)

    cv2.imshow('fused_image', fused_image)
    cv2.imshow('final_image', final_image)

   
    evaluate_image(final_image)
    compare_images(input_image, final_image)

    success =cv2.imwrite(output_image_path, final_image)
    if not success:
        print("Failed to save the image.")
    cv2.waitKey(30000)
    cv2.destroyAllWindows()
    print(f"Enhanced image saved to {output_image_path}")

if __name__ == "__main__":
    input_image_path = "../datasets/448.jpg" 
    output_image_path = "../results/enhanced_image.jpg" 
    pcfb_pipeline(input_image_path, output_image_path)