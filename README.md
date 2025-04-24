# 🌊 UWFNet: Underwater Image Enhancement Network

**UWFNet** is a deep learning-based framework designed to enhance underwater images affected by color distortion, haze, low contrast, and poor lighting. It intelligently fuses details from both the foreground and background regions using an entropy-based image fusion technique, resulting in clearer, more vibrant underwater visuals without requiring any manual adjustments.

---

## 🔍 Overview

Underwater images often suffer from poor visibility due to absorption and scattering of light. **UWFNet** addresses this challenge by combining:

- Foreground-background separate enhancement
- Entropy-guided fusion  
- Color, contrast, and sharpness correction  
- White Balancing

---

## 📁 Project Structure

The project is organized into multiple folders:

- `datasets/` – contains the input and reference images  
- `results/` – holds the output enhanced images  
- `src/` – includes all source code such as the main pipeline, utility functions, and enhancement modules  

---

## 🧪 Datasets Used

This project uses three widely accepted underwater benchmark datasets for evaluation and training:

- **UCCS (300 images)**: Contains controlled underwater scenes with known distortions  
- **UIQS (3630 images)**: A diverse dataset labeled based on underwater image quality  
- **UIEB (890 images)**: Real-world underwater images with varied turbidity and lighting  

These datasets collectively cover a wide range of underwater scenarios, ensuring a robust evaluation of the model.

---

## 🧠 Methodology

The core methodology involves:

- White Balance Correction  
- Decomposition into Foreground and Backgroud
- Percentile Maximum Contrast Enhancement
- Background Dehazing using atmospheric light estimation 
- Foreground-Background Entropy Fusion  

**Entropy-based fusion** ensures that the most informative regions from the foreground and background are blended for the final enhanced output.

---

## 📊 Quantitative Metrics

UWFNet was evaluated using several quality metrics to validate its performance:

| **Metric**                | **Original Image**    | **UWFNet Enhanced**     | **Why It Matters**                          |
|--------------------------|-----------------------|--------------------------|----------------------------------------------|
| Color Saturation (Mean)  | 99.24                 | 112.16                   | Indicates richness of colors                |
| Contrast (Standard Dev)  | 31.79                 | 39.92                    | Reflects depth and separation in tones      |
| Entropy (Grayscale)      | 6.93                  | 7.30                     | Shows the amount of visual information      |
| White Balance (A, B)     | A=100.8, B=141.9      | A=115.0, B=136.4         | Measures color tone neutrality              |
| Edge Sharpness (Sobel)   | 9.25                  | 12.46                    | Evaluates clarity and definition of edges   |

---

## 📷 Visual Results

**UWFNet** outputs significantly improved images with higher clarity, natural lighting, and balanced color tones. Input vs enhanced comparisons are provided in the `results/` folder.

---

## 🎥 How to run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/uwfNet.git
   cd uwfNet
2. **Install the Requirements**
   ```bash
   pip install opencv-python numpy
5. **Run the pipeline**
   ```bash
   python src/main.py

