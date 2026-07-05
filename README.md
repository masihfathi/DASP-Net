# DASP-Net

**DASP-Net** stands for **Degradation-Aware Spatial-Frequency Prompting Network**.

This repository contains the implementation of a lightweight image enhancement framework for **low-light image enhancement**. The main idea is to improve a simple U-Net by adding several guidance maps extracted from the input image itself.

Instead of using only the RGB image as input, DASP-Net uses:

- RGB image
- Illumination map
- Edge map
- Frequency/detail map
- Noise estimation map

Therefore, the proposed model receives a **7-channel input** and predicts a 3-channel enhanced RGB image.

---

## Project Goal

Low-light images often suffer from poor visibility, low contrast, noise amplification, color distortion, and loss of fine details. Many deep learning methods improve image quality but require complex architectures or high computational cost.

This project aims to build a simple and lightweight method that improves low-light images by using additional spatial and frequency-aware prompt maps while keeping the model architecture close to a standard U-Net.

---

## Method Overview

Given a low-light RGB image, DASP-Net first extracts four guidance maps:

1. **Illumination Map**  
   Captures the brightness distribution of the image.

2. **Edge Map**  
   Extracted using the Sobel operator to preserve object boundaries.

3. **Frequency/Detail Map**  
   Extracted using the Laplacian operator to highlight high-frequency details.

4. **Noise Estimation Map**  
   Computed using the residual between the grayscale image and its Gaussian-smoothed version.

The final input is:

```text
RGB + Illumination + Edge + Frequency + Noise = 7 channels