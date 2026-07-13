"""
Evaluation Metrics: DSC & SSIM
================================
Sesuai paper bagian 3.3:
- DSC (Dice Similarity Coefficient): mengukur kesamaan hasil segmentasi
  biner terhadap ground truth.
- SSIM (Structural Similarity Index Measurement): mengukur kesamaan
  struktural citra grayscale.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim_func


def dice_similarity_coefficient(y_true: np.ndarray, y_pred: np.ndarray, smooth: float = 1e-6) -> float:
    y_true = (y_true > 127).astype(np.uint8).flatten()
    y_pred = (y_pred > 127).astype(np.uint8).flatten()
    intersection = np.sum(y_true * y_pred)
    return float((2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth))


def ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        import cv2
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score, _ = ssim_func(img1, img2, full=True)
    return float(score)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64) / 255.0
    y_pred = y_pred.astype(np.float64) / 255.0
    return float(np.mean((y_true - y_pred) ** 2))
