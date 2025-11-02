"""
Author: Chi-An Chen
Date: 2025-10-27
Version: 1.0
"""

import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def calculate_mean_std(data_path):
    """計算資料集的mean和std"""
    images = list(Path(data_path).rglob("*.jpeg")) + list(Path(data_path).rglob("*.jpg"))
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0

    # tqdm 加入進度條顯示
    for img_path in tqdm(images, desc="Processing images", unit="img"):
        img = np.array(Image.open(img_path).convert('RGB')) / 255.0
        pixel_sum += img.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (img ** 2).reshape(-1, 3).sum(axis=0)
        total_pixels += img.shape[0] * img.shape[1]
    
    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)
    
    print(f"\nMean: {mean}")
    print(f"Std: {std}")
    return mean, std

if __name__ == "__main__":
    mean, std = calculate_mean_std("data/train_organized")
    print("Calculated Mean:", mean)
    print("Calculated Std:", std)