"""
Author: Chi-An Chen
Date: 2025-10-27
Version: 2.0
"""
import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 與 CSV 欄名一致 (固定順序)
CLASSES = ["normal", "bacteria", "virus", "COVID-19"]


def build_tfms(train: bool = True, img_size: int = 256, center_crop_ratio: float = 0.95, 
               strong_aug: bool = False):
    """
    構建影像轉換管道
    
    參數:
        train: 是否為訓練模式
        img_size: 目標影像大小
        center_crop_ratio: 中心裁切比例（已棄用，保留相容性）
        strong_aug: 是否使用強增強（for mixup/cutmix）
    """
    base_mean = (0.48811911, 0.48812611, 0.48813389)
    base_std  = (0.24549533, 0.24549459, 0.24549438)

    if train:
        if strong_aug:
            # 強增強版本（適合搭配 Mixup/CutMix）
            tfms = A.Compose([
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_REFLECT),
                A.RandomCrop(height=img_size, width=img_size, p=1.0),

                A.Affine(
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    scale=(1-0.15, 1+0.15),
                    rotate=(-12, 12),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.9
                ),

                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                ], p=0.9),

                A.OneOf([
                    A.UnsharpMask(blur_limit=(3, 7), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.2),

                A.HorizontalFlip(p=0.15),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                               min_holes=2, min_height=8, min_width=8, 
                               fill_value=0, p=0.2),
                
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=base_mean, std=base_std),
                ToTensorV2(),
            ])
        else:
            # 標準訓練增強
            tfms = A.Compose([
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_REFLECT),
                A.RandomCrop(height=img_size, width=img_size, p=1.0),

                A.Affine(
                    translate_percent={'x': (-0.03, 0.03), 'y': (-0.03, 0.03)},
                    scale=(1-0.12, 1+0.12),
                    rotate=(-8, 8),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.8
                ),

                A.OneOf([
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.10, p=1.0),
                ], p=0.8),

                A.UnsharpMask(blur_limit=(3, 5), p=0.15),
                A.HorizontalFlip(p=0.10),
                A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=base_mean, std=base_std),
                ToTensorV2(),
            ])
    else:
        # 驗證/測試：確定性轉換
        tfms = A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),
            A.CenterCrop(height=img_size, width=img_size),  # 加入 CenterCrop 確保一致性
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=base_mean, std=base_std),
            ToTensorV2(),
        ])
    return tfms


def build_tta_tfms(img_size: int = 256, tta_level: str = 'light'):
    """
    為 TTA 構建多種增強策略
    
    參數:
        img_size: 目標影像大小
        tta_level: 'light', 'medium', 'heavy'
    
    返回:
        List[Compose]: 多個轉換管道
    """
    base_mean = (0.48811911, 0.48812611, 0.48813389)
    base_std  = (0.24549533, 0.24549459, 0.24549438)
    
    tta_transforms = []
    
    # 1. 原始（中心裁切）
    tta_transforms.append(A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),
        A.CenterCrop(height=img_size, width=img_size),
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=base_mean, std=base_std),
        ToTensorV2(),
    ]))
    
    # 2. 水平翻轉
    tta_transforms.append(A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),
        A.CenterCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=1.0),
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=base_mean, std=base_std),
        ToTensorV2(),
    ]))
    
    if tta_level in ['medium', 'heavy']:
        # 3-4. 輕微縮放
        for scale in [0.95, 1.05]:
            tta_transforms.append(A.Compose([
                A.LongestMaxSize(max_size=int(img_size * scale), interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),
                A.CenterCrop(height=img_size, width=img_size),
                A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=base_mean, std=base_std),
                ToTensorV2(),
            ]))
    
    if tta_level == 'heavy':
        # 5-6. 輕微旋轉
        for angle in [-5, 5]:
            tta_transforms.append(A.Compose([
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),
                A.Rotate(limit=(angle, angle), border_mode=cv2.BORDER_REFLECT, p=1.0),
                A.CenterCrop(height=img_size, width=img_size),
                A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=base_mean, std=base_std),
                ToTensorV2(),
            ]))
        
        # 7. 輕微亮度調整
        tta_transforms.append(A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT),
            A.CenterCrop(height=img_size, width=img_size),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
            A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=base_mean, std=base_std),
            ToTensorV2(),
        ]))
    
    return tta_transforms


class CXRCSV(Dataset):
    """
    支援：
    - 檔名欄位：優先使用 `new_filename`，若沒有可擴充到 filename/image/img/path。
    - 標籤：優先讀取單一整數欄位 `label`；否則讀取與 CLASSES 對應的 one-hot 欄位並轉為索引。
    - 測試集（has_labels=False）回傳 (tensor, filename)
    - TTA 支援：可指定使用預定義的 TTA transforms
    """

    def __init__(self, csv_path, img_dir, is_train=True, has_labels=True, 
                 img_size=256, strong_aug=False, tta_transforms=None):
        """
        參數:
            csv_path: CSV 檔案路徑
            img_dir: 影像資料夾路徑
            is_train: 是否為訓練模式
            has_labels: 是否有標籤
            img_size: 影像大小
            strong_aug: 是否使用強增強
            tta_transforms: 用於 TTA 的轉換列表（若提供則覆蓋預設轉換）
        """
        self.csv_path = csv_path
        self.img_dir = Path(img_dir)
        self.is_train = is_train
        self.has_labels = has_labels
        self.img_size = img_size
        self.tta_transforms = tta_transforms
        self.tta_index = 0  # 用於循環不同的 TTA 轉換

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # ---- 檔名欄偵測 ----
        fname_col = None
        for cand in ["new_filename", "filename", "image", "img", "path"]:
            if cand in df.columns:
                fname_col = cand
                break
        if fname_col is None:
            raise KeyError(
                f"Cannot find filename column in {csv_path}; "
                f"expected one of new_filename/filename/image/img/path, got: {list(df.columns)}"
            )
        self.fnames = df[fname_col].astype(str).tolist()

        # ---- 標籤處理 ----
        self.labels = None
        if has_labels:
            if "label" in df.columns:
                # 單欄整數標籤
                self.labels = df["label"].astype(np.int64).to_numpy()
            else:
                # one-hot（與 CLASSES 對應）
                missing = [c for c in CLASSES if c not in df.columns]
                if missing:
                    raise KeyError(f"Missing class columns in CSV: {missing}; need {CLASSES}")
                onehot = df[CLASSES].to_numpy(dtype=int)
                s = onehot.sum(axis=1)
                bad = np.where(s != 1)[0]
                if bad.size:
                    raise ValueError(f"{len(bad)} rows are not one-hot (row sums != 1).")
                self.labels = onehot.argmax(axis=1).astype(np.int64)

        # 舊名稱相容（有些程式會取用 targets）
        self.targets = self.labels

        # ---- 影像轉換 ----
        if tta_transforms is not None:
            # 使用提供的 TTA transforms
            self.tfms = None  # 將在 __getitem__ 中動態選擇
            self.base_tfms = build_tfms(train=False, img_size=img_size)
        else:
            self.tfms = build_tfms(train=is_train, img_size=img_size, strong_aug=strong_aug)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fp = self.img_dir / self.fnames[idx]
        img = cv2.imread(str(fp))
        if img is None:
            raise FileNotFoundError(f"Image not found: {fp}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 選擇轉換
        if self.tta_transforms is not None:
            # TTA 模式：循環使用不同的轉換
            tfm = self.tta_transforms[self.tta_index % len(self.tta_transforms)]
            img = tfm(image=img)["image"]
            self.tta_index = (self.tta_index + 1) % len(self.tta_transforms)
        elif self.tfms is not None:
            img = self.tfms(image=img)["image"]

        if self.labels is not None:
            y = int(self.labels[idx])
            return img, y
        else:
            return img, self.fnames[idx]
    
    def set_tta_index(self, index):
        """設定 TTA 轉換索引（用於確定性 TTA）"""
        self.tta_index = index

def get_class_weights(labels, method='sqrt_inv', scale_factor=None):
    """
    計算類別權重
    
    參數:
        labels: 標籤陣列
        method: 'inv' (反比), 'sqrt_inv' (平方根反比), 'cb' (class-balanced)
        scale_factor: 額外的縮放因子字典 {class_name: factor}
    
    返回:
        numpy array of weights
    """
    cls_cnt = np.bincount(labels, minlength=len(CLASSES))
    
    if method == 'inv':
        weights = 1.0 / (cls_cnt + 1e-6)
    elif method == 'sqrt_inv':
        weights = 1.0 / np.sqrt(cls_cnt + 1e-6)
    elif method == 'cb':
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_cnt)
        weights = (1.0 - beta) / np.array(effective_num + 1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 套用額外縮放
    if scale_factor is not None:
        for cls_name, factor in scale_factor.items():
            idx = CLASSES.index(cls_name)
            weights[idx] *= factor
    
    # 正規化
    weights = weights / (weights.mean() + 1e-6)
    
    return weights

def get_weighted_sampler(labels, method='sqrt_inv', scale_factor=None):
    """
    建立 WeightedRandomSampler 來平衡類別
    
    參數:
        labels: 標籤陣列
        method: 權重計算方法 ('inv', 'sqrt_inv', 'cb')
        scale_factor: 額外的縮放因子字典 {class_name: factor}
    
    返回:
        WeightedRandomSampler
    """
    # 計算類別權重
    class_weights = get_class_weights(labels, method=method, scale_factor=scale_factor)
    
    # 為每個樣本分配權重
    sample_weights = class_weights[labels]
    
    # 建立 sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # 允許重複抽樣
    )
    
    return sampler


def get_loaders(csv_train, dir_train, csv_val, dir_val, bs: int = 32, 
                img_size: int = 256, strong_aug: bool = False, num_workers: int = None,
                use_sampler: bool = True, sampler_method: str = 'sqrt_inv',
                sampler_scale_factor: dict = None):
    """
    建立訓練和驗證的 DataLoader
    
    參數:
        csv_train: 訓練集 CSV 路徑
        dir_train: 訓練集影像資料夾
        csv_val: 驗證集 CSV 路徑
        dir_val: 驗證集影像資料夾
        bs: batch size
        img_size: 影像大小
        strong_aug: 是否使用強增強
        num_workers: worker 數量（None=自動）
        use_sampler: 是否使用 WeightedRandomSampler
        sampler_method: sampler 權重計算方法
        sampler_scale_factor: sampler 額外縮放因子
    
    返回:
        (train_loader, val_loader, train_labels)
    """
    ds_tr = CXRCSV(csv_train, dir_train, is_train=True, has_labels=True, 
                   img_size=img_size, strong_aug=strong_aug)
    ds_va = CXRCSV(csv_val, dir_val, is_train=False, has_labels=True, img_size=img_size)

    if num_workers is None:
        num_workers = max(1, os.cpu_count() // 2)

    # 準備 sampler（如果需要）
    sampler = None
    shuffle = True
    if use_sampler:
        sampler = get_weighted_sampler(
            ds_tr.labels, 
            method=sampler_method,
            scale_factor=sampler_scale_factor
        )
        shuffle = False  # 使用 sampler 時不能同時 shuffle

    tr = DataLoader(
        ds_tr, batch_size=bs, 
        shuffle=shuffle,
        sampler=sampler,  # 加入 sampler
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=(num_workers > 0),
        drop_last=True
    )
    va = DataLoader(
        ds_va, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True, 
        persistent_workers=(num_workers > 0)
    )

    return tr, va, ds_tr.labels

# ============ 使用範例 ============
if __name__ == "__main__":
    # 測試資料載入
    tr, va, labels = get_loaders(
        "./datasets/train_data.csv",
        "./datasets/train_images",
        "./datasets/val_data.csv",
        "./datasets/val_images",
        bs=32,
        img_size=384
    )
    
    print(f"Train batches: {len(tr)}")
    print(f"Val batches: {len(va)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # 測試 TTA transforms
    tta_tfms = build_tta_tfms(img_size=384, tta_level='medium')
    print(f"TTA transforms: {len(tta_tfms)}")
    
    # 計算類別權重
    weights = get_class_weights(labels, method='sqrt_inv', 
                               scale_factor={'virus': 1.5, 'COVID-19': 1.2})
    print(f"Class weights: {weights}")



