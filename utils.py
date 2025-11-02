"""
Author: Chi-An Chen
Date: 2025-10-27
Version: 2.0
"""
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def make_scheduler(optimizer, args):
    if args.sched == "cosine":
        if args.warmup_epochs > 0:
            warmup = LinearLR(optimizer,
                              start_factor=args.warmup_start,
                              end_factor=1.0,
                              total_iters=args.warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs),
                eta_min=args.min_lr
            )
            return warmup, cosine
        else:
            return None, CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    mode = "max" if args.monitor == "val_f1" else "min"
    main = ReduceLROnPlateau(
        optimizer, mode=mode,
        factor=args.plateau_factor, patience=args.plateau_patience,
        min_lr=args.min_lr
    )
    if args.warmup_epochs > 0:
        warmup = LinearLR(optimizer,
                          start_factor=args.warmup_start,
                          end_factor=1.0,
                          total_iters=args.warmup_epochs)
        return warmup, main
    else:
        return None, main
    
def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============ 統一處理模型輸出的輔助函數 ============
def extract_logits(output):
    """
    統一提取模型的主要輸出 logits
    處理 Inception V3 的特殊輸出格式
    
    Returns:
        main_logits: 主要分類輸出
        aux_logits: 輔助分類輸出 (僅 Inception V3 訓練時有)
    """
    if isinstance(output, tuple):
        # Inception V3 在某些版本可能返回 tuple
        return output[0], output[1] if len(output) > 1 else None
    elif hasattr(output, 'logits'):
        # Inception V3 標準輸出: InceptionOutputs
        main_logits = output.logits
        aux_logits = output.aux_logits if hasattr(output, 'aux_logits') else None
        return main_logits, aux_logits
    else:
        # 其他模型直接返回 tensor
        return output, None
    
def save_confusion_matrix(figpath, y_true, y_pred, labels):
    title = os.path.splitext(os.path.basename(figpath))[0]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=150)
    plt.close(fig)

import matplotlib.pyplot as plt
import os

def plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs, name="experiment"):
    """
    繪製並儲存 Loss、F1-score、Accuracy 曲線圖
    Args:
        train_losses, val_losses: list, 每個 epoch 的訓練與驗證 loss
        train_f1s, val_f1s: list, 每個 epoch 的訓練與驗證 F1-score
        train_accs, val_accs: list, 每個 epoch 的訓練與驗證 accuracy
        name: str, 結果資料夾名稱 (預設為 'experiment')
    """
    os.makedirs(f"./results/{name}", exist_ok=True)

    # --- Loss Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/{name}/loss_curve.png", dpi=150)
    plt.close()
    print("✅ Saved loss curve → loss_curve.png")

    # --- F1-score Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_f1s, label="Train F1", marker="o")
    plt.plot(val_f1s, label="Val F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1-score")
    plt.title("F1-score Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/{name}/f1_curve.png", dpi=150)
    plt.close()
    print("✅ Saved F1-score curve → f1_curve.png")

    # --- Accuracy Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_accs, label="Train Acc", marker="o")
    plt.plot(val_accs, label="Val Acc", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/{name}/acc_curve.png", dpi=150)
    plt.close()
    print("✅ Saved Accuracy curve → acc_curve.png")

# ✅ 使用範例
# plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs, name="run_01")