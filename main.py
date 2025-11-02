"""
Author: Chi-An Chen
Date: 2025-10-27
Version: 2.0
"""
import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

from model import ModelEMA, make_model
from loss import FocalLoss, CBFocalLoss
from dataloader import CXRCSV, CLASSES, get_loaders, build_tta_tfms
from utils import make_scheduler, get_lr, seed_everything, extract_logits, save_confusion_matrix, plot_training_curves

torch.backends.cudnn.benchmark = True

def train_one(csv_train, dir_train, csv_val, dir_val, args):
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    dir_name = time.strftime('%Y-%m-%d', time.localtime())   
    tr, va, targets = get_loaders(csv_train, dir_train, csv_val, dir_val, bs=args.bs, img_size=args.img_size, use_sampler=True)
    
    try:
        dir_path = f"./results/{dir_name}"
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        print(f"建立目錄錯誤: {e}")

    model = make_model(num_classes=len(CLASSES), dropout=args.dropout, backbone=args.backbone).to(device)
    
    # ============ 全參數統一學習率訓練 ============
    if args.backbone_lr_ratio is not None and args.backbone_lr_ratio < 1.0:
        # 使用差異化學習率
        if args.backbone.startswith("resnet") or args.backbone == "inception_v3":
            classifier_name = 'fc'
        elif args.backbone.startswith("densenet"):
            classifier_name = 'classifier'
        elif args.backbone.startswith("efficientnet") or args.backbone.startswith("convnext"):
            classifier_name = 'classifier'
        else:
            classifier_name = 'fc'
        
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if classifier_name in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        opt = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_ratio, 'weight_decay': args.weight_decay},
            {'params': classifier_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        ])
        print(f"Using differentiated LR: backbone={args.lr * args.backbone_lr_ratio:.2e}, classifier={args.lr:.2e}")
    else:
        # 全參數統一學習率
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using unified LR: {args.lr:.2e} for all parameters")
    
    warmup_sched, main_sched = make_scheduler(opt, args)
    
    steps_per_epoch = len(tr)
    ema = ModelEMA(model, half_life_steps=steps_per_epoch * 2)
    
    # ============ 選擇 Loss Function ============
    cls_cnt = np.bincount(targets, minlength=len(CLASSES))
    
    if args.loss == "ce":
        # Cross Entropy Loss (原始版本)
        freq = cls_cnt / (cls_cnt.sum() + 1e-6)
        w = (1.0 / np.sqrt(freq + 1e-6))
        scale = np.ones_like(w)
        scale[CLASSES.index("virus")] *= 1.35
        w = w * scale
        w = w / (w.mean() + 1e-6)
        
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device), 
            label_smoothing=args.label_smoothing
        )
        print(f"Using Cross Entropy Loss with label_smoothing={args.label_smoothing}")
        
    elif args.loss == "focal":
        # Focal Loss: 更關注困難樣本，對 F1 有幫助
        freq = cls_cnt / (cls_cnt.sum() + 1e-6)
        w = (1.0 / np.sqrt(freq + 1e-6))
        scale = np.ones_like(w)
        scale[CLASSES.index("virus")] *= 1.5  # Focal Loss 可以更激進
        w = w * scale
        w = w / (w.mean() + 1e-6)
        
        crit = FocalLoss(
            alpha=torch.tensor(w, dtype=torch.float32, device=device),
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        print(f"Using Focal Loss with gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}")
        
    elif args.loss == "cb_focal":
        # Class-Balanced Focal Loss: 自動計算類別權重
        crit = CBFocalLoss(
            samples_per_class=cls_cnt,
            beta=args.cb_beta,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        print(f"Using Class-Balanced Focal Loss with beta={args.cb_beta}, gamma={args.focal_gamma}")

    elif args.loss == "combined":
        # Combined Loss: CE主導 + CB Focal輔助
        # CE: 保持整體穩定性
        freq = cls_cnt / (cls_cnt.sum() + 1e-6)
        w = (1.0 / np.sqrt(freq + 1e-6))
        scale = np.ones_like(w)
        scale[CLASSES.index("virus")] *= 1.35
        w = w * scale
        w = w / (w.mean() + 1e-6)
        
        ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device),
            label_smoothing=args.label_smoothing
        )
        
        # CB Focal: 針對少數類別
        cb_focal_loss = CBFocalLoss(
            samples_per_class=cls_cnt,
            beta=args.cb_beta,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        
        # 權重建議: CE為主(0.7-0.8), CB Focal為輔(0.2-0.3)
        ce_weight = 0.8
        cb_weight = 0.2
        
        def combined_loss(logits, targets):
            loss_ce = ce_loss(logits, targets)
            loss_cb = cb_focal_loss(logits, targets)
            return ce_weight * loss_ce + cb_weight * loss_cb
        
        crit = combined_loss
        print(f"Using Combined Loss: {ce_weight}*CE + {cb_weight}*CB_Focal")
    
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())  

    best_f1, best_path = -1, f"./results/{dir_name}/best_{getattr(args, 'backbone', 'model')}.pt"
    train_losses, val_losses, train_f1s, val_f1s = [], [], [], []
    train_accs, val_accs = [], []
    patience_left = args.es_patience

    autocast = torch.amp.autocast

    for ep in range(1, args.epochs + 1):
        # ===== Train =====
        model.train()
        running_loss = 0.0
        progress = tqdm(tr, desc=f"Epoch {ep}/{args.epochs}", ncols=200, leave=True)

        for x, y in progress:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=torch.cuda.is_available()):
                output = model(x)
                
                # ============ 統一處理模型輸出 ============
                main_logits, aux_logits = extract_logits(output)
                
                # 計算主損失
                loss = crit(main_logits, y)
                
                # 如果有輔助輸出（僅 Inception V3 訓練時），加入輔助損失
                if aux_logits is not None:
                    # 輔助損失使用相同的 criterion 或簡單的 CE
                    if callable(crit):
                        aux_loss = crit(aux_logits, y)
                    else:
                        # 如果 crit 不是 callable (理論上不會發生)
                        aux_loss = nn.functional.cross_entropy(aux_logits, y)
                    
                    # 輔助損失權重：Inception V3 論文建議 0.3-0.4
                    loss = loss + 0.3 * aux_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ema.update(model)

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(tr)

        # --- Evaluate train & val ---
        _ , y_tr, yhat_tr = eval(model, tr, device, crit)
        va_loss, y_va, yhat_va = eval(model, va, device, crit)

        train_f1 = f1_score(y_tr, yhat_tr, average="macro")
        val_f1 = f1_score(y_va, yhat_va, average="macro")
        train_acc = accuracy_score(y_tr, yhat_tr)
        val_acc = accuracy_score(y_va, yhat_va)

        train_losses.append(avg_train_loss)
        val_losses.append(va_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        tqdm.write(
            f"[Epoch {ep}/{args.epochs}] "
            f"train_loss={avg_train_loss:.4f}, val_loss={va_loss:.4f}, "
            f"train_F1={train_f1:.4f}, val_F1={val_f1:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        # --- LR Scheduler step ---
        prev_lr = get_lr(opt)
        if warmup_sched is not None and ep <= args.warmup_epochs:
            warmup_sched.step()
            curr_lr = get_lr(opt)
            tqdm.write(f"[LR] warmup {ep}/{args.warmup_epochs}: {prev_lr:.2e} -> {curr_lr:.2e}")
        else:
            if args.sched == "plateau":
                metric = val_f1 if args.monitor == "val_f1" else va_loss
                main_sched.step(metric)
            else:
                main_sched.step()
            curr_lr = get_lr(opt)
            if curr_lr < prev_lr:
                tqdm.write(f"[LR] reduced: {prev_lr:.2e} -> {curr_lr:.2e}")
            else:
                tqdm.write(f"[LR] current: {curr_lr:.2e}")

        # --- Save best & Early Stopping ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = args.es_patience
            tqdm.write(f"[F1] New_best_f1: {best_f1:.4e}, patience left: {patience_left}")
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(ema.ema.state_dict(), best_path)
        else:
            patience_left -= 1
            tqdm.write(f"[F1] best_f1: {best_f1:.4e}, patience left: {patience_left}")
            if patience_left <= 0:
                print(f"Early stopping at epoch {ep} (best val_F1={best_f1:.4f})")
                break

    print(f"Best val macro-F1: {best_f1:.4f}")

    best_state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state)

    _, y_tr, yhat_tr = eval(ema.ema, tr, device, criterion=None)
    _, y_va, yhat_va = eval(ema.ema, va, device, criterion=None)

    rep_tr = classification_report(y_tr, yhat_tr, target_names=CLASSES, digits=4)
    rep_va = classification_report(y_va, yhat_va, target_names=CLASSES, digits=4)

    save_confusion_matrix("Confusionmatrix_train.png", y_tr, yhat_tr, CLASSES)
    save_confusion_matrix("Confusionmatrix_val.png",   y_va, yhat_va, CLASSES)
    print("Saved confusion matrices → cm_train.png, cm_val.png")

    report_path = f"report_{args.backbone}.txt" if hasattr(args, "backbone") else "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Classification Report (Train) ===\n")
        f.write(rep_tr + "\n\n")
        f.write("=== Classification Report (Val) ===\n")
        f.write(rep_va + "\n")
    print("Saved reports.")

    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, train_accs, val_accs, name=dir_name)

    return best_path

@torch.no_grad()
def eval(model, loader, device, criterion=None, use_amp=True):
    model.eval()
    tot_loss, n_batches = 0.0, 0
    y_true, y_pred, all_probs = [], [], []
    autocast = torch.amp.autocast if hasattr(torch, "amp") else torch.cuda.amp.autocast

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with autocast('cuda', enabled=(use_amp and torch.cuda.is_available())):
            output = model(x)
            logits, _ = extract_logits(output) # 統一處理輸出
            probs = logits.softmax(1)
            
            if criterion is not None:
                loss = criterion(logits, y)
                tot_loss += float(loss.item())
                n_batches += 1
                
        all_probs.append(probs.cpu().numpy())
        y_true.append(y.cpu().numpy())
        y_pred.append(probs.argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    avg_loss = (tot_loss / n_batches) if n_batches > 0 else None
    return avg_loss, y_true, y_pred

# ============ Test-Time Augmentation (TTA) ============
@torch.no_grad()
def test_with_tta(ckpt_path, csv_test, dir_test, args):
    """
    使用 Test-Time Augmentation 進行預測
    支援兩種模式：
    1. 隨機 TTA (原始方法)
    2. 確定性 TTA (使用 build_tta_tfms)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(num_classes=len(CLASSES), dropout=args.dropout, backbone=args.backbone).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True) 
    model.load_state_dict(state)
    model.eval()

    num_w = max(1, os.cpu_count() // 2)
    
    # ============ 使用改進的 TTA 策略 ============
    if args.tta_method == "deterministic":
        tta_level = 'light' if args.tta_times <= 2 else ('medium' if args.tta_times <= 4 else 'heavy')
        tta_tfms = build_tta_tfms(img_size=args.img_size, tta_level=tta_level)
        actual_tta_times = len(tta_tfms)
        print(f"Running Deterministic TTA with {actual_tta_times} predefined augmentations (level: {tta_level})")
        
        all_predictions = []
        filenames = []
        
        for tta_idx, tfm in enumerate(tta_tfms):
            ds_te = CXRCSV(csv_test, dir_test, is_train=False, has_labels=False, img_size=args.img_size)
            ds_te.tfms = tfm  # 使用特定的 TTA 轉換
            
            te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, num_workers=num_w, pin_memory=True)
            
            batch_probs = []
            for x, fns in tqdm(te, desc=f"TTA {tta_idx+1}/{actual_tta_times}", ncols=150):
                x = x.to(device)
                output = model(x)
                # 統一處理輸出
                logits, _ = extract_logits(output)
                probs = logits.softmax(1).cpu().numpy()
                batch_probs.append(probs)
                if tta_idx == 0:
                    filenames.extend(fns)
            
            all_predictions.append(np.concatenate(batch_probs, axis=0))
    
    else:
        # 使用隨機 TTA (原始方法)
        print(f"Running Random TTA with {args.tta_times} augmentations...")
        
        all_predictions = []
        filenames = []
        
        for tta_idx in range(args.tta_times):
            ds_te = CXRCSV(csv_test, dir_test, is_train=True, has_labels=False, img_size=args.img_size)
            te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, num_workers=num_w, pin_memory=True)
            
            batch_probs = []
            for x, fns in tqdm(te, desc=f"TTA {tta_idx+1}/{args.tta_times}", ncols=150):
                x = x.to(device)
                output = model(x)
                # 統一處理輸出
                logits, _ = extract_logits(output)
                probs = logits.softmax(1).cpu().numpy()
                batch_probs.append(probs)
                if tta_idx == 0:
                    filenames.extend(fns)
            
            all_predictions.append(np.concatenate(batch_probs, axis=0))
    
    # 檢查是否有收集到檔名
    if len(filenames) == 0:
        raise ValueError("No filenames collected! Check if test dataset is empty.")
    
    # 平均所有 TTA 的預測結果
    avg_probs = np.mean(all_predictions, axis=0)  # (N, num_classes)
    
    print(f"Total predictions: {len(avg_probs)}, Total filenames: {len(filenames)}")
    
    pred_idx = avg_probs.argmax(1)
    one_hot = np.eye(len(CLASSES), dtype=int)[pred_idx]
    
    # 儲存結果
    rows = []
    for fn, oh in zip(filenames, one_hot):
        rows.append([fn] + list(oh))
    
    out_csv = getattr(args, "out_csv", "submit.csv")
    pd.DataFrame(rows, columns=["new_filename"] + CLASSES).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} (TTA-enhanced submission)")
    
    # 也儲存概率版本（可選）
    if args.save_probs:
        prob_csv = out_csv.replace('.csv', '_probs.csv')
        prob_df = pd.DataFrame(avg_probs, columns=CLASSES)
        prob_df.insert(0, 'new_filename', filenames)
        prob_df.to_csv(prob_csv, index=False)
        print(f"Saved probability scores → {prob_csv}")

@torch.no_grad()
def test(ckpt_path, csv_test, dir_test, args):
    """原始測試函數 (不使用 TTA) """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(num_classes=len(CLASSES), dropout=args.dropout, backbone=args.backbone).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True) 
    model.load_state_dict(state)
    model.eval()

    ds_te = CXRCSV(csv_test, dir_test, is_train=False, has_labels=False, img_size=args.img_size)
    num_w = max(1, os.cpu_count() // 2)
    te = DataLoader(ds_te, batch_size=args.bs, shuffle=False, num_workers=num_w, pin_memory=True)

    rows = []
    for x, fns in tqdm(te, desc="Saving submission", ncols=150):
        x = x.to(device)
        output = model(x)
        # 統一處理輸出
        logits, _ = extract_logits(output)
        probs = logits.softmax(1).cpu().numpy()
        pred_idx = probs.argmax(1)
        one_hot = np.eye(len(CLASSES), dtype=int)[pred_idx]
        for fn, oh in zip(fns, one_hot):
            rows.append([fn] + list(oh))     

    out_csv = getattr(args, "out_csv", "submit.csv")
    pd.DataFrame(rows, columns=["new_filename"] + CLASSES).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} (one-hot submission)")

if __name__ == "__main__":
    seed_everything(42)
    ap = argparse.ArgumentParser()
    # hyperparams
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=480)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--es-patience", type=int, default=50, help="Early stopping patience")
    ap.add_argument("--backbone-lr-ratio", type=float, default=None, help="backbone 學習率比例 (None=全參數統一學習率, 0.05-0.2=差異化學習率)")
    ap.add_argument("--dropout", type=float, default=0.2)
    
    # ============ Loss Function 參數 ============
    ap.add_argument("--loss", type=str, default="combined", 
                    choices=["ce", "focal", "cb_focal", "combined"],
                    help="損失函數類型: ce=CrossEntropy, focal=FocalLoss, cb_focal=Class-Balanced Focal Loss")
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="Label smoothing 係數 (0.0-0.2)")
    ap.add_argument("--focal-gamma", type=float, default=1.5,
                    help="Focal Loss gamma 參數 (1.0-3.0)，越大越關注困難樣本")
    ap.add_argument("--cb-beta", type=float, default=0.999,
                    help="Class-Balanced Loss beta 參數 (0.9-0.9999)")

    # dataset paths
    ap.add_argument("--train-csv", type=str, default="./datasets/train_data.csv")
    ap.add_argument("--train-dir", type=str, default="./datasets/train_images")
    ap.add_argument("--val-csv",   type=str, default="./datasets/val_data.csv")
    ap.add_argument("--val-dir",   type=str, default="./datasets/val_images")
    ap.add_argument("--test-dir",  type=str, default="./datasets/test_images")
    ap.add_argument("--test-csv",  type=str, default="test_data.csv")
    ap.add_argument("--out-csv",   type=str, default="Submission.csv")

    # Scheduler
    ap.add_argument("--sched", type=str, default="cosine", choices=["plateau", "cosine"])
    ap.add_argument("--monitor", type=str, default="val_f1", choices=["val_f1", "val_loss"])
    ap.add_argument("--plateau-factor", type=float, default=0.1)
    ap.add_argument("--plateau-patience", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--backbone", type=str, default="efficientnet_v2_s", 
                    choices=["resnet18","resnet34","densenet121","efficientnet_b0",
                            "efficientnet_v2_s","efficientnet_v2_m","efficientnet_v2_l", "inception_v3", "coatnet_0", "coatnet_1"])
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--warmup-start", type=float, default=0.1)
    
    # ============ TTA 參數 ============
    ap.add_argument("--use-tta", action="store_true", 
                    help="使用 Test-Time Augmentation")
    ap.add_argument("--tta-method", default="deterministic", 
                    choices=["deterministic", "random"],
                    help="TTA 方法: deterministic=確定性轉換, random=隨機增強")
    ap.add_argument("--tta-times", type=int, default=5,
                    help="TTA 增強次數 (建議 3-10)")
    ap.add_argument("--save-probs", action="store_true",
                    help="儲存預測概率")
    
    # ============ Test-only 模式 ============
    ap.add_argument("--test-only", action="store_true",
                    help="只進行測試，不訓練（需要指定 --ckpt-path）")
    ap.add_argument("--ckpt-path", type=str, default=None,
                    help="模型權重路徑（用於 test-only 模式）")

    args = ap.parse_args()

    # ============ Test-only 模式 ============
    if args.test_only:
        if args.ckpt_path is None:
            raise ValueError("--test-only 模式需要指定 --ckpt-path")
        
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"找不到模型檔案: {args.ckpt_path}")
        
        print(f"\n{'='*60}")
        print(f"Test-Only Mode: Loading checkpoint from {args.ckpt_path}")
        print(f"{'='*60}\n")
        
        # Create test csv if missing
        if not os.path.exists(args.test_csv):
            test_files = sorted(os.listdir(args.test_dir))
            pd.DataFrame({"new_filename": test_files}).to_csv(args.test_csv, index=False)
        
        # Test with or without TTA
        if args.use_tta:
            test_with_tta(args.ckpt_path, args.test_csv, args.test_dir, args)
        else:
            test(args.ckpt_path, args.test_csv, args.test_dir, args)
        
        print(f"\nTest completed! Submission saved to {args.out_csv}")
        exit(0)

    # ============ 正常訓練+測試流程 ============
    # Step 1: Train model
    ckpt = train_one(args.train_csv, args.train_dir, args.val_csv, args.val_dir, args)

    # Step 2: Create test csv if missing
    if not os.path.exists(args.test_csv):
        test_files = sorted(os.listdir(args.test_dir))
        pd.DataFrame({"new_filename": test_files}).to_csv(args.test_csv, index=False)

    # Step 3: Test with or without TTA
    if args.use_tta:
        test_with_tta(ckpt, args.test_csv, args.test_dir, args)
    else:
        test(ckpt, args.test_csv, args.test_dir, args)



# python main.py --backbone efficientnet_v2_m --img-size 480 --epochs 300 --bs 48 --lr 5e-4 --weight-decay 5e-4 --dropout 0.25 --loss combined --focal-gamma 1.5 --cb-beta 0.999 --label-smoothing 0.02 --sched cosine --warmup-epochs 10 --min-lr 1e-5 --es-patience 30 --use-tta --tta-times 4 --save-probs --out-csv Submission.csv

# python main.py --backbone efficientnet_v2_m --img-size 480 --epochs 300 --bs 48 --lr 5e-4 --weight-decay 5e-4 --dropout 0.25 --loss combined --focal-gamma 1.5 --cb-beta 0.999 --label-smoothing 0.015 --sched cosine --warmup-epochs 10 --min-lr 1e-5 --es-patience 30 --use-tta --tta-times 4 --save-probs --out-csv Submission.csv
