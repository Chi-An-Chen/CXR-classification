"""
Author: Chi-An Chen
Date: 2025-10-27
Version: 2.0
"""
import torch
import numpy as np
import torch.nn as nn

# ============ Focal Loss for F1 Score Optimization ============
class FocalLoss(nn.Module):
    """
    Focal Loss: 更關注困難樣本，對 F1 score 優化特別有效
    
    參數:
        alpha: 類別權重 (tensor, shape=[num_classes])
        gamma: focusing 參數，越大越關注困難樣本 (建議 2.0-3.0)
        label_smoothing: label smoothing 係數
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # 計算 cross entropy loss
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # 計算 pt (預測正確類別的概率)
        pt = torch.exp(-ce_loss)
        
        # Focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # 套用類別權重
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

# ============ Class-Balanced Focal Loss ============
class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss: 結合類別平衡和 Focal Loss
    特別適合醫學影像的不平衡分類問題
    
    參數:
        samples_per_class: 每個類別的樣本數 (list or array)
        beta: 平衡參數 (0.9-0.999)，越接近 1 越重視少數類別
        gamma: focusing 參數 (建議 2.0-3.0)
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        device = inputs.device
        if self.weights.device != device:
            self.weights = self.weights.to(device)
        
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_t = self.weights[targets]
        loss = alpha_t * focal_weight * ce_loss
        
        return loss.mean()