"""
Author: Chi-An Chen
Date: 2025-10-27
Version: 2.0
"""
import os
import copy
import torch
import torch.nn as nn
from torchvision import models

from coanet import coatnet_0, coatnet_1, coatnet_2

def make_model(num_classes=4, dropout=0.1, backbone="resnet18"):
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    
    if backbone == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    
    if backbone == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    
    # ============ ResNeXt-50 ============
    if backbone == "resnext50_32x4d":
        m = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    
    # ============ ResNeXt-101 ============
    if backbone == "resnext101_32x8d":
        m = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m
    
    # ============ EfficientNet-V2-S ============
    if backbone == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m
    
    # ============ EfficientNet-V2-M ============
    if backbone == "efficientnet_v2_m":
        m = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m
    
    if backbone == "efficientnet_v2_l":
        m = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m
    
    # ============ ConvNeXt Tiny ============
    if backbone == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_f = m.classifier[2].in_features  
        m.classifier = nn.Sequential(
            m.classifier[0],
            m.classifier[1],
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        return m
    
    # ============ Inception V3 ============
    if backbone == "inception_v3":
        m = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True  # 保持 aux_logits=True 以載入預訓練權重
        )
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, num_classes)
        )
        # 調整輔助分類器
        if hasattr(m, "AuxLogits"):
            in_f_aux = m.AuxLogits.fc.in_features
            m.AuxLogits.fc = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_f_aux, num_classes)
            )
        return m
    
    elif backbone == "coatnet_1":
        m = coatnet_1()
        in_f = m.fc.in_features  # 768
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    
    raise ValueError("unknown backbone")


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, half_life_steps: int = 2000):
        self.decay = float(pow(0.5, 1.0 / max(1, half_life_steps)))
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd, esd = model.state_dict(), self.ema.state_dict()
        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
            else:
                esd[k] = msd[k]