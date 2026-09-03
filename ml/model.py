# -*- coding: utf-8 -*-
"""轻量级缺陷分类 CNN（全卷积 + 全局平均池化）。

设计要点（答辩可讲）：
  1. 输入为 384×384 灰度图（预处理已等比缩放+居中填充，四周为 0 背景）；
  2. 全卷积主干 + AdaptiveAvgPool，因此对输入分辨率不敏感，后端可换分辨率推理；
  3. BatchNorm + ReLU + Dropout，轻量、CPU 友好（参数量 < 1M）。
"""
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # stage1: 384 -> 96 (stride2 + maxpool2)
            nn.Conv2d(in_channels, base, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # stage2: 96 -> 24
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # stage3: 24 -> 12
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            # stage4: 12 -> 6
            nn.Conv2d(base * 4, base * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(num_classes: int = 2, in_channels: int = 1) -> nn.Module:
    """返回 SmallCNN 实例。"""
    return SmallCNN(in_channels=in_channels, num_classes=num_classes)
