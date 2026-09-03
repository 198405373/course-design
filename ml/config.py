# -*- coding: utf-8 -*-
"""全局配置：路径、统一输入尺寸、计算设备。"""
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "KolektorSDD"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
MASKS_DIR = PROCESSED_DIR / "masks"
ANNOTATIONS_CSV = PROCESSED_DIR / "annotations.csv"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

# 预处理统一画布尺寸（scripts/preprocess.py --img-size 需与此一致）
IMG_SIZE = 384

# 归一化参数（全图归一化，训练/推理一致）
MEAN, STD = 0.5, 0.5

# 类名（0=无缺陷, 1=有缺陷）
CLASS_NAMES = ["正常", "缺陷"]

# 计算设备：优先 CUDA，回退 CPU（代码在任何机器都能跑）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_num_threads(n=None):
    """限制 CPU 线程数（可选，便于后台/演示环境稳定）。"""
    if n:
        torch.set_num_threads(n)
