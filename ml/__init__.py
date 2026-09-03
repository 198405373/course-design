# -*- coding: utf-8 -*-
"""ml 算法包：缺陷检测系统的算法模块（CNN/特征/数据）共用入口。"""
from .config import (PROJECT_ROOT, DATA_DIR, IMAGES_DIR, MASKS_DIR, RAW_DIR,
                     PROCESSED_DIR, ANNOTATIONS_CSV, SPLITS_DIR, MODELS_DIR,
                     REPORTS_DIR, FIGURES_DIR, IMG_SIZE, CLASS_NAMES, DEVICE,
                     ensure_dirs, set_num_threads)
from .dataset import DefectDataset, load_splits, build_label_map, compute_class_weights
from .model import build_model
from .transforms import letterbox_gray, to_tensor, load_gray_bytes
from .features import (extract_features, extract_features_batch,
                       horizontal_band_features)

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "IMAGES_DIR", "MASKS_DIR", "RAW_DIR",
    "PROCESSED_DIR", "ANNOTATIONS_CSV", "SPLITS_DIR", "MODELS_DIR",
    "REPORTS_DIR", "FIGURES_DIR", "IMG_SIZE", "CLASS_NAMES", "DEVICE",
    "ensure_dirs", "set_num_threads",
    "DefectDataset", "load_splits", "build_label_map", "compute_class_weights",
    "build_model",
    "letterbox_gray", "to_tensor", "load_gray_bytes",
    "extract_features", "extract_features_batch", "horizontal_band_features",
]
