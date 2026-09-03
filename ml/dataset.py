# -*- coding: utf-8 -*-
"""数据集加载与划分工具（torch Dataset）。

约定（与 scripts/preprocess.py 输出对齐）：
  - images/ 为 384×384 灰度 .jpg（等比缩放+居中填充）；
  - annotations.csv 中 has_defect 列为图像级标签；
  - splits/*.txt 每行是一个 processed 图像名（如 kos01_Part5.jpg）。
"""
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ANNOTATIONS_CSV, IMAGES_DIR, IMG_SIZE, MEAN, SPLITS_DIR, STD

# 归一化参数（全图归一化，训练/推理一致）
# MEAN, STD = 0.5, 0.5  (已在 config 中定义)


def build_label_map(csv_path: Path = ANNOTATIONS_CSV) -> Dict[str, int]:
    """image -> has_defect(0/1) 图像级标签映射。"""
    df = pd.read_csv(csv_path)
    return (df.drop_duplicates("image")
              .set_index("image")["has_defect"].astype(int).to_dict())


def load_splits(splits_dir: Path = SPLITS_DIR) -> Dict[str, List[str]]:
    """读取 train/val/test 三个划分，返回 {split: [图名,...]}。"""
    out: Dict[str, List[str]] = {}
    for name in ("train", "val", "test"):
        f = splits_dir / f"{name}.txt"
        if not f.exists():
            raise FileNotFoundError(f"缺少划分文件：{f}")
        ids = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
               if ln.strip()]
        out[name] = ids
    return out


def compute_class_weights(label_map: Dict[str, int]) -> torch.Tensor:
    """按类别频率反比计算 loss 权重，缓解 缺陷:正常≈1:6.7 的不平衡。"""
    counts = np.bincount([int(v) for v in label_map.values()], minlength=2)
    total = counts.sum()
    weights = total / (len(counts) * counts.astype(float) + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


class DefectDataset(Dataset):
    """灰度缺陷图像分类数据集。

    参数：
      image_names: 图像文件名列表（processed 名）；
      label_map  : image -> has_defect；
      transform  : 是否为训练（开增强：随机水平翻转）；
    """
    def __init__(self, image_names: List[str],
                 label_map: Optional[Dict[str, int]] = None,
                 train: bool = False,
                 aug: str = "flip",
                 image_dir: Path = IMAGES_DIR,
                 img_size: int = IMG_SIZE):
        """aug: 'flip'=仅水平翻转(默认，稳定) / 'full'=含平移与灰度抖动 / 'none'=不增强。

        注意：本数据集样本极少（train≈279），实测 'full' 增强会导致拟合不足、
        验证指标下降，故默认使用轻量 'flip'；'full' 作为可调选项保留。
        """
        if label_map is None:
            label_map = build_label_map()
        # 仅保留目录中实际存在且标签可查的图，避免脏索引
        names = [n for n in image_names
                 if n in label_map and (image_dir / n).exists()]
        if not names:
            raise ValueError("image_names 与 label_map/images 无交集，请检查划分与标注是否一致")
        self.names = names
        self.label_map = label_map
        self.train = train
        self.aug = aug
        self.image_dir = Path(image_dir)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = cv2.imread(str(self.image_dir / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"图像读取失败：{self.image_dir / name}")
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        if self.train:
            if self.aug == "full":
                x = self._augment_full(x)
            elif self.aug == "flip" and np.random.rand() < 0.5:
                x = np.ascontiguousarray(x[:, ::-1])
        x = (x - MEAN) / STD
        x = torch.from_numpy(x).unsqueeze(0)  # (1,H,W)
        label = int(self.label_map[name])
        return x, label

    @staticmethod
    def _augment_full(x: np.ndarray, rng: np.random.RandomState = None):
        """完整的在线增强（水平翻转 + 小幅平移 + 亮度/对比度抖动）。

        预处理已将图像等比缩放并居中填充（四周为 0 背景、余量充足），
        小幅平移不会把内容移出画布；灰度抖动让网络对光照更鲁棒。
        注意：实测在样本量极小时此增强易导致拟合不足，请按需启用。
        """
        rng = rng or np.random
        if rng.rand() < 0.5:                       # 水平翻转
            x = np.ascontiguousarray(x[:, ::-1])
        dy = int(rng.uniform(-10, 10))             # 平移
        dx = int(rng.uniform(-10, 10))
        if dx or dy:
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            x = cv2.warpAffine(x, m, (x.shape[1], x.shape[0]),
                               flags=cv2.INTER_LINEAR, borderValue=0.0)
        alpha = float(rng.uniform(0.9, 1.1))       # 对比度
        beta = float(rng.uniform(-0.05, 0.05))     # 亮度
        x = np.clip(x * alpha + beta, 0.0, 1.0)
        return x
