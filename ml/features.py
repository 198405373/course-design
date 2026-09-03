# -*- coding: utf-8 -*-
"""传统特征工程：面向工业表面缺陷的手工特征提取（技术方向：模式识别/特征工程）。

与深度学习端到端特征学习形成"对照基线"，两者在后端同时参与推理。

特征构成（约 310 维，全部可在答辩中解释其物理含义）：
  1. 全局灰度统计：均值/标准差/偏度/峰度 —— 反映整体明暗与一致性；
  2. 4×4 分块统计（每块 mean/std）—— 捕捉缺陷所在局部的灰度突变；
  3. 局部二值模式 LBP(r=1,8 邻域) 全局直方图(256) —— 刻画表面纹理规律，
     正常件呈均匀纹理，缺陷会破坏局部纹理分布；
  4. Sobel 梯度方向直方图(18) + 平均幅值 —— 缺陷边缘的方向信息；
  5. Canny 边缘密度 —— 缺陷边界比例。
"""
from math import cos, pi, sin

import cv2
import numpy as np


def _lbp_image(img: np.ndarray, radius: int = 1, points: int = 8) -> np.ndarray:
    """计算 LBP 码图（默认 r=1，8 邻域，reflect 填充边界）。"""
    assert radius == 1 and points == 8, "本实现仅支持 r=1/8 邻"
    img = img.astype(np.float64)
    p = np.pad(img, 1, mode="reflect")
    h, w = img.shape
    # 8 邻相对坐标，按角度顺序（r=1 下即上下左右+对角）
    offsets = [(int(round(sin(2 * pi * k / points) * radius)),
                int(round(cos(2 * pi * k / points) * radius)))
               for k in range(points)]
    codes = np.zeros((h, w), dtype=np.uint16)
    for i, (dy, dx) in enumerate(offsets):
        nb = p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        codes |= ((nb >= img).astype(np.uint16) << i)
    return codes


def _gradient_hist(img: np.ndarray, bins: int = 18):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = np.arctan2(gy, gx) * 180.0 / pi
    ang = np.mod(ang, 180.0)
    h = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)[0]
    s = h.sum() + 1e-9
    return h / s, float(mag.mean())


def extract_features(gray: np.ndarray) -> np.ndarray:
    """从 384×384 uint8 灰度图提取特征向量（顺序固定，训练/推理必须一致）。"""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float64)
    feats: list = []

    # 1) 全局统计
    feats += [g.mean(), g.std(), float(((g - g.mean()) ** 3).mean() / (g.std() ** 3 + 1e-9)),
              float(((g - g.mean()) ** 4).mean() / (g.std() ** 4 + 1e-9))]

    # 2) 4×4 分块 mean/std
    H, W = gray.shape
    for i in range(4):
        for j in range(4):
            blk = g[i * H // 4:(i + 1) * H // 4, j * W // 4:(j + 1) * W // 4]
            feats += [blk.mean(), blk.std()]

    # 3) LBP 全局直方图
    lbp = _lbp_image(gray)
    hist = np.histogram(lbp, bins=256, range=(0, 256), density=True)[0]
    feats += hist.tolist()

    # 4) 梯度方向直方图 + 平均幅值
    ghist, mag_mean = _gradient_hist(gray)
    feats += ghist.tolist() + [mag_mean]

    # 5) Canny 边缘密度
    edge = cv2.Canny(gray, 50, 150)
    feats.append(float((edge > 0).mean()))

    return np.asarray(feats, dtype=np.float32)


def extract_features_batch(gray_images) -> np.ndarray:
    """批量提取，返回 (N, D) float32。gray_images 可迭代。"""
    return np.stack([extract_features(g) for g in gray_images])


def horizontal_band_features(gray: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """把图沿高度切成 n_bands 条横向带，逐带提特征。

    用途：细长缺陷只落在少数条带内，带内缺陷占比大、可被纹理特征判别，
    据此既判"是否有缺陷"又能给出缺陷所在高度位置（滑窗/分块识别思想）。
    返回 (n_bands, D) float32。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh = h // n_bands
    feats = []
    for i in range(n_bands):
        band = gray[i * bh:(i + 1) * bh]
        if band.shape[0] < 4:
            band = np.pad(band, ((0, 4 - band.shape[0]), (0, 0)), mode="edge")
        feats.append(extract_features(band))
    return np.asarray(feats, dtype=np.float32)
