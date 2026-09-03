# -*- coding: utf-8 -*-
"""图像预处理公共函数（供推理服务 / 特征提取 / 训练统一使用）。

必须与 scripts/preprocess.py 产出的几何完全一致：
  等比缩放使长边=IMG_SIZE → 居中放入 IMG_SIZE×IMG_SIZE 画布（0 背景）。
归一化约定与 ml/dataset.py 一致：x=(gray/255 - 0.5)/0.5。
"""
import cv2
import numpy as np

from .config import IMG_SIZE, MEAN, STD


def letterbox_gray(img: np.ndarray, size: int = IMG_SIZE):
    """等比缩放+居中填充，返回 (画布uint8, scale, off_x, off_y)。支持任意输入宽高。"""
    if img.ndim == 3:  # 彩色转灰度（后端上传可能为彩色）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    rh, rw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    ox = (size - rw) // 2
    oy = (size - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = resized
    return canvas, (scale, ox, oy)


def to_tensor(gray_canvas: np.ndarray) -> np.ndarray:
    """uint8 画布 -> (1,size,size) float32 张量，归一化与训练一致。"""
    x = gray_canvas.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.expand_dims(x, axis=0)


def load_gray_bytes(data: bytes):
    """从字节解码图像（cv2.imdecode 兼容 .jpg/.png），返回灰度图或 None。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
