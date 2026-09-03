# -*- coding: utf-8 -*-
"""推理服务层：加载双模型，对一张图像输出 CNN 与传统 ML 双路结果。

模型只在首次调用时加载（模块级缓存），避免每次请求都读盘。
"""
from typing import Dict, Optional

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F

from ml import (build_model, DEVICE, MODELS_DIR, IMAGES_DIR, ensure_dirs,
                letterbox_gray, to_tensor)
from ml.features import horizontal_band_features

N_BANDS = 8  # 与 scripts/train_ml.py 一致

_cnn_model: Optional[torch.nn.Module] = None
_ml_pipe = None
_cnn_meta: Dict = {}
_WARMED = False


def warmup():
    """预加载双模型（幂等），避免首个请求因加载模型而变慢。"""
    global _WARMED
    if not _WARMED:
        _WARMED = True
        _load_models()


def _load_models():
    """惰性加载 CNN 权重与传统 ML pipeline。"""
    global _cnn_model, _ml_pipe, _cnn_meta
    ensure_dirs()
    if _cnn_model is None:
        ckpt = torch.load(MODELS_DIR / "best_cnn.pt", map_location="cpu")
        _cnn_meta = ckpt.get("meta", {})
        m = build_model().to(DEVICE)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        _cnn_model = m
    if _ml_pipe is None:
        _ml_pipe = joblib.load(MODELS_DIR / "best_ml.pkl")


def predict_image(data: bytes) -> Dict:
    """输入图像字节，返回双路检测结果。

    返回结构：
      p_cnn   CNN 缺陷概率
      p_ml    传统分带 RF 缺陷概率（=各带最大缺陷概率）
      ml_band 传统路判定的缺陷所在条带（0~N_BANDS-1，用于定位提示）
    """
    _load_models()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法解码图像，请上传 jpg/png 格式图片")
    canvas, _ = letterbox_gray(img)

    # 1) CNN 路：to_tensor 返回 (C,H,W)=(1,384,384)，补 batch 维后送入模型
    x = torch.from_numpy(to_tensor(canvas)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = _cnn_model(x)
        p_cnn = float(F.softmax(logits, dim=1)[0, 1].item())

    # 2) 传统分带 RF 路
    feats = horizontal_band_features(canvas, N_BANDS)  # (N_BANDS, D)
    probs = _ml_pipe.predict_proba(feats)[:, 1]        # (N_BANDS,)
    p_ml = float(probs.max())
    ml_band = int(probs.argmax())

    return {
        "p_cnn": round(p_cnn, 4),
        "p_ml": round(p_ml, 4),
        "ml_band": ml_band,
    }


def decide_verdict(p_cnn: float, p_ml: float, threshold: float = 0.5) -> str:
    """双证据融合三态裁决（质检业务规则）：
      - 两路同判缺陷 → NG（高风险，直接判不合格）
      - 仅一路判缺陷（分歧）→ REVIEW（疑似，转人工复检）
      - 两路均正常   → OK
    设计说明：CNN 灵敏（低漏检），传统特征高特异（低误报），
    两者一致时置信最高，分歧样本交由人工复核，兼顾检出率与误报率。
    """
    cnn_ng = p_cnn >= threshold
    ml_ng = p_ml >= threshold
    if cnn_ng and ml_ng:
        return "NG"
    if cnn_ng or ml_ng:
        return "REVIEW"
    return "OK"
