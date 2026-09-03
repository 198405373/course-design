# -*- coding: utf-8 -*-
"""算法/数据层单元测试。"""
import numpy as np
import pytest
import torch

from ml import (build_label_map, load_splits, letterbox_gray, to_tensor,
                extract_features, horizontal_band_features, IMAGES_DIR,
                IMG_SIZE, MODELS_DIR)
from ml.dataset import DefectDataset


def test_splits_clean_and_exist():
    """划分文件只含干净图名且图片都存在、无 label 污染。"""
    lm = build_label_map()
    sp = load_splits()
    assert set(sp) == {"train", "val", "test"}
    for split in sp.values():
        for n in split:
            assert "_label" not in n, f"{n} 含 label 污染"
            assert (IMAGES_DIR / n).exists(), f"{n} 图片缺失"
            assert n in lm, f"{n} 无标签"
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == 399


def test_dataset_item():
    sp = load_splits()
    ds = DefectDataset(sp["val"], train=False)
    x, y = ds[0]
    assert x.shape == (1, IMG_SIZE, IMG_SIZE)
    assert x.dtype == torch.float32
    assert y in (0, 1)


def test_letterbox_shape_and_geometry():
    # 细长原图（宽500 高1250）等比放入 384×384
    tall = np.full((1250, 500), 128, dtype=np.uint8)
    tall[100:200, 200:300] = 200
    canvas, (scale, ox, oy) = letterbox_gray(tall)
    assert canvas.shape == (IMG_SIZE, IMG_SIZE)
    assert canvas.dtype == np.uint8
    # 等比：高被缩到 384 → scale = 384/1250；内容高度≈384，宽度≈153
    rh = round(1250 * scale)
    rw = round(500 * scale)
    assert rh == IMG_SIZE and rw < IMG_SIZE
    # 画布中部内容非零、两侧 pad 为 0
    assert canvas[:, :ox].max() == 0
    assert canvas[oy:oy + rh, ox:ox + rw].max() == 200


def test_to_tensor_normalization():
    canvas = np.full((IMG_SIZE, IMG_SIZE), 128, dtype=np.uint8)
    x = to_tensor(canvas)
    assert x.shape == (1, IMG_SIZE, IMG_SIZE)
    expected = (128 / 255.0 - 0.5) / 0.5  # 与训练归一化一致
    assert abs(float(x[0, 0, 0]) - expected) < 1e-5
    assert np.isfinite(x).all()


def test_feature_extract_dim_and_finite():
    img = np.random.default_rng(0).integers(0, 256, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
    feats = extract_features(img)
    assert feats.shape == (312,)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_band_features_shape():
    img = np.random.default_rng(1).integers(0, 256, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
    bands = horizontal_band_features(img, 8)
    assert bands.shape == (8, 312)


def test_verdict_three_state():
    from backend.service import decide_verdict
    assert decide_verdict(0.9, 0.8, 0.5) == "NG"      # 双证据
    assert decide_verdict(0.9, 0.2, 0.5) == "REVIEW"  # 分歧
    assert decide_verdict(0.1, 0.2, 0.5) == "OK"


@pytest.mark.parametrize("ckpt", ["best_cnn.pt"])
def test_cnn_loads_and_predicts(ckpt):
    path = MODELS_DIR / ckpt
    if not path.exists():
        pytest.skip("模型文件缺失，跳过推理测试")
    from ml import build_model, DEVICE
    import torch.nn.functional as F
    ck = torch.load(path, map_location="cpu")
    m = build_model().to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval()
    x = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        p = F.softmax(m(x), dim=1)[0, 1].item()
    assert 0.0 <= p <= 1.0
