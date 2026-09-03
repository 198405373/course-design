# -*- coding: utf-8 -*-
"""后端 API 端到端测试（用 Flask test client，无真实网络）。"""
import io
import json

import pytest

from backend import db
from backend.app import create_app


@pytest.fixture()
def client(isolated_db, tmp_path, monkeypatch):
    import backend.app as app_mod
    # 隔离上传图像目录，避免测试写入/清空真实 backend/uploads
    up = tmp_path / "uploads"
    up.mkdir(exist_ok=True)
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", up)
    app = create_app()   # init_db 落在 isolated_db(tmp)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _png_bytes(gray_bytes: bytes) -> bytes:
    import cv2, numpy as np
    arr = np.frombuffer(gray_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    return enc.tobytes()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_missing_image(client):
    r = client.post("/api/predict", data={})
    assert r.status_code == 400


def test_predict_bad_bytes(client):
    r = client.post("/api/predict",
                    data={"image": (io.BytesIO(b"not-an-image"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_defect_and_normal(client, sample_images):
    defect_bytes, normal_bytes, dname, nname = sample_images
    # 缺陷图（jpg 字节直接上传）
    r = client.post("/api/predict",
                    data={"image": (io.BytesIO(defect_bytes), dname)},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["verdict"] in ("OK", "NG", "REVIEW")
    assert 0.0 <= body["p_cnn"] <= 1.0
    assert 0.0 <= body["p_ml"] <= 1.0
    assert isinstance(body["ml_band"], int)
    assert body["id"] > 0
    # 正常图用 png 编码验证可接受
    r2 = client.post("/api/predict",
                     data={"image": (io.BytesIO(_png_bytes(normal_bytes)), "n.png"),
                           "threshold": "0.5"},
                     content_type="multipart/form-data")
    assert r2.status_code == 200


def test_records_and_stats_flow(client, sample_images):
    defect_bytes, _, dname, _ = sample_images
    # 先造一条记录
    r = client.post("/api/predict",
                    data={"image": (io.BytesIO(defect_bytes), dname)},
                    content_type="multipart/form-data")
    rid = r.get_json()["id"]

    recs = client.get("/api/records").get_json()["records"]
    assert any(x["id"] == rid for x in recs)

    st = client.get("/api/stats").get_json()
    assert st["total"] >= 1
    assert st["ok"] + st["ng"] + st["review"] == st["total"]


def test_review_update_and_errors(client, sample_images):
    defect_bytes, _, dname, _ = sample_images
    r = client.post("/api/predict",
                    data={"image": (io.BytesIO(defect_bytes), dname)},
                    content_type="multipart/form-data")
    rid = r.get_json()["id"]

    ok = client.post(f"/api/records/{rid}/review", json={"result": "pass"})
    assert ok.status_code == 200
    assert ok.get_json()["review_result"] == "pass"

    # 非法复检值
    bad = client.post(f"/api/records/{rid}/review", json={"result": "maybe"})
    assert bad.status_code == 400
    # 不存在记录
    missing = client.post("/api/records/99999/review", json={"result": "fail"})
    assert missing.status_code == 404


def test_reset_records(client, sample_images):
    defect_bytes, _, dname, _ = sample_images

    def upload():
        r = client.post("/api/predict",
                        data={"image": (io.BytesIO(defect_bytes), dname)},
                        content_type="multipart/form-data")
        return r.get_json()

    assert upload()["id"] >= 1
    assert client.get("/api/stats").get_json()["total"] >= 1

    # 清空
    r = client.post("/api/records/reset")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert client.get("/api/records").get_json()["records"] == []
    assert client.get("/api/stats").get_json()["total"] == 0

    # 自增 ID 已重置：新记录从 #1 开始
    assert upload()["id"] == 1
