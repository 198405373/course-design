# -*- coding: utf-8 -*-
"""Flask 应用入口与 REST API。

启动（本地演示）：
    D:\\ANACONDA\\envs\\cv_tutorial\\python.exe backend/app.py
接口一览：
  GET  /                     前端页面（frontend/index.html，缺失时返回接口说明）
  GET  /api/health           存活检查
  POST /api/predict          上传图像 -> 双模型检测（multipart: image + threshold?）
  GET  /api/records          检测历史（?limit=&offset=）
  POST /api/records/<id>/review   人工复检回填（json: {"result":"pass|fail"}）
  GET  /api/stats            质检统计（总数/OK/NG/待复检/缺陷率）
  GET  /api/image/<id>       取回已检测图像副本
"""
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

# 允许 `python backend/app.py` 直接运行时导入项目根下的 backend / ml 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.db as db
import backend.service as service
from backend.service import predict_image, decide_verdict
from ml import IMAGES_DIR

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR.parent / "frontend"


def create_app() -> Flask:
    # 静态资源（css/js/lib）由 Flask 托管；frontend/index.html 由 '/' 路由发送
    (FRONTEND_DIR / "static").mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, static_folder=str(FRONTEND_DIR / "static"),
                static_url_path="/static")
    db.init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 后台预热双模型（只做一次），使首个请求也能保持低延迟
    if not service._WARMED:
        threading.Thread(target=service.warmup, daemon=True).start()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify({"error": "缺少 image 文件字段"}), 400
        try:
            threshold = float(request.form.get("threshold", 0.5))
        except ValueError:
            return jsonify({"error": "threshold 必须为数字"}), 400
        data = f.read()
        t0 = time.perf_counter()
        try:
            res = predict_image(data)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        res = dict(res)
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        verdict = decide_verdict(res["p_cnn"], res["p_ml"], threshold)
        rid = db.insert_record(f.filename, res["p_cnn"], res["p_ml"],
                               res["ml_band"], verdict)
        # 保存检测图像副本（供前端历史回看）；失败不阻断主流程
        saved = None
        try:
            arr = np.frombuffer(data, dtype="uint8")
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            save_path = UPLOAD_DIR / f"{rid}.jpg"
            cv2.imwrite(str(save_path), img)
            saved = f"/api/image/{rid}"
        except Exception:
            pass
        return jsonify({
            "id": rid, "filename": f.filename, "verdict": verdict,
            "p_cnn": res["p_cnn"], "p_ml": res["p_ml"],
            "ml_band": res["ml_band"], "threshold": threshold,
            "latency_ms": res["latency_ms"],
            "image_url": saved, "message": verdict_label(verdict),
        }), 200

    @app.get("/api/records")
    def records():
        limit = request.args.get("limit", default=50, type=int)
        offset = request.args.get("offset", default=0, type=int)
        limit = max(1, min(limit, 200)); offset = max(0, offset)
        return jsonify({"records": db.list_records(limit=limit, offset=offset)})

    @app.post("/api/records/reset")
    def reset():
        """清空全部检测历史与统计，并清理上传图像副本（演示/录制前重置）。"""
        db.reset_records()
        for f in UPLOAD_DIR.glob("*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass
        return jsonify({"ok": True})

    @app.post("/api/records/<int:rid>/review")
    def review(rid: int):
        body = request.get_json(silent=True) or {}
        result = (body.get("result") or "").lower()
        try:
            ok = db.update_review(rid, result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if not ok:
            return jsonify({"error": "记录不存在"}), 404
        return jsonify({"id": rid, "review_result": result})

    @app.post("/api/records/reset")
    def reset_records():
        """清空全部检测记录（统计随之归零、ID 从 #1 重新开始）。
        请求体可选 {"also_images": true} 一并删除历史图像副本。
        用于演示/录视频前"一键清零"，需谨慎（前端有二次确认）。
        """
        body = request.get_json(silent=True) or {}
        also = bool(body.get("also_images", False))
        cleared = db.reset_records()
        cleared_images = 0
        if also and UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.glob("*.jpg"):
                try:
                    f.unlink()
                    cleared_images += 1
                except OSError:
                    pass
        return jsonify({"ok": True, "cleared_records": cleared,
                        "cleared_images": cleared_images})

    @app.get("/api/stats")
    def stat():
        return jsonify(db.stats())

    @app.get("/api/image/<int:rid>")
    def image(rid: int):
        path = UPLOAD_DIR / f"{rid}.jpg"
        if not path.exists():
            return jsonify({"error": "图像不存在"}), 404
        return send_file(str(path), mimetype="image/jpeg")

    @app.get("/")
    def index():
        idx = FRONTEND_DIR / "index.html"
        if idx.exists():
            return send_file(str(idx))
        # 前端尚未构建时的占位说明
        return jsonify({
            "service": "工业产品质量智能检测系统 API",
            "hint": "前端 frontend/index.html 尚未创建；可用 POST /api/predict 上传图片进行检测",
            "endpoints": ["/api/health", "/api/predict", "/api/records",
                          "/api/records/<id>/review", "/api/stats"],
        })

    return app


def verdict_label(v: str) -> str:
    return {"OK": "正常", "NG": "缺陷（高风险）", "REVIEW": "疑似-待复检"}.get(v, v)


app = create_app()

if __name__ == "__main__":
    # 本地演示：所有接口可访问；debug 关闭保证稳定
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=False, threaded=True)
