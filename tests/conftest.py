# -*- coding: utf-8 -*-
"""pytest 共享夹具：项目路径、隔离数据库、样例图片。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.db as db_mod
from ml import IMAGES_DIR, build_label_map


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """每个测试使用独立 SQLite，避免污染 backend/defects.db。"""
    db_path = tmp_path / "test_defects.db"
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db(db_path)
    return db_path


@pytest.fixture(scope="session")
def sample_images():
    """返回 (缺陷图字节, 正常图字节)。"""
    label_map = build_label_map()
    defect = normal = None
    for f in sorted(IMAGES_DIR.glob("*.jpg")):
        lbl = label_map.get(f.name)
        if lbl == 1 and defect is None:
            defect = f
        elif lbl == 0 and normal is None:
            normal = f
        if defect and normal:
            break
    assert defect and normal, "未找到样例图"
    return (defect.read_bytes(), normal.read_bytes(),
            defect.name, normal.name)
