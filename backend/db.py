# -*- coding: utf-8 -*-
"""SQLite 数据访问层：检测记录、人工复检、统计。"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "defects.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    saved_image   TEXT,
    p_cnn         REAL NOT NULL,
    p_ml          REAL NOT NULL,
    ml_band       INTEGER,
    verdict       TEXT NOT NULL,          -- OK / NG / REVIEW
    review_result TEXT,                   -- NULL / pass / fail （人工复检闭环）
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_time ON records(created_at);
"""

VERDICT_OK, VERDICT_NG, VERDICT_REVIEW = "OK", "NG", "REVIEW"


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None):
    """建表（幂等）。测试可传临时 db_path 隔离。"""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def insert_record(filename: str, p_cnn: float, p_ml: float, ml_band: Optional[int],
                  verdict: str, saved_image: Optional[str] = None,
                  db_path: Optional[Path] = None) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO records(filename,saved_image,p_cnn,p_ml,ml_band,verdict,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (filename, saved_image, float(p_cnn), float(p_ml), ml_band, verdict, now))
        return int(cur.lastrowid)


def list_records(limit: int = 50, offset: int = 0,
                 db_path: Optional[Path] = None) -> List[Dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM records ORDER BY id DESC LIMIT ? OFFSET ?",
            (int(limit), int(offset))).fetchall()
    return [dict(r) for r in rows]


def get_record(rid: int, db_path: Optional[Path] = None) -> Optional[Dict]:
    with _connect(db_path) as conn:
        r = conn.execute("SELECT * FROM records WHERE id=?", (int(rid),)).fetchone()
    return dict(r) if r else None


def reset_records(db_path: Optional[Path] = None) -> None:
    """清空全部检测记录并重置自增 ID（从 #1 重新开始）。"""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM records")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='records'")


def update_review(rid: int, result: str, db_path: Optional[Path] = None) -> bool:
    """人工复检回填结果（pass=合格放行 / fail=确认缺陷）。"""
    if result not in ("pass", "fail"):
        raise ValueError("result must be 'pass' or 'fail'")
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE records SET review_result=? WHERE id=?",
            (result, int(rid)))
        return cur.rowcount > 0


def reset_records(db_path: Optional[Path] = None) -> int:
    """清空检测记录并重置自增 ID（从 #1 重新开始），返回删除行数。"""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM records")
        deleted = cur.rowcount
        conn.execute("DELETE FROM sqlite_sequence WHERE name='records'")
    return deleted


def reset_records(db_path: Optional[Path] = None,
                  upload_dir: Optional[Path] = None) -> int:
    """清空检测记录并重置自增 ID（下一条从 #1 开始）；可选清理上传图像副本。

    用于答辩/录视频前把系统还原到"从零开始"状态。
    """
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM records")
        try:
            # 自增计数器复位（sqlite_sequence 在首次插入后才会存在）
            conn.execute("DELETE FROM sqlite_sequence WHERE name='records'")
        except sqlite3.OperationalError:
            pass
    if upload_dir is not None:
        up = Path(upload_dir)
        if up.exists():
            for f in up.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
    return cur.rowcount


def stats(db_path: Optional[Path] = None) -> Dict:
    """质检统计：总数/各类别/缺陷率/复检分布（供前端图表）。"""
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM records").fetchone()["c"]
        by_verdict = {r["verdict"]: r["c"] for r in
                      conn.execute("SELECT verdict, COUNT(*) c FROM records "
                                   "GROUP BY verdict").fetchall()}
        by_review = {r["review_result"]: r["c"] for r in
                     conn.execute("SELECT review_result, COUNT(*) c FROM records "
                                  "WHERE review_result IS NOT NULL GROUP BY review_result").fetchall()}
        recent = conn.execute(
            "SELECT id, created_at, verdict FROM records "
            "ORDER BY id DESC LIMIT 50").fetchall()
    defect_cnt = by_verdict.get(VERDICT_NG, 0) + by_verdict.get(VERDICT_REVIEW, 0)
    return {
        "total": total,
        "ok": by_verdict.get(VERDICT_OK, 0),
        "ng": by_verdict.get(VERDICT_NG, 0),
        "review": by_verdict.get(VERDICT_REVIEW, 0),
        "defect_rate": round(defect_cnt / total, 4) if total else 0.0,
        "review_pass": by_review.get("pass", 0),
        "review_fail": by_review.get("fail", 0),
        "recent": [{"id": r["id"], "created_at": r["created_at"], "verdict": r["verdict"]}
                   for r in recent],
    }
