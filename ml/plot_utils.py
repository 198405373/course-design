# -*- coding: utf-8 -*-
"""绘图工具：配置无界面后端与中文字体（Windows 常见字体回退）。"""
import os
import matplotlib
from matplotlib import font_manager

_CJK_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
]


def setup_matplotlib(agg: bool = True):
    """agg=True 使用无界面后端（适合脚本出图）。配置中文字体避免乱码。"""
    if agg:
        matplotlib.use("Agg")
    chosen = None
    for path in _CJK_CANDIDATES:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                chosen = font_manager.FontProperties(fname=path).get_name()
                break
            except Exception:
                continue
    if chosen:
        matplotlib.rcParams["font.family"] = chosen
        matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen
