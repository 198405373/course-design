# 制造智能技术课程设计 —— 工业产品质量智能检测系统

基于 **KolektorSDD**（换向器塑料表面缺陷）数据集的工业表面缺陷智能检测 B/S 应用。
Vibe Coding 开发：全程使用 AI 编程工具（DeepSeek / Trae / VS Code Copilot）辅助完成。

> 技术栈：Flask + SQLite + PyTorch(CNN) + scikit-learn(RF/传统特征) + OpenCV + 原生前端

## 目录结构

```
course-design/
├── data/
│   ├── raw/KolektorSDD/        # 原始数据（不入库，需自行解压/下载）
│   ├── processed/              # 预处理产物（可再生，不入库）
│   │   ├── images/  masks/     # 384×384 等比缩放+居中填充 图/掩码
│   │   ├── annotations.csv     # 缺陷框标注（YOLO 归一化）
│   │   └── dataset_meta.json
│   └── splits/{train,val,test}.txt   # 分层划分（seed=42）279/60/60
├── ml/                         # 算法包（后端与训练共用）
│   ├── transforms.py           # letterbox / 归一化（训练推理一致）
│   ├── features.py             # 传统特征（LBP/梯度/统计 + 分带）
│   ├── dataset.py / model.py   # torch 数据集 / SmallCNN
│   └── config.py / plot_utils.py
├── scripts/
│   ├── preprocess.py           # 数据预处理（修复版）
│   ├── train.py                # CNN 训练
│   ├── train_ml.py             # 传统分带 RF 训练
│   ├── evaluate.py             # CNN 评估
│   └── audit.py                # 数据审计工具
├── backend/                    # Flask 后端（API + SQLite + 推理服务）
│   ├── app.py  db.py  service.py
├── frontend/                   # 前端 UI（上传/结果/历史/统计，ECharts 本地库）
├── docs/                       # 需求规格说明书 / 设计报告撰写大纲
├── process/                    # AI 协作过程档案（prompt 日志）
├── tests/                      # pytest 自动化测试（14 项）
├── models/                     # 训练产物：best_cnn.pt / best_ml.pkl
├── reports/  figures/          # 评估报告与图表
└── requirements.txt
```

## 环境准备

```bash
conda create -n cv_tutorial python=3.10 -y
conda activate cv_tutorial
# 若 torch 需 CPU 版，改用：pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 数据（KolektorSDD，CC BY-NC-SA 4.0）

- 来源：https://www.vicos.si/resources/kolektorsdd/ ，解压后目录应含 `kos01..kos50`
- 399 张 500×1240~1284 灰度图：**52 缺陷 / 347 正常**（缺陷为极细暗条，类不平衡严重）

```bash
# 1) 预处理：只保留真实工件图，等比缩放+居中填充至 384×384，分层划分
python scripts/preprocess.py --img-size 384
# 2) 数据审计（校验干净）
python scripts/audit.py
```

## 模型训练与评估

```bash
# CNN 分类（GPU/CPU 自适应）
python scripts/train.py --epochs 60 --batch-size 32 --lr 1e-3 --weight-decay 0
python scripts/evaluate.py --ckpt models/best_cnn.pt

# 传统分带 RF（对照基线）
python scripts/train_ml.py
```

关键实测结论（test 60 张 / 8 缺陷）：

| 模型 | Balanced Acc | Recall(缺陷) | Precision | AUC | 备注 |
|---|---|---|---|---|---|
| SmallCNN(241K) | 0.784 | 0.875(漏1) | 0.304 | 0.860 | 高灵敏 |
| 分带RF(传统特征) | 0.563 | 0.125 | 1.000 | 0.793 | 高特异 |

> 深度学习在细弱缺陷上显著优于全局手工特征；后端因此采用**双证据融合**：CNN 高灵敏 + 传统特征高特异，二者一致判 NG、分歧转人工复检（REVIEW），业务上兼顾漏检率与误报率。

### 双证据融合评估（业务口径，测试集 60 张 / 缺陷 8）
`python scripts/eval_fusion.py` 输出（`reports/evaluation_fusion.json`）：

| 真实类别 | 系统裁决 | 说明 |
|---|---|---|
| 缺陷 8 张 | NG 1 · REVIEW(待复检) 6 · 漏放 1 | **拦截率 0.875**（缺陷被留下不直接放行） |
| 正常 52 张 | OK 放行 36 · 误标 REVIEW 16 · **误判 NG 0** | 无合格品被直接判废，多检的进人工复检 |

业务价值：单模型硬阈值难以同时压低漏检与误报；融合后**合格品误判为废品为 0**，
其余不确定样本显式进入人工复检，符合"宁多检、不漏放、人工把关"的质检逻辑。

## 后端 API 与运行

Windows 双击 `run_demo.cmd` 一键启动并打开浏览器；或手动：

```bash
python backend/app.py          # http://127.0.0.1:5000
```

| 接口 | 说明 |
|---|---|
| `GET /` | 前端页面（未建时返回接口说明） |
| `POST /api/predict` | multipart 上传 `image`（+可选 `threshold`），返回双路概率与判定 |
| `GET /api/records` | 检测历史 `?limit=&offset=` |
| `POST /api/records/<id>/review` | 人工复检回填 `{"result":"pass\|fail"}` |
| `GET /api/stats` | 质检统计（总数/OK/NG/待复检/缺陷率/复检分布） |
| `GET /api/image/<id>` | 历史检测图像 |

判定规则（`backend/service.py::decide_verdict`）：`NG`=两路同判缺陷；`REVIEW`=仅一路判缺陷（分歧→人工复核）；`OK`=两路均正常。

```bash
curl -F "image=@data/processed/images/kos01_Part5.jpg" http://127.0.0.1:5000/api/predict
```

## 自动化测试

```bash
python -m pytest tests -q     # 14 passed
```

覆盖：数据划分/命名干净性、letterbox 几何、归一化、特征维度、CNN 推理形状、三态判定、API 全流程（上传/历史/统计/复检/异常）。

## 技术方向覆盖（答辩用）

| 技术方向 | 落地点 | 系统中的作用 |
|---|---|---|
| 深度学习 / CNN 图像分类 | `ml/model.py`、`train.py` | 主判定器（高灵敏检出缺陷） |
| 模式识别 / 传统特征工程 | `ml/features.py`、`train_ml.py` | 分带 LBP/纹理特征 + RF 对照路（高特异）与 CNN 双证据融合 |
| 数据预处理与增强 | `preprocess.py`、`ml/dataset.py` | letterbox 保形缩放、归一化、分层划分、在线增强 |
| 图像/质检业务逻辑 | `backend/` | 双证据裁决、人工复检闭环、SQLite 历史与统计 |

## 文档
- `docs/需求规格说明书.md`：需求、功能/非功能、接口、验收标准（含「作者确认项」）
- `docs/设计报告撰写大纲.md`：报告章节与材料索引（正文建议自行撰写）
- `process/AI协作过程档案.md`：vibe coding 过程档案（阶段/决策/纠错案例）
- `制造智能技术课程设计任务书`：课程要求原文

## 说明与致谢

- 本项目使用 Vibe Coding 方法开发；数据修复、双模型对照、双证据融合等设计决策均保留在 git 提交历史与设计报告中。
- 早期预处理脚本存在"把标注掩码当图入库、掩码文件重名覆盖"两处 BUG，经审计发现并修复（见 `scripts/preprocess.py` 头注释），作为 AI 使用披露/纠错案例记录。
- 数据集版权归原作者所有（CC BY-NC-SA 4.0），仅用于课程学习。
