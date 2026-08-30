# course-design## 数据来源与预处理

### 数据来源
- 使用 **KolektorSDD** 工业缺陷检测数据集。
- 官网链接：https://www.vicos.si/resources/kolektorsdd/
- 原始数据存放于 `/data/raw/KolektorSDD/`。

### 数据预处理
运行 `scripts/preprocess.py` 完成：
- 统一尺寸为 500×500
- 从掩码提取边界框 (YOLO 格式)
- 按 70%/15%/15% 划分训练/验证/测试集
- 输出到 `/data/processed/` 和 `/data/splits/`

预处理后的文件：
- `/data/processed/images/` —— 统一尺寸的图像
- `/data/processed/masks/` —— 对应的二值掩码
- `/data/processed/annotations.csv` —— 所有边界框标注
- `/data/splits/train.txt`, `val.txt`, `test.txt` —— 数据集划分
