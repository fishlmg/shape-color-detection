# 形状与颜色检测

基于 OpenCV + PyTorch 的几何形状识别项目：自动生成合成数据集，训练形状分类模型，并支持 PyTorch 与 ONNX Runtime 两种推理方式。

## 文件结构

```
├── generate_dataset.py   # 合成数据集生成（形状 + 标注）
├── solve.py              # OpenCV 颜色/形状检测（HSV 阈值 + 轮廓）
├── solve.ipynb           # 检测流程 Notebook
├── train.ipynb           # 形状分类模型训练
├── inference.py          # PyTorch 推理
├── onnxinf.py            # ONNX Runtime 推理
├── shape_model*.pth      # 训练权重（不入库）
├── shape_model*.onnx     # 导出的 ONNX 模型（不入库）
└── images/               # 生成的数据集图片（不入库）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 生成数据集
python generate_dataset.py

# 2. 训练模型
jupyter notebook train.ipynb

# 3. 推理
python inference.py      # PyTorch
python onnxinf.py        # ONNX Runtime
```

## 说明

数据集图片（`images/`）与模型权重不包含在仓库中，`.gitignore` 已排除，本地保留。
