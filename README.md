# 形状与颜色检测

几何形状识别项目：自动生成**合成数据集**，用 PyTorch 训练形状分类模型，并提供 PyTorch 与 ONNX Runtime 两种推理方式；同时包含基于 OpenCV 传统视觉（HSV 阈值 + 轮廓分析）的形状/颜色检测方案。

Shape and color detection with synthetic data: an OpenCV HSV-and-contour pipeline, plus a PyTorch CNN with ONNX Runtime inference.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)

## 📷 效果展示

仓库里目前没有已提交的检测结果图。本段是占位，等真实输出放进 `docs/images/` 后再换成图片。

| 想展示的内容 | 建议路径 | 从哪里来 |
|---|---|---|
| 合成形状样本（圆 / 三角 / 矩形） | `docs/images/dataset-sample.png` | `python generate_dataset.py` 之后，从本地 `images/train/` 里选图 |
| 按颜色抠出的轮廓 | `docs/images/color-masks.png` | `python solve.py` 写到本地 `outputs/`（读取仓库外的 `./data/d.png` 与 `shape_model.onnx`） |
| ONNX 推理标注 | `docs/images/onnx-output.png` | `python onnxinf.py` 生成的本地 `output.png` |

只提交这几张示例。`images/` 数据集和 `.pth` / `.onnx` 权重已在 `.gitignore` 中排除。

## ✨ 特性

- **合成数据**：`generate_dataset.py` 程序化生成形状图片与标注，无需人工标注
- **双方案对比**：传统 CV（OpenCV 轮廓）与深度学习（CNN 分类）
- **双推理引擎**：PyTorch 原生推理 与 ONNX Runtime 跨框架推理
- 覆盖圆形、三角形、矩形等几何形状的检测与分类

## 🗂 项目结构

```
.
├── generate_dataset.py   # 合成数据集生成（形状渲染 + 自动标注）
├── solve.py              # OpenCV 传统方案：HSV 颜色分割 + 轮廓检测
├── solve.ipynb           # 传统方案 Notebook（含可视化）
├── train.ipynb           # CNN 形状分类模型训练
├── inference.py          # PyTorch 模型推理
├── onnxinf.py            # ONNX Runtime 推理
├── shape_model*.pth      # PyTorch 权重（不入库）
├── shape_model*.onnx     # ONNX 模型（不入库）
└── images/               # 生成的数据集图片（不入库）
```

## 🚀 快速开始

```bash
pip install -r requirements.txt

# 1. 生成合成数据集
python generate_dataset.py

# 2a. 传统视觉方案（无需训练）
python solve.py

# 2b. 训练深度学习模型
jupyter notebook train.ipynb

# 3. 推理
python inference.py       # PyTorch 引擎
python onnxinf.py         # ONNX Runtime 引擎
```

## 🧪 技术要点

**OpenCV 传统方案（solve.py）：**

1. HSV 颜色空间阈值分割（支持红/绿/蓝/黄/橙/紫/粉/青/黑等）
2. `findContours()` 提取轮廓
3. 轮廓顶点数 / 多边形拟合判定形状（圆/三角/矩形）
4. 轮廓矩（moments）计算目标中心

**深度学习方案：**

1. 合成数据生成 → CNN 分类训练
2. 导出 ONNX，使用 onnxruntime 部署（不依赖 PyTorch）

## 📦 依赖

PyTorch、torchvision、OpenCV、ONNX Runtime、NumPy、Matplotlib、Pillow —— 详见 [requirements.txt](requirements.txt)

## 📝 说明

数据集图片（`images/`，约 117MB）与模型权重（`.pth` / `.onnx`）不包含在仓库中（`.gitignore` 已排除），运行生成/训练脚本即可复现。

## 🔗 相关项目

- [opencv-learn](https://github.com/fishlmg/opencv-learn) — OpenCV 基础练习：模糊、轮廓、HSV 分割、模板匹配
- [gd-vision-competition](https://github.com/fishlmg/gd-vision-competition) — 竞赛向 HSV 颜色检测与二维码识别
- [cat-dog-classification](https://github.com/fishlmg/cat-dog-classification) — PyTorch CNN 训练、ONNX 导出与 OpenCV 部署
- [pytorch-learn](https://github.com/fishlmg/pytorch-learn) — PyTorch 笔记：CNN、迁移学习与训练技巧

## 📄 许可证

[MIT License](LICENSE)

---

作者：[fishlmg](https://github.com/fishlmg)
