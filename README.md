# Yolo26 ModelDeploy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Rockchip%20|%20Qualcomm-orange)
![Model](https://img.shields.io/badge/Model-YOLO26s%20(Detect%20+%20Pose)-red)
![Quantization](https://img.shields.io/badge/Quant-INT8%20|%20FP16-lightgrey)

## 📖 概述

本模块聚焦 YOLO26 的 **目标检测 (Detect)** 和 **姿态估计 (Pose)** 两种任务，提供从 PyTorch 到边缘端 NPU (Rockchip NPU / Qualcomm HTP) 的完整转换与推理流程。

> 同时本项目也是本小姐🍃的项目 [Focus-Finder](https://github.com/YeWenxuan64/Focus-Finder) 的模型部署部分喵~

---

### 🧬 关于 YOLO26

**YOLO26** 是基于 [Ultralytics](https://docs.ultralytics.com/) 框架训练的轻量级目标检测与姿态估计模型，沿用 YOLO 系列经典的 anchor-free 检测头设计。本模块选用 **YOLO26s**（小模型），在精度与推理速度之间取得良好平衡，专为边缘端 NPU 部署优化。

YOLO26 支持的视觉任务：

| 任务 | 说明 | 输出 |
|------|------|------|
| **Detect（目标检测）** | 识别并定位图像中的物体 | 边界框 + 类别（COCO 80 类） |
| **Pose（姿态估计）** | 检测人体关键点 | 边界框 + 17 个关键点坐标 |

> **注：**
> YOLO 系列历经多个团队迭代：YOLOv1–v4 由 Joseph Redmon 团队开创；YOLOv6 由美团视觉智能部提出；YOLOv7/v9 由 Chien-Yao Wang 团队研发；YOLOv8/11/12 等近期版本由 Ultralytics 主导发布。YOLO 的发展是整个社区共同努力的成果。
> 本模块的 YOLO26s 基于标准 Ultralytics 框架训练，ONNX 导出后经过模型裁剪等优化，以适配 Rockchip / Qualcomm NPU。

---

### ✨ 功能亮点

- **双任务支持** — 目标检测（COCO 80 类）+ 姿态估计（COCO 17 关键点），同一套工具链
- **全链路自动化** — 一条命令完成 `.pt` → ONNX → RKNN / QNN，`set_config()` 自动管理配置，无需手动编辑 yaml
- **双平台部署** — Rockchip NPU（RK3588/RK3576）+ Qualcomm HTP（QCS6490），INT8 量化
- **Sigmoid 重排优化** — `move_sigmoid_into_concat_branches()` 将 Concat 后的 Sigmoid 前移到各分支 Conv 后，便于 NPU 工具链融合 Conv+Sigmoid
- **ONNX 模型裁剪** — `trim_model_to_outputs()` 截断多余后处理节点，精简计算图
- **onnxslim 瘦身** — 自动移除冗余算子、折叠常量，减小模型体积
- **输出顺序自适应** — `identify_output_order()` 按张量尺寸自动匹配模型输出，兼容不同推理后端
- **Anchor-based 解码** — `build_anchor_grids()` 预计算三检测头锚点网格，高效还原边界框
- **内置实时演示** — 开箱即用的视频流推理 + `timeit` FPS 统计，按 `q` 退出
- **易于集成** — 提供 Python API 接口，三行代码即可嵌入自有项目

---

**上游项目：** 
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) — Ultralytics YOLO 官方框架
- [Edge_ModelDeploy](https://github.com/YeWenxuan64/Edge_ModelDeploy) — 模型转换工具链


**本子模块：**`Yolo26_ModelDeploy` 聚焦 YOLO26 系列模型（Detect + Pose）的完整部署链路：

| 阶段 | 说明 | 脚本 |
|------|------|------|
| **PyTorch → ONNX** | 利用 ultralytics 导出 ONNX，Sigmoid 重排 + 模型裁剪 + onnxslim 瘦身 | [`Yolo26_pytorch2onnx.py`](./Yolo26_pytorch2onnx.py) |
| **ONNX → RKNN** | 使用父项目 `utilities/onnx_to_rknn.py`，INT8 量化 + FlashAttention 优化，部署至 Rockchip NPU | [`Yolo26_onnx2rknn.py`](./Yolo26_onnx2rknn.py) |
| **ONNX → QNN** | 使用父项目 `utilities/onnx_to_qnn.py`，INT8 量化（SQNR + Entropy），部署至 Qualcomm HTP | [`Yolo26_onnx2qnn.py`](./Yolo26_onnx2qnn.py) |
| **推理** | 加载 ONNX / RKNN / QNN 模型，实时检测 + 可视化 | [`yolo26_main.py`](./yolo26_main.py) / [`yolo26_pose_main.py`](./yolo26_pose_main.py) |


## 🏗️ 项目结构

```
Edge_ModelDeploy/                   # 父项目根目录
├── datasets/                        # 量化校准数据集
├── utilities/                       # 父项目工具链
├── README.md                        # 父项目 README
├── README_TOOLUSE.md
├── requirements.txt                 # 父项目 Python 依赖
├── LICENSE
├── ...
└── Yolo26_ModelDeploy/              # 本模块根目录
    ├── Yolo26_pytorch2onnx.py       # PyTorch → ONNX 导出 + 优化
    ├── Yolo26_onnx2rknn.py          # ONNX → RKNN（使用父项目 utilities/onnx_to_rknn.py）
    ├── Yolo26_onnx2qnn.py           # ONNX → QNN（使用父项目 utilities/onnx_to_qnn.py）
    ├── yolo26_main.py               # 目标检测推理 + 可视化
    ├── yolo26_pose_main.py          # 姿态估计推理 + 可视化
    ├── yolo26_pytorch_test.py       # PyTorch 模型快速验证
    ├── models_convert/
    │   ├── original/
    │   │   ├── ... .pt              # PyTorch 权重（会在转换时由 ultralytics 自动下载）
    │   │   └── ultralytics/         # ultralytics/ultralytics submodule
    │   ├── onnx/
    │   │   └── ...   # 导出后的 ONNX
    │   ├── rknn/
    │   │   └── ...   # RKNN 模型
    │   └── qnn/
    │       └── ...   # QNN 模型
    ├── README.md
    └── LICENSE
```


## 📦 部署流程

本模块是 [Edge_ModelDeploy](https://github.com/YeWenxuan64/Edge_ModelDeploy) 的子模块，**必须先部署父项目**。

### 1. 克隆父项目
```bash
git clone https://github.com/YeWenxuan64/Edge_ModelDeploy.git
cd Edge_ModelDeploy
```

### 1.1 安装依赖
按照父项目 [Edge_ModelDeploy README.md](../README.md) 的指示，进行**依赖安装与量化校准数据集的准备**

### 2. 递归克隆本模块

```bash
git clone --recurse-submodules https://github.com/YeWenxuan64/Yolo26_ModelDeploy.git
```

> 会递归克隆 `ultralytics` 子模块（位于 `Yolo26_ModelDeploy/models_convert/original/ultralytics`）<br>
> 如果 submodule 未拉取，可手动拉取：
> ```bash
> cd Yolo26_ModelDeploy
> git submodule update --init --recursive
> ```
> 或查看 [models_convert/original/README.md](./models_convert/original/README.md)


## 🔄 模型转换

转换流程：**`.pt` → ONNX（导出 + 优化） → RKNN / QNN**

所有转换脚本支持 `--yolo_type` 参数切换模型类型：

| 参数值 | 模型 | 检测内容 |
|--------|------|----------|
| `yolo` | yolo26s | 目标检测（COCO 80 类） |
| `yolo-pose` | yolo26s-pose | 姿态估计（COCO 17 关键点） |

### 步骤 1: PyTorch → ONNX

```bash
cd Yolo26_ModelDeploy
python Yolo26_pytorch2onnx.py --yolo_type yolo
# or python Yolo26_pytorch2onnx.py --yolo_type yolo-pose
```

**可修改的参数** — 编辑 `Yolo26_pytorch2onnx.py` 中 `export()` 函数内的 `set_config()` 调用：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `imgsz` | `[320, 640]` | 输入图像尺寸 `[H, W]` |
| `batch` | `1` | 批次大小 |
| `max_det` | `256` | 每图最大检测数 |

> `set_config()` 从 `ultralytics/cfg/default.yaml` 加载默认配置并动态覆写，**无需手动编辑 config yaml 文件**。

**ONNX 后处理优化** — `modify()` 函数自动执行以下步骤：
1. **Sigmoid 重排** — `move_sigmoid_into_concat_branches()` 将 `/model.23/Concat_1` 后的 Sigmoid 移到各分支 Conv 后
2. **模型裁剪** — `trim_model_to_outputs()` 只保留 bbox + class（detect）或 bbox + class + kpts（pose）输出，截断冗余后处理
3. **onnxslim 瘦身** — 移除冗余算子、折叠常量

输出：`models_convert/onnx/yolo26s_[1,3,320,640].onnx`


### 步骤 2.1: ONNX → RKNN

```bash
python Yolo26_onnx2rknn.py --yolo_type yolo
# or python Yolo26_onnx2rknn.py --yolo_type yolo-pose
```

输出：`models_convert/rknn/yolo26s_i8[1,320,640,3].rknn`

> RKNN 转换启用了 `extra_optimize(flash_attantion=True)` 以优化 Attention 计算。

**可配置参数**：修改脚本中 `OnnxToRKNN` 对象方法的参数<br>
详见父项目 [Edge_ModelDeploy README_TOOLUSE.md](https://github.com/YeWenxuan64/Edge_ModelDeploy/blob/main/README_TOOLUSE.md)


### 步骤 2.2: ONNX → QNN

```bash
python Yolo26_onnx2qnn.py --yolo_type yolo
# or python Yolo26_onnx2qnn.py --yolo_type yolo-pose
```

输出：`models_convert/qnn/yolo26s_i8[1,320,640,3].bin`

> QNN 量化策略：`param_quant_method='sqnr'`（参数量化）+ `act_quant_method='entropy'`（激活量化）。

**可配置参数**：修改脚本中 `OnnxToQNN` 对象方法的参数<br>
详见父项目 [Edge_ModelDeploy README_TOOLUSE.md](https://github.com/YeWenxuan64/Edge_ModelDeploy/blob/main/README_TOOLUSE.md)

## 🧠 推理

推理后端依赖 [Edge_Inferencer](https://github.com/YeWenxuan64/Edge_Inferencer) 模块，支持 ONNX Runtime 和 NPU 推理切换。<br>
推理后端的部署与使用，详见 [Edge_Inferencer README.md](https://github.com/YeWenxuan64/Edge_Inferencer/blob/main/README.md)

两个推理入口，分别对应 Detect 和 Pose 任务。内置实时视频流演示，按 `q` 退出

### 🎯 目标检测

```bash
python yolo26_main.py
```

**类：** `Yolo26`

**`__init__` 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | — | 模型文件路径 |
| `model_size` | `tuple` | `(640, 320)` | 模型输入尺寸 `(width, height)` |
| `need_preprocess` | `bool` | `False` | 是否对输入做 `/255` 归一化 |
| `conf_thresh` | `float` | `0.25` | 置信度阈值 |
| `cores` | `tuple` | `(0,)` | NPU 绑核或并发数，如 `(0,1,2)` |
| `mult_task` | `bool` | `False` | 是否启用多任务并发推理 |

**`detect()` 方法：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `color_image` | `np.ndarray` | — | 输入 RGB 图像 `(H, W, 3)` |
| `block` | `bool` | `True` | 是否阻塞等待推理结果 |
| `scale` | `tuple` | `(1.0, 1.0)` | 缩放比例，用于坐标还原 |
| `offset` | `tuple` | `(0, 0)` | 偏移量 `(x, y)`，用于坐标还原 |

**返回值：** `np.ndarray (N, 6)` → `[[x1, y1, x2, y2, class_id, score]...]` 或 `None`

**`release()` 方法：** 释放 NPU 资源

- **可视化：** `draw_yolo()`，绘制边界框 + 类别标签 + 置信度
- **预处理：** `resize_image()` 等比缩放 + 居中 padding（黑色填充）
- **后处理：** Anchor-based 解码 `bbox_anchor()` + `identify_output_order()` 输出顺序自适应


### 🦴 姿态估计

```bash
python yolo26_pose_main.py
```

**类：** `Yolo26Pose`

**`__init__` 参数：**

同 `Yolo26` 类

**`detect()` 方法：**

参数同 `Yolo26` 类

**返回值：** `np.ndarray (N, 57)` → `[[x1, y1, x2, y2, class_id, score, kpt_x, kpt_y, kpt_conf, ...]...]` 或 `None`

> 返回的 57 个值 = 6（bbox+class+score）+ 51（17 关键点 × 3 值 x/y/conf）

**`release()` 方法：** 释放 NPU 资源

- **可视化：** `draw_yolo_pose()`，绘制边界框 + 17 关键点圆圈 + 19 条骨骼连线
- **预处理：** `resize_image()` 等比缩放 + 居中 padding（黑色填充）
- **后处理：** Anchor-based 解码 `bbox_anchor()` + `kpts_anchor()` + `identify_output_order()` 输出顺序自适应


### 🔗 尝试集成到自己的项目

```python
from yolo26_main import Yolo26, draw_yolo, resize_image
# 或
# from yolo26_pose_main import Yolo26Pose, draw_yolo_pose, resize_image

# 初始化
model = Yolo26(model_path='path/to/model.onnx', conf_thresh=0.25, need_preprocess=True, cores=(0,))

# 预处理
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
resized, scale, offset = resize_image(rgb, (640, 320))

# 推理
result = model.detect(resized, scale=(scale, scale), offset=offset)

# 可视化
if result is not None:
    frame = draw_yolo(frame, result)
```


### 🧪 PyTorch 模型快速验证

```bash
python yolo26_pytorch_test.py
```

使用原始 `.pt` 权重在视频流上运行推理，用于快速验证模型效果。

---

### ⚡ 性能数据

> ⚠️ 以下数据为参考值，实际性能取决于具体硬件型号、驱动版本、量化精度及输入分辨率。<br>
> - 测试前均设置 CPU 频率为用户最高可设置频率
> - 推理时间为模型**在 NPU 的**推理时间
> - 后处理时间为将推理结果处理为可用数据（如 bbox、class、score 等）的时间
> - 推理时间和后处理时间**不包括**预处理时间，如缩放图像、bgr 转 rgb、nwc 转 nhwc、归一化等，**也不包括**拷贝数据时间
> - 并发推理时间**包含**推理时间、后处理时间以及拷贝数据时间，但不包括预处理时间

| 平台 | 芯片 | 模型 | 精度 | 输入尺寸 (H×W) | 推理时间 | 后处理时间 | 并发推理 |
|------|------|------|------|----------------|----------|------------|------|
| Rockchip NPU | RK3588 | yolo26s | INT8 | 320×640 | — | — | — |
| Rockchip NPU | RK3588 | yolo26s-pose | INT8 | 320×640 | — | — | — |
| Qualcomm HTP | QCS6490 | yolo26s | INT8 | 320×640 | — | — | — |
| Qualcomm HTP | QCS6490 | yolo26s-pose | INT8 | 320×640 | — | — | — |

> 💡 **如何获取实际数据：** 
> - 运行 `yolo26_main.py` / `yolo26_pose_main.py` 时终端会实时输出 FPS 和平均耗时，记录稳定后的数值即可
> - ai_inferencer 模块提供了 `timeit` 装饰器，可用于测量任意函数的耗时
> - 使用 `VizTracer` 工具 Tracer 要运行的文件，之后浏览统计并统计结果

---

| 平台          | 芯片    | 模型         | 精度  | 输入尺寸(HxW) | 推理时间 | 后处理时间 | 并发推理 | 备注 |
|--------------|---------|--------------|------|---------|------|------|-----|-----|
| Rockchip NPU | RK3588  | yolo26s      | INT8 | 320×640 | —  | —  | —  | 待实测 |
| Rockchip NPU | RK3588  | yolo26s-pose | INT8 | 320×640 | —  | —  | —  | 待实测 |
| Qualcomm HTP | QCS6490 | yolo26s      | INT8 | 320×640 | —  | —  | —  | 待实测 |
| Qualcomm HTP | QCS6490 | yolo26s-pose | INT8 | 320×640 | —  | —  | —  | 待实测 |


> 💡 **如何获取实际数据：** 
> - 运行 `yolo26_main.py` / `yolo26_pose_main.py` 时终端会实时输出 FPS，记录稳定后的数值即可。
> - 使用 `VizTracer` 工具 Tracer 要运行的文件，之后浏览统计并统计结果

---


## License

MIT License — Copyright (c) 2026 叶文轩

各子模块的模型、代码，以及数据集遵循其原始 License。<br>
ultralytics/ultralytics仓库的许可证：
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)