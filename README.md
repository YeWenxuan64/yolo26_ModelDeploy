# Yolo26 ModelDeploy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Rockchip%20|%20Qualcomm-orange)
![Model](https://img.shields.io/badge/Model-YOLO26s%20(Detect%20+%20Pose)-red)
![Quantization](https://img.shields.io/badge/Quant-INT8%20|%20FP16-lightgrey)

## 概述
YOLO26 目标检测 (Detect) 和姿态估计 (Pose) 模型部署模块<br>
从 PyTorch 到边缘端 NPU (Rockchip NPU / Qualcomm HTP) 的完整转换与推理流程

---

### ✨ 功能亮点

- **双任务支持** — 目标检测（COCO 80 类）+ 姿态估计（COCO 17 关键点），同一套工具链
- **全链路自动化** — 一条命令完成 `.pt` → ONNX → RKNN / QNN，无需手动中间步骤
- **双平台部署** — Rockchip NPU（RK3588/RK3576）+ Qualcomm HTP（QCS6490），INT8 量化
- **灵活配置** — 输入尺寸、批次大小、检测上限均可调，适配不同硬件约束
- **内置实时演示** — 开箱即用的视频流推理 + FPS 统计，按 `q` 退出
- **易于集成** — 提供 Python API 接口，三行代码即可嵌入自有项目

---

**上游项目：** [Focus-Finder_ModelDeploy](https://github.com/YeWenxuan64/Focus-Finder_ModelDeploy) — 模型转换工具链

**本子模块：**`Yolo26_ModelDeploy`聚焦 YOLO26 系列模型（Detect + Pose）的完整部署链路：

| 阶段 | 说明 | 脚本 |
|------|------|------|
| **PyTorch → ONNX** | 利用 ultralytics 导出 ONNX | [`Yolo26_pytorch2onnx.py`](./Yolo26_pytorch2onnx.py) |
| **ONNX → RKNN** | 使用父项目 `utilities/onnx_to_rknn.py`，INT8 量化部署至 Rockchip NPU | [`Yolo26_onnx2rknn.py`](./Yolo26_onnx2rknn.py) |
| **ONNX → QNN** | 使用父项目 `utilities/onnx_to_qnn.py`，INT8 量化部署至 Qualcomm HTP | [`Yolo26_onnx2qnn.py`](./Yolo26_onnx2qnn.py) |
| **推理** | 加载 ONNX / RKNN / QNN 模型，实时检测 + 可视化 | [`yolo26_main.py`](./yolo26_main.py) / [`yolo26_pose_main.py`](./yolo26_pose_main.py) |

---


## 项目结构

```
Focus-Finder_ModelDeploy/        # 父项目根目录
├── datasets/                    # 量化校准数据集
├── utilities/                   # 父项目工具链
├── README.md                    # 父项目 README
├── README_TOOLUSE.md            
├── requirements.txt             # 父项目python依赖
├── LICENSE
├── ...
└── Yolo26_ModelDeploy/          # 本模块根目录
    ├── Yolo26_pytorch2onnx.py   # PyTorch → ONNX 导出
    ├── Yolo26_onnx2rknn.py      # ONNX → RKNN（使用父项目 utilities/onnx_to_rknn.py）
    ├── Yolo26_onnx2qnn.py       # ONNX → QNN（使用父项目 utilities/onnx_to_qnn.py）
    ├── yolo26_main.py           # 目标检测推理 + 可视化
    ├── yolo26_pose_main.py      # 姿态估计推理 + 可视化
    ├── models_convert/
    │   ├── original/
    │   │   ├── ... .pt          # PyTorch 权重（会在转换时由ultralytics自动下载）
    │   │   └── ultralytics/     # ultralytics submodule
    │   ├── onnx/
    │   │   └── ...   # 导出后的 ONNX
    │   ├── rknn/
    │   │   └── ...   # RKNN 模型
    │   └── qnn/
    │       └── ...   # QNN 模型
    ├── README.md
    └── LICENSE
```


## 部署流程
本模块是 [Focus-Finder_ModelDeploy](https://github.com/YeWenxuan64/Focus-Finder_ModelDeploy) 的子模块，**必须先部署父项目**。

### 1. 克隆父项目
```bash
git clone https://github.com/YeWenxuan64/Focus-Finder_ModelDeploy.git
cd Focus-Finder_ModelDeploy
```

### 1.5. 安装依赖
按照父项目 [Focus-Finder_ModelDeploy README.md](../README.md) 的指示，进行**依赖安装与量化校准数据集的准备**


### 2. 克隆本模块

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


## 模型转换

转换流程：**`.pt` → ONNX → RKNN / QNN**

转换脚本支持 `--yolo_type` 参数切换模型类型：

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

**可修改的参数**：
Yolo26_pytorch2onnx.py脚本中的 `set_config()`
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `imgsz` | `[320, 640]` | 输入图像尺寸 `[H, W]` |
| `batch` | `1` | 批次大小 |
| `max_det` | `256` | 每图最大检测数 |

输出：`models_convert/onnx/yolo26s_[1,3,320,640].onnx`


### 步骤 2.1: ONNX → RKNN

```bash
python Yolo26_onnx2rknn.py --yolo_type yolo
# or python Yolo26_onnx2rknn.py --yolo_type yolo-pose
```

**可配置参数**: 修改脚本中 `OnnxToRKNN` 对象方法的参数<br>
详见父项目 [Focus-Finder_ModelDeploy README_TOOLUSE.md](../README_TOOLUSE.md)


输出：`models_convert/rknn/yolo26s_i8[1,320,640,3].rknn`


### 步骤 2.2: ONNX → QNN

```bash
python Yolo26_onnx2qnn.py --yolo_type yolo
# or python Yolo26_onnx2qnn.py --yolo_type yolo-pose
```

**可配置参数**: 修改脚本中 `OnnxToQNN` 对象方法的参数<br>
详见父项目 [Focus-Finder_ModelDeploy README_TOOLUSE.md](../README_TOOLUSE.md)

输出：`models_convert/qnn/yolo26s_i8[1,320,640,3].bin`


## 推理
推理后端依赖 [Edge_Inferencer](https://github.com/YeWenxuan64/Edge_Inferencer) 模块，支持 ONNX Runtime 和 NPU 推理切换。<br>
推理后端的部署与使用，详见 [Edge_Inferencer README.md](https://github.com/YeWenxuan64/Edge_Inferencer/blob/main/README.md)


两个推理入口，分别对应 Detect 和 Pose 任务。内置实时视频流演示，按 `q` 退出。


### 目标检测

```bash
python yolo26_main.py
```

- **类：** `Yolo26`
- **输出：** `np.ndarray (N, 6)` → `[[x1, y1, x2, y2, class_id, score]...]`
- **可视化：** `draw_yolo()` — 绘制边界框 + 类别标签 + 置信度
- **可调参数：** `model_path`, `cores`, `conf_threshold`, `need_preprocess`


### 姿态估计

```bash
python yolo26_pose_main.py
```

- **类：** `Yolo26Pose`
- **输出：** `np.ndarray (N, 57)` → `[[x1, y1, x2, y2, class_id, score, kpt_x, kpt_y, kpt_conf, ...]...]`
- **可视化：** `draw_yolo_pose()` — 绘制边界框 + 17 关键点圆圈 + 19 条骨骼连线
- **可调参数：** `model_path`, `cores`, `conf_threshold`, `need_preprocess`


---

### 性能数据

> ⚠️ 以下数据为参考值，实际性能取决于具体硬件型号、驱动版本、量化精度及输入分辨率。<br>
> 推理时间为模型在NPU的推理时间，后处理时间为将推理结果处理为可用数据(如bbox，class，score等)的时间。<br>
> 不包括预处理时间，如缩放图像、bgr转rgb、nwc转nhwc、归一化等。

| 平台          | 芯片    | 模型         | 精度  | 输入尺寸(HxW) | 推理时间(ms) | 后处理时间(ms) | 备注 |
|--------------|---------|--------------|------|---------|------|-----------|-----|
| Rockchip NPU | RK3588  | yolo26s      | INT8 | 320×640 | — | — | 待实测 |
| Rockchip NPU | RK3588  | yolo26s-pose | INT8 | 320×640 | — | — | 待实测 |
| Qualcomm HTP | QCS6490 | yolo26s      | INT8 | 320×640 | — | — | 待实测 |
| Qualcomm HTP | QCS6490 | yolo26s-pose | INT8 | 320×640 | — | — | 待实测 |


> 💡 **如何获取实际数据：** 运行 `yolo26_main.py` / `yolo26_pose_main.py` 时终端会实时输出 FPS，记录稳定后的数值即可。

---


### 尝试集成到自己的项目

```python
from yolo26_main import Yolo26, draw_yolo, resize_image
# 或
# from yolo26_pose_main import Yolo26Pose, draw_yolo_pose, resize_image

# 初始化
model = Yolo26(model_path='path/to/model.onnx', cores=(0,), conf_threshold=0.25, need_preprocess=True)

# 预处理
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
resized, scale, offset = resize_image(rgb, (640, 320))

# 推理
result = model.yolo26_detect(resized)

# 后处理 & 可视化
if result is not None:
    draw_yolo(frame, result, scale, offset)

# 退出时释放
model.release()
```





## License

MIT License — Copyright (c) 2026 叶文轩

各子模块的模型、代码，以及数据集遵循其原始 License。<br>
ultralytics/ultralytics仓库的许可证：
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)