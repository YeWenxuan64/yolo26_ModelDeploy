import os
import sys
import argparse
from pathlib import Path

import torch
import onnx
import onnxslim
from onnxslim.utils import summarize_model, print_model_info_as_table




current_dir = Path(__file__).resolve().parent # 获取当前脚本所在目录的绝对路径
project_root = current_dir / 'models_convert/original/ultralytics'


# 定义上下文管理器以安全地更改目录
class temporary_sys_path:
    def __init__(self, new_path: str):
        self.new_path = str(new_path)
        
    def __enter__(self):
        sys.path.insert(0, self.new_path)
        return self
        
    def __exit__(self, etype, value, traceback):
        if self.new_path in sys.path:
            sys.path.remove(self.new_path)

class temporary_chdir:
    def __init__(self, new_path):
        self.new_path = str(new_path)
        self.saved_path = None
        
    def __enter__(self):
        self.saved_path = os.getcwd() # 保存进入前的当前目录
        os.chdir(self.new_path)       # 切换到新目录
        
    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)     # 无论代码块是否报错，都恢复原来的目录


sys.path.append(str(current_dir))
sys.path.append(str(project_root))
with temporary_sys_path(current_dir):
    from models_convert.original.ultralytics.ultralytics import YOLO
    from models_convert.original.ultralytics.ultralytics.utils.export.engine import torch2onnx


# ================= 导出参数 =================
IMG_SIZE = [320, 640]       # (h, w) 输入尺寸, 需为 32 的倍数
BATCH = 1
ONNX_OPSET = 13

yolo_pth_path = str(current_dir / 'models_convert/original/yolo26s.pt')
yolo_pose_pth_path = str(current_dir / 'models_convert/original/yolo26s-pose.pt')
yolo_raw_onnx_path = str(current_dir / 'models_convert/original/yolo26s_from_pytorch.onnx')
yolo_pose_raw_onnx_path = str(current_dir / 'models_convert/original/yolo26s-pose_from_pytorch.onnx')


# ================= PyTorch 直出路线 =================
class YOLO26RawExport(torch.nn.Module):
    """YOLO26 导出包装: 只出各尺度原始分支, 解码交给端侧 CPU 后处理。

    理由: 官方 end2end 图连带的 anchors 解码与 topk 索引还原会导出 Mod / TopK / Gather /
    Expand / Tile 等节点, NPU 不支持或效率极低; 这里不调 Detect.forward, 这些节点根本不被追踪。
    附带好处: sigmoid 写在 Python 侧, 导出图呈 Conv+Sigmoid 相邻, 工具链可直接融合。

    输出顺序固定为 box P3/P4/P5 -> cls P3/P4/P5 -> (pose) kpts P3/P4/P5。
    """

    def __init__(self, net: torch.nn.Module, head: torch.nn.Module):
        """Args:
            net: 末层 head 已被替换为 Identity 的检测模型, 前向返回 head 的多尺度输入列表。
            head: 原始 Detect / Pose26 模块, 提供 one2one 分支权重与 nl。
        """
        super().__init__()
        self.net = net
        self.head = head
        heads = head.one2one                  # fuse() 已删除 one2many 分支, 只剩 one2one
        self.box_head = heads["box_head"]
        self.cls_head = heads["cls_head"]
        self.pose_head = heads.get("pose_head")
        self.kpts_head = heads.get("kpts_head")

    def forward(self, image: torch.Tensor):
        feats = self.net(image)               # [P3, P4, P5]
        nl = self.head.nl
        outputs = [self.box_head[i](feats[i]) for i in range(nl)]
        outputs += [torch.sigmoid(self.cls_head[i](feats[i])) for i in range(nl)]
        if self.kpts_head is not None:        # pose: cv4 取特征 -> cv4_kpts 出 nk = 17*3 通道
            kpt_feats = [self.pose_head[i](feats[i]) for i in range(nl)]
            outputs += [self.kpts_head[i](kpt_feats[i]) for i in range(nl)]
        return tuple(outputs)


def build_raw_export(pth_path: str) -> YOLO26RawExport:
    """加载权重并按官方 export 的前置流程融合模型, 组装成 YOLO26RawExport。"""
    net:torch.nn.Module = YOLO(pth_path).model
    for p in net.parameters():
        p.requires_grad = False
    net.eval().float()
    net = net.fuse(verbose=False)             # Conv+BN 融合; end2end 时删除 one2many 分支
    for m in net.modules():                   # 与官方 export 一致: C2f/C3k2 改用 split 前向, 图中 Slice 变 Split
        if hasattr(m, "forward_split"):
            m.forward = m.forward_split

    head = net.model[-1]
    # head 换成 Identity 后, _predict_once 的 m.f 多输入组装逻辑原样保留, 前向即得多尺度特征列表
    identity = torch.nn.Identity()
    identity.f, identity.i = head.f, head.i
    net.model[-1] = identity

    return YOLO26RawExport(net, head)


def export_raw(yolo_type: str = "yolo") -> str:
    """导出原始分支图: backbone + neck + 各尺度 Conv 分支, 无解码/后处理节点。"""
    if yolo_type == "yolo":
        pth_path, raw_path = yolo_pth_path, yolo_raw_onnx_path
    elif yolo_type == "yolo-pose":
        pth_path, raw_path = yolo_pose_pth_path, yolo_pose_raw_onnx_path
    else:
        raise ValueError("yolo_type must be 'yolo' or 'yolo-pose'")
    print(f"Exporting raw model: {yolo_type}")

    wrapper = build_raw_export(pth_path)
    nl = wrapper.head.nl
    output_names = [f"box_p{i + 3}" for i in range(nl)] + [f"cls_p{i + 3}" for i in range(nl)]
    if wrapper.kpts_head is not None:
        output_names += [f"kpts_p{i + 3}" for i in range(nl)]

    im = torch.zeros(BATCH, 3, *IMG_SIZE)
    with temporary_chdir(current_dir):
        torch2onnx(wrapper, im, raw_path, opset=ONNX_OPSET,
                   input_names=["images"], output_names=output_names, dynamic=None)
    print(f"Raw export saved: {raw_path}")

    return raw_path


def simplify(onnx_path: str) -> str:
    """onnxslim 精简 + shape inference + checker, 输出按输入形状命名的最终 ONNX。"""
    onnx_model = onnx.load_model(onnx_path)
    original_info = summarize_model(onnx_model, os.path.basename(onnx_path))

    onnx_model = onnxslim.slim(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model, check_type=True, strict_mode=True)
    onnx.checker.check_model(onnx_model, full_check=True)

    images_info = next(i for i in onnx_model.graph.input if i.name == "images")
    shape_str = ",".join(str(d.dim_value) for d in images_info.type.tensor_type.shape.dim)
    stem = Path(onnx_path).stem.replace("_from_pytorch", "")
    output_path = str(current_dir / 'models_convert/onnx' / f"{stem}_[{shape_str}].onnx")

    onnx.save_model(onnx_model, output_path)

    slimmed_info = summarize_model(onnx_model, os.path.basename(output_path))
    print_model_info_as_table([original_info, slimmed_info])
    print(f"Simplified model saved: {output_path}")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO26 model exporter')

    parser.add_argument(
        '--yolo_type', 
        type=str, 
        default='yolo',
        choices=['yolo', 'yolo-pose'],
        required=False,
        help='Type of YOLO model to process (default: yolo)'
    )

    yolo_type = parser.parse_args().yolo_type

    # yolo_type = "yolo"
    # yolo_type = "yolo-pose"

    simplify(export_raw(yolo_type))