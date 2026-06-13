import os
import sys
import re
import yaml
import argparse
import pathlib
from pathlib import Path
from types import SimpleNamespace

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


def load_config(config_path):

    def yaml_load(file, append_filename=False):
        """
        Load YAML data from a file.

        Args:
            file (str, optional): File name. Default is 'data.yaml'.
            append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

        Returns:
            (dict): YAML data and file name.
        """
        assert pathlib.Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()  # string

            # Remove special characters
            if not s.isprintable():
                s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

            # Add YAML filename to dict and return
            data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
            if append_filename:
                data["yaml_file"] = str(file)
            return data

    class IterableSimpleNamespace(SimpleNamespace):
        """
        An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

        This class extends the SimpleNamespace class with additional methods for iteration, string representation,
        and attribute access. It is designed to be used as a convenient container for storing and accessing
        configuration parameters.

        Methods:
            __iter__: Returns an iterator of key-value pairs from the namespace's attributes.
            __str__: Returns a human-readable string representation of the object.
            __getattr__: Provides a custom attribute access error message with helpful information.
            get: Retrieves the value of a specified key, or a default value if the key doesn't exist.

        Examples:
            >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
            >>> for k, v in cfg:
            ...     print(f"{k}: {v}")
            a: 1
            b: 2
            c: 3
            >>> print(cfg)
            a=1
            b=2
            c=3
            >>> cfg.get("b")
            2
            >>> cfg.get("d", "default")
            'default'

        Notes:
            This class is particularly useful for storing configuration parameters in a more accessible
            and iterable format compared to a standard dictionary.
        """

        def __iter__(self):
            """Return an iterator of key-value pairs from the namespace's attributes."""
            return iter(vars(self).items())

        def __str__(self):
            """Return a human-readable string representation of the object."""
            return "\n".join(f"{k}={v}" for k, v in vars(self).items())

        def __getattr__(self, attr):
            """Custom attribute access error message with helpful information."""
            name = self.__class__.__name__
            raise AttributeError(
                f"""
                '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
                'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
                {config_path} with the latest version from
                https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
                """
            )

        def get(self, key, default=None):
            """Return the value of the specified key if it exists; otherwise, return the default value."""
            return getattr(self, key, default)

    # Default configuration
    config_dict = yaml_load(config_path)
    for k, v in config_dict.items():
        if isinstance(v, str) and v.lower() == "none":
            config_dict[k] = None
    config = IterableSimpleNamespace(**config_dict)

    return config



def set_config(config, task:str, model:str, imgsz:list[int,int], batch:int=1, max_det:int=300):
    """
    Set the configuration for the YOLO model.

    Args:
        task (str): The task for the YOLO model, e.g. "detect", "segment", "classify", "pose", "obb".
        model (str): The path to the pytorch model file.
        imgsz (list): The image size [h,w] for the model, e.g. [320, 640].
        batch (int): The batch size for the model.
        max_det (int): The maximum number of detections per image.
    """

    config.task = task          # (str) YOLO task, i.e. detect, segment, classify, pose, obb
    config.mode = "export"      # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark
    # Train settings
    config.model = model        # (str, optional) path to model file, i.e. yolov8n.pt or yolov8n.yaml
    config.batch = batch        # (int | float) batch size as int (e.g. 16), or float 0.0–1.0 for AutoBatch fraction of GPU memory
    config.imgsz = imgsz        # (int | list) train/val use int (square); predict/export may use [h,w]
    # Val/Test settings
    config.max_det = max_det    # (int) maximum number of detections per image
    # Export settings
    config.format = "onnx"      # (str) target format, e.g. torchscript|onnx|openvino|engine|coreml|saved_model|pb|tflite|edgetpu|tfjs|paddle|mnn|ncnn|imx|rknn|executorch|axelera
    config.simplify = False     # (bool) ONNX/engine only; run graph simplifier for cleaner ONNX before runtime conversion
    config.opset = 13           # (int, optional) ONNX/engine only; opset version for export; leave unset to use a tested default

    return config

def replace_mod(model:onnx.ModelProto, node_name:str|None=None) -> onnx.ModelProto:
    graph = model.graph
    if node_name is None:
        node_name = "/model.23/Mod"

    # 1. 查找目标 Mod 节点
    mod_node = None
    mod_idx = -1
    for idx, node in enumerate(graph.node):
        if node.op_type == 'Mod' or node.name == node_name:
            mod_node = node
            mod_idx = idx
            break

    if not mod_node:
        print(f"错误: 未找到 Mod 节点 (name={node_name})")
        return model

    print(f"找到节点: {mod_node.name} (index: {mod_idx})")
    
    in_a, in_b = mod_node.input[0], mod_node.input[1]
    out = mod_node.output[0]

    # 2. 构建替换逻辑: out = in_a - in_b * (in_a // in_b)
    # 注意: 假设 in_a 为非负数（YOLO 索引场景），ONNX 整数 Div 等同于 //
    div_out = f"{out}_div"
    mul_out = f"{out}_mul"

    div_node = onnx.helper.make_node('Div', [in_a, in_b], [div_out], name=f"{mod_node.name}_Div")
    mul_node = onnx.helper.make_node('Mul', [div_out, in_b], [mul_out], name=f"{mod_node.name}_Mul")
    sub_node = onnx.helper.make_node('Sub', [in_a, mul_out], [out], name=f"{mod_node.name}_Sub")

    # 3. 替换原节点
    graph.node.remove(mod_node)
    graph.node.insert(mod_idx, div_node)
    graph.node.insert(mod_idx + 1, mul_node)
    graph.node.insert(mod_idx + 2, sub_node)

    print(f"MOD 替换完成")

    return model




yolo_config_path = str(project_root / 'ultralytics/cfg/default.yaml')

yolo_onnx_path = str(current_dir / 'models_convert/original/yolo26s.onnx')
yolo_onnx_output_path = str(current_dir / 'models_convert/onnx/yolo26s_[1,3,320,640].onnx')

yolo_pose_onnx_path = str(current_dir / 'models_convert/original/yolo26s-pose.onnx')
yolo_pose_onnx_output_path = str(current_dir / 'models_convert/onnx/yolo26s-pose_[1,3,320,640].onnx')


def export(yolo_type:str="yolo"):
    if yolo_type == "yolo":
        pth_path = "./models_convert/original/yolo26s.pt"
        task = "detect"

    elif yolo_type == "yolo-pose":
        pth_path = "./models_convert/original/yolo26s-pose.pt"
        task = "pose"

    else:
        raise ValueError("yolo_type must be 'yolo', 'yolo-pose'")
    print(f"Exporting model: {yolo_type}")

    config = load_config(yolo_config_path)
    config = set_config(config, task=task, model=pth_path, imgsz=[320, 640], batch=1, max_det=256)

    with temporary_chdir(current_dir):
        model = YOLO(config.model)
        model.export(**vars(config))


def modify(yolo_type:str="yolo"):
    if yolo_type == "yolo":
        onnx_model_path = yolo_onnx_path
        onnx_model_output_path = yolo_onnx_output_path
    elif yolo_type == "yolo-pose":
        onnx_model_path = yolo_pose_onnx_path
        onnx_model_output_path = yolo_pose_onnx_output_path
    else:
        raise ValueError("yolo_type must be 'yolo' or 'yolo-pose'")
    print(f"Simpling model: {yolo_type}")

    onnx_model = onnx.load_model(onnx_model_path)

    original_info = summarize_model(onnx_model, os.path.basename(onnx_model_path))

    onnx_model = replace_mod(onnx_model)
    
    onnx_model = onnxslim.slim(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model, check_type=True, strict_mode=True)

    onnx.save_model(onnx_model, onnx_model_output_path)

    slimmed_info = summarize_model(onnx_model, os.path.basename(onnx_model_output_path))
    print_model_info_as_table([original_info, slimmed_info])




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

    export(yolo_type)
    modify(yolo_type)