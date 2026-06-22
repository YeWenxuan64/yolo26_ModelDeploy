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
    config.end2end = True
    # Export settings
    config.format = "onnx"      # (str) target format, e.g. torchscript|onnx|openvino|engine|coreml|saved_model|pb|tflite|edgetpu|tfjs|paddle|mnn|ncnn|imx|rknn|executorch|axelera
    config.simplify = False     # (bool) ONNX/engine only; run graph simplifier for cleaner ONNX before runtime conversion
    config.opset = 13           # (int, optional) ONNX/engine only; opset version for export; leave unset to use a tested default

    return config


def insert_sigmoid_after_class_convs(model: onnx.ModelProto, conv_node_names: list[str]) -> onnx.ModelProto:
    """
    在指定的 Conv 节点之后插入 Sigmoid，并更新所有下游引用。
    用于将 Concat 之后的 Sigmoid 移到 Concat 之前的各分支 Conv 后面，
    使转换器（RKNN/QNN）可以将 Conv+Sigmoid 融合。
    """
    graph = model.graph
    old_nodes = list(graph.node)
    new_nodes = []

    # 建立 conv_output -> sigmoid_output 映射
    sigmoid_map = {}
    for node in old_nodes:
        if node.name in conv_node_names:
            sigmoid_map[node.output[0]] = node.output[0] + "_sigmoid"

    # 重建节点列表，在每个目标 Conv 后插入 Sigmoid
    new_sigmoid_names = set()
    for node in old_nodes:
        new_nodes.append(node)
        if node.name in conv_node_names:
            sigmoid_node = onnx.helper.make_node(
                "Sigmoid",
                inputs=[node.output[0]],
                outputs=[sigmoid_map[node.output[0]]],
                name=node.name.replace("/Conv", "/Sigmoid"),
            )
            new_nodes.append(sigmoid_node)
            new_sigmoid_names.add(sigmoid_node.name)

    # 更新所有下游节点的输入引用（跳过新插入的 Sigmoid 本身）
    for node in new_nodes:
        if node.name in new_sigmoid_names:
            continue
        for i, inp in enumerate(node.input):
            if inp in sigmoid_map:
                node.input[i] = sigmoid_map[inp]

    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"Inserted Sigmoid after {len(conv_node_names)} class Conv nodes")
    return model

def move_sigmoid_into_concat_branches(model: onnx.ModelProto, concat_node_name: str) -> onnx.ModelProto:
    """
    自动探测：若 Concat 节点的输出下游紧接 Sigmoid，则将该 Sigmoid 移到 Concat
    各输入分支的最后一个 Conv 之后（便于转换器融合 Conv+Sigmoid）。

    工作流程：
      1. 找到 Concat 节点
      2. 检查 Concat 的输出消费者是否为 Sigmoid
      3. 沿 Concat 的每个输入逆向追溯到对应的 Conv 节点
      4. 在每路 Conv 后插入 Sigmoid，更新下游引用
    """
    graph = model.graph
    nodes = list(graph.node)

    # 1. 定位 Concat 节点
    concat_node = None
    for node in nodes:
        if node.name == concat_node_name:
            concat_node = node
            break
    if concat_node is None:
        raise ValueError(f"Concat node '{concat_node_name}' not found")
    if concat_node.op_type != "Concat":
        raise ValueError(f"Node '{concat_node_name}' is {concat_node.op_type}, not Concat")

    concat_out = concat_node.output[0]

    # 2. 检查 Concat 输出消费者是否为 Sigmoid
    sigmoid_consumer = None
    for node in nodes:
        if concat_out in node.input:
            if node.op_type == "Sigmoid":
                sigmoid_consumer = node
                print(f"  Detected Sigmoid after Concat: {node.name}")
                break
            else:
                raise ValueError(
                    f"Concat output '{concat_out}' consumed by {node.op_type} "
                    f"('{node.name}'), expected Sigmoid"
                )

    if sigmoid_consumer is None:
        print(f"  No Sigmoid found after Concat '{concat_node_name}', nothing to move")
        return model

    # 3. 建立 tensor → producer 映射
    tensor_to_producer = {}
    for node in nodes:
        for out in node.output:
            tensor_to_producer[out] = node

    # 4. 沿 Concat 每个输入逆向追溯，找到对应的 Conv
    conv_node_names = []
    for inp_name in concat_node.input:
        producer = tensor_to_producer.get(inp_name)
        if producer is None:
            print(f"  Warning: input '{inp_name}' has no producer, skipping")
            continue

        # 如果是 Reshape，继续往前找 Conv
        if producer.op_type == "Reshape":
            conv_inp = producer.input[0]  # Reshape 的第一个输入来自 Conv
            producer = tensor_to_producer.get(conv_inp)

        if producer is None or producer.op_type != "Conv":
            print(f"  Warning: could not find Conv upstream of '{inp_name}', skipping")
            continue

        conv_node_names.append(producer.name)
        print(f"  Branch: ... → {producer.name} → {inp_name}")

    if not conv_node_names:
        raise ValueError("No Conv nodes found upstream of Concat inputs")

    # 5. 插入 Sigmoid
    model = insert_sigmoid_after_class_convs(model, conv_node_names)

    # 6. 移除原来 Concat 后面的 Sigmoid 节点
    #    先将其输出引用全部重定向到 Concat 的输出（确保下游不断连）
    graph = model.graph
    old_sigmoid_out = sigmoid_consumer.output[0]
    nodes = list(graph.node)
    for node in nodes:
        for i, inp in enumerate(node.input):
            if inp == old_sigmoid_out:
                node.input[i] = concat_out
    #    再删除旧 Sigmoid
    nodes = [n for n in nodes if n.name != sigmoid_consumer.name]
    del graph.node[:]
    graph.node.extend(nodes)
    print(f"  Removed original Sigmoid '{sigmoid_consumer.name}', rewired {old_sigmoid_out} → {concat_out}")

    return model

def trim_model_to_outputs(model:onnx.ModelProto, output_node_names:list[str]) -> onnx.ModelProto:
    """
    Keep nodes up to the farthest target node (inclusive), and set multiple
    target nodes' first outputs as the model outputs.

    Args:
        model: ONNX model
        output_node_names: list of node names whose first output will become model outputs.
                          The cut point is the latest (farthest in graph order) of these nodes.
    """
    model = onnx.shape_inference.infer_shapes(model)

    # Find all target nodes and the latest one
    target_indices = {}
    for i, node in enumerate(model.graph.node):
        if node.name in output_node_names:
            target_indices[node.name] = i

    missing = set(output_node_names) - set(target_indices.keys())
    if missing:
        raise ValueError(f"Nodes not found: {missing}")

    cut_idx = max(target_indices.values())

    # Gather all output tensor info
    outputs = []
    for name in output_node_names:
        idx = target_indices[name]
        node = model.graph.node[idx]
        out_name = node.output[0]
        # Get output shape
        for v in list(model.graph.value_info) + list(model.graph.output):
            if v.name == out_name:
                shape = [d.dim_value for d in v.type.tensor_type.shape.dim]
                break
        else:
            raise ValueError(f"Could not determine shape of '{out_name}'.")
        outputs.append(onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, shape))
        print(f"  Output[{name}]: {out_name} {shape}")

    new_graph = onnx.helper.make_graph(
        list(model.graph.node[: cut_idx + 1]),
        model.graph.name + "_trimmed",
        model.graph.input,
        outputs,
        model.graph.initializer,
    )

    new_model = onnx.helper.make_model(new_graph, opset_imports=model.opset_import)
    print(f"Trimmed model: cut at index {cut_idx}, {len(outputs)} outputs")

    return new_model



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

    # 自动将 Concat_1 后的 Sigmoid 移到各分支 Conv 后（便于 Conv+Sigmoid 融合）
    onnx_model = move_sigmoid_into_concat_branches(onnx_model, "/model.23/Concat_1")

    if yolo_type == "yolo":
        # 双输出截断：原始 bbox + 已 sigmoid 的 class
        trim_nodes = ["/model.23/Concat", "/model.23/Concat_1"]
    else:
        # pose 三输出截断：bbox + 已 sigmoid 的 class + keypoints
        trim_nodes = ["/model.23/Concat", "/model.23/Concat_1", "/model.23/Concat_2"]

    onnx_model = trim_model_to_outputs(onnx_model, trim_nodes)

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