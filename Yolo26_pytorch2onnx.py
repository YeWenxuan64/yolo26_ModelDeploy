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


def insert_sigmoid_after_convs(model:onnx.ModelProto, conv_node_names:list[str]) -> onnx.ModelProto:
    """
    在指定的 Conv 节点之后插入 Sigmoid, 并更新所有下游引用。
    用于将 Concat 之后的 Sigmoid 移到 Concat 之前的各分支 Conv 后面，
    使转换器可以将 Conv+Sigmoid 融合。
    """
    graph = model.graph
    nodes = graph.node

    # 建立 conv_output -> sigmoid_output 映射
    conv_to_sigmoid_map:dict[str, str] = {}
    sigmoid_node_list:list[onnx.NodeProto] = []
    sigmoid_node_insert_index_list:list[int] = []

    for i, node in enumerate(nodes):
        if node.name in conv_node_names:
            target_conv_output_name = node.output[0]
            sigmoid_output_name = f"{target_conv_output_name}_sigmoid"

            sigmoid_node = onnx.helper.make_node(
                "Sigmoid",
                inputs=[node.output[0]],
                outputs=[sigmoid_output_name],
                name=node.name.replace("/Conv", "/Sigmoid"),
            )

            conv_to_sigmoid_map[target_conv_output_name] = sigmoid_output_name
            sigmoid_node_list.append(sigmoid_node)
            sigmoid_node_insert_index_list.append(i+1)

    # 更新所有下游引用
    for node in nodes:
        for i, node_input in enumerate(node.input):
            if node_input in conv_to_sigmoid_map.keys():
                node.input[i] = conv_to_sigmoid_map[node_input]

    # 插入 Sigmoid 节点
    for i in range(len(sigmoid_node_list)):
        sigmoid_node = sigmoid_node_list.pop(0)
        insert_index = sigmoid_node_insert_index_list.pop(0)

        nodes.insert(insert_index, sigmoid_node)
        print(f"Inserted Sigmoid after {sigmoid_node.input[0]}")

    return model

def move_sigmoid_into_concat_branches(model:onnx.ModelProto, concat_node_name:str) -> onnx.ModelProto:
    """
    自动探测：若 Concat 节点的输出下游紧接 Sigmoid, 则将该 Sigmoid 移到 Concat
    各输入分支的最后一个 Conv 之后(便于转换器融合 Conv+Sigmoid)。

    工作流程：
      1. 找到 Concat 节点
      2. 检查 Concat 的输出消费者是否为 Sigmoid
      3. 沿 Concat 的每个输入逆向追溯到对应的 Conv 节点
      4. 在每路 Conv 后插入 Sigmoid, 更新下游引用
    """
    graph = model.graph
    nodes = graph.node

    # 1. 定位 Concat 节点
    concat_node = None
    for node in nodes:
        if node.name == concat_node_name:
            concat_node = node
            break

    if concat_node is None or concat_node.op_type != "Concat":
        raise ValueError(f"Concat node '{concat_node_name}' not found")


    concat_out = concat_node.output[0]

    # 2. 检查 Concat 输出消费者是否为 Sigmoid
    sigmoid_consumer = None
    for node in nodes:
        if concat_out in node.input:
            if node.op_type == "Sigmoid":
                sigmoid_consumer = node
                print(f"  Detected Sigmoid after Concat: {node.name}")
                break

    if sigmoid_consumer is None:
        print(f"  No Sigmoid found after Concat '{concat_node_name}', nothing to move")
        return model

    # 3. 建立 tensor → producer 映射
    tensor_to_producer:dict[str, onnx.NodeProto] = {}
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
        print(f"  {producer.name} → {inp_name}")

    if not conv_node_names:
        raise ValueError("No Conv nodes found upstream of Concat inputs")

    # 5. 插入 Sigmoid
    model = insert_sigmoid_after_convs(model, conv_node_names)

    # 6. 移除原来 Concat 后面的 Sigmoid 节点
    #    先将其输出引用全部重定向到 Concat 的输出（确保下游不断连）
    old_sigmoid_out = sigmoid_consumer.output[0]

    for node in nodes:
        for i, inp in enumerate(node.input):
            if inp == old_sigmoid_out:
                node.input[i] = concat_out
                
    #    再删除旧 Sigmoid
    nodes.remove(sigmoid_consumer)

    model = onnx.shape_inference.infer_shapes(model, check_type=True)
    print(f"  Removed original Sigmoid '{sigmoid_consumer.name}', rewired {old_sigmoid_out} → {concat_out}")

    return model

def find_nodes_before_nodes(model:onnx.ModelProto, node_names:list[str]) -> list[str]:
    """
    找到所有在指定节点之前的节点（深度为1，即目标节点的直接输入生产者）。
    仅返回 graph.node 中的节点，排除 Initializer。

    返回的节点名按双重顺序排列：
      1. 先按 node_names 中目标节点的顺序；
      2. 再按每个目标节点的 inputs 列表顺序。
    """
    graph = model.graph
    nodes = graph.node

    # 收集所有 Initializer 的 tensor 名，用于排除
    initializer_names = {init.name for init in graph.initializer}

    # 建立 tensor 名 → 生产者节点 的映射
    tensor_to_producer: dict[str, onnx.NodeProto] = {}
    for node in nodes:
        for out in node.output:
            tensor_to_producer[out] = node

    # 建立 node 名 → node 的映射，方便按名查找目标节点
    name_to_node: dict[str, onnx.NodeProto] = {}
    for node in nodes:
        name_to_node[node.name] = node

    producer_name_list: list[str] = []
    for target_name in node_names:
        target_node = name_to_node.get(target_name)
        if target_node is None:
            continue
        # 按目标节点 inputs 的顺序遍历
        for inp_name in target_node.input:
            # 跳过 Initializer（常量）和 graph input
            if inp_name in initializer_names:
                continue

            producer = tensor_to_producer.get(inp_name)

            if producer is None or producer.op_type == "Constant":
                continue

            producer_name_list.append(producer.name)

    return producer_name_list

def trim_model_to_outputs(model: onnx.ModelProto, output_node_names: list[str]) -> onnx.ModelProto:
    """
    使用 DAG 反向遍历算法裁剪 ONNX 模型。
    仅保留计算目标输出所需的精确节点，自动剔除死代码，完全不依赖节点列表索引。
    """
    # 1. 运行形状推导，确保 value_info 包含完整的类型和维度信息
    model = onnx.shape_inference.infer_shapes(model)
    
    # 去重并保持顺序
    seen = set()
    unique_output_names = [n for n in output_node_names if not (n in seen or seen.add(n))]
    
    # 2. 构建图结构映射表 (Producer Map & Node ID Map)
    # producer_map: 张量名 -> 产生该张量的节点对象 (相当于图的反向边)
    # node_id_map: 使用 id(node) 作为键，避免 Protobuf 对象极慢的 Hash 和 Eq 计算
    producer_map = {}
    node_id_map = {} 
    
    for node in model.graph.node:
        node_id_map[id(node)] = node
        for output_name in node.output:
            producer_map[output_name] = node

    # 3. 识别目标节点
    target_nodes = [n for n in model.graph.node if n.name in unique_output_names]
    
    missing = set(unique_output_names) - {n.name for n in target_nodes}
    if missing:
        raise ValueError(f"Nodes not found: {missing}")

    # 4. 核心算法：深度优先搜索 (DFS) 逆向寻找所有祖先节点
    required_node_ids = set()
    stack = [id(n) for n in target_nodes]
    
    while stack:
        curr_id = stack.pop()
        if curr_id in required_node_ids:
            continue
            
        required_node_ids.add(curr_id)
        curr_node = node_id_map[curr_id]
        
        # 逆向遍历当前节点的所有输入张量
        for input_tensor in curr_node.input:
            # 忽略可选的空输入 ("")
            if input_tensor and input_tensor in producer_map:
                parent_node = producer_map[input_tensor]
                parent_id = id(parent_node)
                if parent_id not in required_node_ids:
                    stack.append(parent_id)

    # 5. 过滤节点：保留在依赖树中的节点，自动维持原有的拓扑排序
    trimmed_nodes = [node for node in model.graph.node if id(node) in required_node_ids]

    # 6. 死代码消除 (Dead Code Elimination)
    # 找出所有被保留节点实际消耗的张量，剔除未使用的 Inputs 和 Initializers (权重)
    required_tensors = set()
    for node in trimmed_nodes:
        required_tensors.update(node.input)
        
    trimmed_initializers = [init for init in model.graph.initializer if init.name in required_tensors]
    trimmed_inputs = [inp for inp in model.graph.input if inp.name in required_tensors]

    # 7. 构建新图的 Outputs (支持动态维度与真实数据类型)
    outputs = []
    # 汇总所有可能包含形状/类型信息的来源
    value_info_map = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input)}
    
    for node in target_nodes:
        if not node.output:
            raise ValueError(f"Target node '{node.name}' has no outputs.")
            
        out_name = node.output[0]
        elem_type = onnx.TensorProto.FLOAT  # 默认回退
        shape = []                     # 默认动态/未知
        
        if out_name in value_info_map:
            v = value_info_map[out_name]
            if v.type.HasField('tensor_type'):
                # 提取真实的数据类型 (如 FLOAT16, INT64 等)
                elem_type = v.type.tensor_type.elem_type
                
                # 兼容动态维度 (dim_param) 和静态维度 (dim_value)
                if v.type.tensor_type.HasField('shape'):
                    for d in v.type.tensor_type.shape.dim:
                        if d.HasField('dim_value'):
                            shape.append(d.dim_value)
                        elif d.HasField('dim_param'):
                            shape.append(d.dim_param)
                        else:
                            shape.append(None)
        
        outputs.append(onnx.helper.make_tensor_value_info(out_name, elem_type, shape))
        print(f"  Output[{node.name}]: {out_name} | type: {elem_type} | shape: {shape}")

    # 8. 组装并返回新模型
    new_graph = onnx.helper.make_graph(
        trimmed_nodes,
        model.graph.name + "_trimmed",
        trimmed_inputs,
        outputs,
        trimmed_initializers,
    )

    new_model = onnx.helper.make_model(new_graph, opset_imports=model.opset_import)
    print(f"Successfully trimmed model.")
    
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
        target_concat_nodes = ["/model.23/Concat", "/model.23/Concat_1"]
    else:
        # pose 三输出截断：bbox + 已 sigmoid 的 class + keypoints
        target_concat_nodes = ["/model.23/Concat", "/model.23/Concat_1", "/model.23/Concat_2"]

    reshape_nodes = find_nodes_before_nodes(onnx_model, target_concat_nodes)
    nodes_before_reshape = find_nodes_before_nodes(onnx_model, reshape_nodes)

    onnx_model = trim_model_to_outputs(onnx_model, nodes_before_reshape)


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