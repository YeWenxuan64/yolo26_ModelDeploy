import sys
import argparse
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

from utilities.onnx_to_qnn import OnnxToQNN
from utilities.utils import fmt_model_name_with_shape



# 模型文件路径
YOLO26_MODEL_PATH = str(current_dir / 'models_convert/onnx/yolo26s_[1,3,320,640].onnx')
YOLO26_POSE_MODEL_PATH = str(current_dir / 'models_convert/onnx/yolo26s-pose_[1,3,320,640].onnx')

QNN_OUTPUT_DIR = current_dir / 'models_convert/qnn'

DATASET_PATH = str(parent_dir / 'datasets/datasets.txt')
DATASET_PATH = str(parent_dir / 'datasets/datasets_short.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO model egde converter')

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

    if yolo_type == 'yolo':
        model_path = YOLO26_MODEL_PATH
        qnn_model_name = fmt_model_name_with_shape(model_path, model_name="yolo26s_i8{shapes}.bin", use_nhwc=True)
        qnn_model = str(QNN_OUTPUT_DIR / qnn_model_name)

    elif yolo_type == 'yolo-pose':
        model_path = YOLO26_POSE_MODEL_PATH
        qnn_model_name = fmt_model_name_with_shape(model_path, model_name="yolo26s-pose_i8{shapes}.bin", use_nhwc=True)
        qnn_model = str(QNN_OUTPUT_DIR / qnn_model_name)
        
    else:
        raise ValueError("yolo_type must be 'yolo', 'yolo-pose'")
    
    onnx_to_qnn = OnnxToQNN(model_path, qnn_model, DATASET_PATH)
    onnx_to_qnn.set_quantization_method(param_quant_method='sqnr', act_quant_method='entropy')

    onnx_to_qnn.set_do_accuracy_analysis([str(parent_dir / 'datasets/bus.jpg')])

    onnx_to_qnn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[255, 255, 255]])
    #onnx_to_qnn.clean()