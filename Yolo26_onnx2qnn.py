import sys
import argparse
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))


from utilities.onnx_to_qnn import OnnxToQNN



# 模型文件路径
YOLO26_MODEL_PATH = str(current_dir / 'models_convert/onnx/yolo26s_[1,3,320,640].onnx')
YOLO26_POSE_MODEL_PATH = str(current_dir / 'models_convert/onnx/yolo26s-pose_[1,3,320,640].onnx')

# 导出路径
YOLO26_QNN_MODEL = str(current_dir / 'models_convert/rknn/yolo26s_i8[1,320,640,3].bin')
YOLO26_POSE_QNN_MODEL = str(current_dir / 'models_convert/rknn/yolo26s-pose_i8[1,320,640,3].bin')

DATASET_PATH = str(parent_dir / 'datasets/datasets.txt')



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

    if yolo_type == 'yolo':
        model_path = YOLO26_MODEL_PATH
        qnn_model = YOLO26_QNN_MODEL

    elif yolo_type == 'yolo-pose':
        model_path = YOLO26_POSE_MODEL_PATH
        qnn_model = YOLO26_POSE_QNN_MODEL
        
    else:
        raise ValueError("yolo_type must be 'yolo', 'yolo-pose'")
    
    onnx_to_qnn = OnnxToQNN(model_path, qnn_model, DATASET_PATH)

    onnx_to_qnn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[255, 255, 255]])
    onnx_to_qnn.clean()