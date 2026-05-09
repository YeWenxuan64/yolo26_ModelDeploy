import os
import sys
import argparse

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = os.path.abspath(current_path)

sys.path.insert(0, os.path.dirname(current_path))

from utilities.onnx_to_rknn import OnnxToRKNN



# 模型文件路径
YOLO26_MODEL_PATH = os.path.join(current_path, 'models_convert/onnx/yolo26s_[1,3,320,640].onnx')
YOLO26_POSE_MODEL_PATH = os.path.join(current_path, 'models_convert/onnx/yolo26s-pose_[1,3,320,640].onnx')

# 导出路径
YOLO26_RKNN_MODEL = os.path.join(current_path, 'models_convert/rknn/yolo26s_i8[1,320,640,3].rknn')
YOLO26_POSE_RKNN_MODEL = os.path.join(current_path, 'models_convert/rknn/yolo26s-pose_i8[1,320,640,3].rknn')

DATASET_PATH = os.path.join(os.path.dirname(current_path), 'datasets/datasets.txt')

TARGET_PLATFORM = 'rk3588'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO model converter')

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
        rknn_model = YOLO26_RKNN_MODEL

    elif yolo_type == 'yolo-pose':
        model_path = YOLO26_POSE_MODEL_PATH
        rknn_model = YOLO26_POSE_RKNN_MODEL
        
    else:
        raise ValueError("yolo_type must be 'yolo', 'yolo-pose'")



    onnx_to_rknn = OnnxToRKNN(model_path, rknn_model, DATASET_PATH, TARGET_PLATFORM)
    #onnx_to_rknn.set_do_accuracy_analysis('/home/yewenxuan/convert_models/convert_models/datasets/bus.jpg')

    onnx_to_rknn.extra_optimize(flash_attantion=True)

    onnx_to_rknn.convert(mean_rgb=[[0, 0, 0]], std_rgb=[[255, 255, 255]])
    onnx_to_rknn.clean()
