import os
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

def resize_image(image:np.ndarray, input_size:tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    # 调整图像大小并保持宽高比
    h, w = image.shape[:2]
    r = min(input_size[1] / h, input_size[0] / w)
    new_h, new_w = int(h * r), int(w * r)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # 计算填充大小
    pad_h = input_size[1] - new_h
    pad_w = input_size[0] - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 使用copyMakeBorder填充图像
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 灰色填充
    return padded, r, left, top

def preprocess_image(image:np.ndarray, input_size:tuple[int, int], mean_rgb:list[int]=[0, 0, 0], std_rgb:list[int]=[1, 1, 1]) -> tuple[np.ndarray, float, int, int]:
    """预处理图像：调整大小、归一化、增加批次维度"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    padded, r, left, top = resize_image(image, input_size)
    
    # 归一化并调整维度顺序
    mean_rgb = np.array(mean_rgb, dtype=np.float32).reshape((1, 1, 3))
    std_rgb = np.array(std_rgb, dtype=np.float32).reshape((1, 1, 3))
    normalized = (padded.astype(np.float32) - mean_rgb) / std_rgb

    normalized = normalized.transpose(2, 0, 1)
    normalized = np.expand_dims(normalized, axis=0)
    
    return normalized, r, left, top

def process_predictions(output: np.ndarray) -> list[list[int, int, int, int, int, float]]:
    """处理模型输出，返回置信度最高的3个检测结果"""

    conf_threshold:float=0.1

    # 输出形状为 (1, 300, 38)
    output = output.squeeze()  # 移除批次维度，形状变为 (300, 38)
    output = output[:, :6]

    # 分离边界框坐标、类别ID和分数
    boxes = output[:, :4]  # (300, 4)
    scores = output[:, 4]  # (300,)
    class_ids = output[:, 5]  # (300,)

    # 应用置信度阈值过滤
    mask = scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    
    if len(filtered_boxes) == 0:
        return []
    
    idxs = np.argsort(filtered_scores, axis=0)[::-1]  # 按置信度降序排序
    filtered_boxes = filtered_boxes[idxs]
    filtered_scores = filtered_scores[idxs]
    filtered_class_ids = filtered_class_ids[idxs]

    results = []


    for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
        x1, y1, x2, y2 = box
        
        results.append([int(x1), int(y1), int(x2), int(y2), int(class_id), float(score)])

    return results


class GenYoloDetedDataset:
    def __init__(self, dataset_path:str, model_path:str):
        self.dataset_path = Path(dataset_path).resolve()
        self.model_path = model_path

        self.human_list = [0]
        self.animal_list = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 77]
        self.vehicle_list = [2, 3, 4, 5, 6, 7, 8, 30, 31, 33, 36, 37]

    def gerenate(self, model_input_size_wh:tuple[int, int]=(640, 640)) -> Path|str:
        # 初始化ONNX运行时
        session = ort.InferenceSession(self.model_path)
        input_name = session.get_inputs()[0].name

        # 读取图片路径列表
        image_path_set = set()
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    image_path_set.add(str(Path(line_s)))

        # 构建完整图片路径
        dataset_dir = str(self.dataset_path.parent)
        full_img_path_list = [os.path.join(dataset_dir, img_path) for img_path in image_path_set]


        for image_path in full_img_path_list:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {image_path}")
                continue
            
            # 预处理
            input_tensor, scale, x_offset, y_offset = preprocess_image(image, input_size=model_input_size_wh, std_rgb=[255, 255, 255])
            
            # 推理
            output = session.run(None, {input_name: input_tensor})[0]
            
            # 处理预测结果
            results = process_predictions(output)

            self.display(image, results, scale, x_offset, y_offset)
            #time.sleep(0.5)


        cv2.destroyAllWindows()
        del session

    def display(self, image: np.ndarray, boxes: list[list[int, int, int, int, int, float]], scale: float, x_offset: int, y_offset: int):
        display_image = image.copy()  # 创建显示用的图像副本
        
        for i, (x1, y1, x2, y2, class_id, conf) in enumerate(boxes):
            # 将坐标转换回原始图像尺寸
            x1 = int((x1 - x_offset) / scale)
            y1 = int((y1 - y_offset) / scale)
            x2 = int((x2 - x_offset) / scale)
            y2 = int((y2 - y_offset) / scale)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image.shape[1]))
            y1 = max(0, min(y1, image.shape[0]))
            x2 = max(0, min(x2, image.shape[1]))
            y2 = max(0, min(y2, image.shape[0]))

            id = int(class_id)

            if id in self.human_list:
                color = (255, 0, 0)
            # elif id == 1 or id in self.animal_list:
            #     color = (0, 255, 0)
            # elif id == 2 or id in self.vehicle_list:
            #     color = (0, 0, 255)
            elif id == 1:
                color = (0, 255, 0)
            elif id == 2:
                color = (0, 0, 255)
            else:
                color = (255, 255, 255)
            
            # 在显示图像上绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            # 在边界框上方显示置信度
            cv2.putText(display_image, f'{id} {conf:.2f}', (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        display_image = resize_image(display_image, (848, 848))[0]

        # 显示带有边界框的图像
        cv2.imshow('Detection Results', display_image)
        if cv2.waitKey(1) == ord('q'):
            exit(1)
        

def main():
    # 配置参数
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录的绝对路径
    parent_dir = os.path.dirname(current_dir) # 获取当前文件所在目录的父目录的绝对路径
    project_dir = os.path.dirname(parent_dir) # 获取当前文件所在目录的父目录的父目录的绝对路径

    dataset_path = os.path.join(project_dir, 'datasets/datasets_full.txt')  # 输入图片索引文本
    model_path = os.path.join(parent_dir, 'models_convert/onnx/yolo26s_[1,3,320,640].onnx')


    # 创建对象并生成数据集
    dataset_generator = GenYoloDetedDataset(dataset_path, model_path)

    dataset_generator.gerenate(model_input_size_wh=(640, 320))


if __name__ == "__main__":
    main()
