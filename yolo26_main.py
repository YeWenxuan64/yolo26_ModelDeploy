import sys
import time
from pathlib import Path
import cv2
import numpy as np




current_dir = Path(__file__).parent.resolve()

class temporary_sys_path:
    def __init__(self, new_path: str):
        self.new_path = str(new_path)
        
    def __enter__(self):
        sys.path.insert(0, self.new_path)
        return self
        
    def __exit__(self, etype, value, traceback):
        # 安全移除：避免原本就在 sys.path 里导致误删
        if self.new_path in sys.path:
            sys.path.remove(self.new_path)

with temporary_sys_path(current_dir.parent):
    from Edge_Inferencer.ai_inferencer import AIInferencer




CLASSES = ("person", "bicycle", "car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
           "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
           "bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote ","keyboard ","cell phone","microwave",
           "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier", "toothbrush")

COLOR_LIST = [(4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183), (17, 31, 104), (255, 111, 221), (255, 68, 79), (204, 237, 0), (0, 243, 68), (189, 0, 255),
              (0, 180, 255), (221, 0, 186), (0, 255, 255), (38, 192, 0), (1, 255, 179), (125, 36, 255), (123, 0, 104), (255, 27, 108), (252, 109, 47), (162, 255, 11),
              (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 153, 255), (153, 204, 255), (255, 102, 255), (255, 51, 255), (102, 178, 255), (51, 153, 255),
              (255, 153, 153), (255, 102, 102), (255, 51, 51), (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0), (0, 0, 255), (255, 0, 0), (216, 216, 216)]


class Yolo26:
    def __init__(self, model_path:str, need_preprocess:bool=False, conf_threshold:float=0.25, cores:tuple[int]=(0,), mult_task:bool=False):
        """
        args:
            model_path: model_path
            cores: cores
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold # 0.25
        self.need_preprocess = need_preprocess
        self.mult_task = mult_task
        
        self.CLASSES = CLASSES
        self.color_list = COLOR_LIST
        
        self.output_shape = (-1, 6)
        self.yolo26_infer = AIInferencer(self.model_path, cores=cores, mult_task=self.mult_task)

    def preprocess(self, color_image:np.ndarray) -> np.ndarray:
        color_float = color_image.astype(np.float32) / 255.0
        return color_float

    def post_process(self, infer_output:list[np.ndarray], scale:tuple[float, float]=(1.0, 1.0), offset:tuple[int, int]=(0, 0)) -> np.ndarray|None:
        # 输出形状为 (1, N, 6)
        output = infer_output[0].reshape(self.output_shape)  # 移除批次维度，形状变为 (N, 6)

        # 分离边界框坐标、类别ID和分数
        boxes = output[:, :4]  # (N, 4)
        scores = output[:, 4]  # (N,)
        class_ids = output[:, 5]  # (N,)

        # 应用置信度阈值过滤
        mask = scores > self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]
        
        if filtered_boxes.size == 0:
            return None
        
        idxs = np.argsort(filtered_scores, axis=0)[::-1]  # 按置信度降序排序
        filtered_boxes = filtered_boxes[idxs]             # (x1, y1, x2, y2)
        filtered_scores = filtered_scores[idxs]
        filtered_class_ids = filtered_class_ids[idxs]

        filtered_boxes[..., 0::2] -= offset[0]
        filtered_boxes[..., 1::2] -= offset[1]
        filtered_boxes[..., 0::2] /= scale[0]
        filtered_boxes[..., 1::2] /= scale[1]

        results = np.hstack((filtered_boxes, np.vstack(filtered_class_ids), np.vstack(filtered_scores)), dtype=np.float32)
        return results

    def detect(self, color_image:np.ndarray, block:bool=True, scale:tuple[float, float]=(1.0, 1.0), offset:tuple[int, int]=(0, 0)) -> np.ndarray|None:
        """
        Args:
            color_image: np.arraylike(h, w, 3)
            block: bool, if True, block until get result

        Returns:
            np.arraylike(n, 6) [[x1, y1, x2, y2, class, score]...]
        """

        if self.need_preprocess:
            color_image = self.preprocess(color_image)

        input_data = np.expand_dims(color_image, axis=0) # 添加batch维度

        outputs = self.yolo26_infer.inferfacer.put([input_data])
        if self.mult_task:
            outputs = self.yolo26_infer.inferfacer.get(block=block)

        detect_result = None
        if outputs is not None:
            detect_result = self.post_process(outputs, scale, offset) # 使用后处理从将推理结果获取检测结果

        return detect_result

    def release(self):
        ret = self.yolo26_infer.release()
        if ret:
            print('Yolo26 Released')


def draw_yolo(image:np.ndarray, detect_result:np.ndarray):
    for i in range(detect_result.shape[0]):
        x1, y1, x2, y2, class_id, score = detect_result[i, :6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        class_id = min(int(class_id), len(CLASSES) - 1)
        score = float(score)

        class_name = CLASSES[class_id]
        rgb_color = COLOR_LIST[class_id % len(COLOR_LIST)]
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, 2)
        cv2.putText(image, f'{class_name} {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

    return image


def resize_image(image:np.ndarray, target_shape:tuple[int, int], allow_crop:bool=True) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Args:
        image: 输入图像 np.ndarray
        target_shape: 目标尺寸 tuple(width, height)
        allow_crop: 是否裁剪图像 bool

    Returns:
        letterbox_image: 调整后的图像 np.ndarray
        ratio: 缩放比例 float
        offset: 调整后图像的偏移坐标 tuple(int, int) # (x, y)
    """

    original_w, original_h = image.shape[1::-1]  # (w, h)
    target_w, target_h = target_shape

    w_scale = target_w / original_w
    h_scale = target_h / original_h

    if allow_crop is True:
        scale = max(w_scale, h_scale) # 裁剪时，基于最大边缩放，填满目标尺寸
    else:
        scale = min(w_scale, h_scale) # 不裁剪时，基于最小边缩放，确保图像完整显示

    new_shape = (original_w * scale, original_h * scale) # (w, h)
    offset = ((target_w - new_shape[0]) // 2, (target_h - new_shape[1]) // 2)
    
    warp_matrix = np.array([[scale, 0, offset[0]], [0, scale, offset[1]]], dtype=np.float32)
    letterbox_image = cv2.warpAffine(image, warp_matrix, target_shape, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return letterbox_image, scale, offset




if __name__ == '__main__':
    model_path = str(current_dir / 'models_convert/onnx/yolo26s_[1,3,320,640].onnx')
    video_path = str(current_dir.parent / 'datasets/loco640.mp4')

    yolo26 = Yolo26(model_path=model_path, need_preprocess=True)
    cap = cv2.VideoCapture(video_path)

    time_array = np.zeros(30, dtype=np.float32)
    time_list_idx = 0
    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_resized, scale, offsets = resize_image(rgb_frame, (640, 320))

        start_time = time.time()
        detect_result = yolo26.detect(rgb_frame_resized, scale=(scale, scale), offset=offsets)
        end_time = time.time()

        time_array[time_list_idx] = end_time - start_time
        time_list_idx = (time_list_idx + 1) % time_array.size

        if detect_result is not None:
            frame = draw_yolo(frame, detect_result)
        cv2.imshow('Yolo26', frame)


        if end_time - last_print_time >= 1.0:
            last_print_time = end_time
            
            average_time = np.mean(time_array)
            fps = 1.0 / average_time
            print(f"FPS: {fps:.2f}")

        key = cv2.waitKey(1)
        if key == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    yolo26.release()