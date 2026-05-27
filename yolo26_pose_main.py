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




CLASSES = ("person",)
COLOR_LIST = [(4, 42, 255)]

POSE_PALETTE = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 153, 255),
                (153, 204, 255), (255, 102, 255), (255, 51, 255), (102, 178, 255), (51, 153, 255),
                (255, 153, 153), (255, 102, 102), (255, 51, 51), (153, 255, 153), (102, 255, 102),
                (51, 255, 51), (0, 255, 0), 
                (0, 0, 255), (255, 0, 0), (255, 255, 255)]

KEYPOINTS_COLOR = [POSE_PALETTE[i] for i in [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

LIMBS_COLOR = [POSE_PALETTE[i] for i in [16, 16, 16, 16, 9, 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0]]


# COCO 17个关键点的骨骼连线定义 (索引从0开始)
# 0鼻子, 1左眼, 2右眼, 3左耳, 4右耳
# 5左肩, 6右肩, 7左肘, 8右肘, 9左腕, 10右腕, 11左髋关节, 12右髋关节
# 13左膝, 14右膝, 15左脚踝, 16右脚踝
SKELETONS = [(0, 1), (0, 2), (1, 3), (2, 4),          # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 上半身
            (5, 11), (6, 12), (11, 12),              # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)]  # 下半身





class Yolo26Pose:
    def __init__(self, model_path:str, need_preprocess:bool=False, conf_threshold:float=0.25, cores:tuple[int]=(0,), mult_task:bool=False):
        """
        args:
            model_path: model_path
            cores: cores
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold # 0.25
        self.need_preprocess = need_preprocess
        
        self.classes = CLASSES
        self.color_list = COLOR_LIST
        self.keypoints_color = KEYPOINTS_COLOR
        self.limbs_color = LIMBS_COLOR
        self.skeletons = SKELETONS
        
        self.output_shape = (-1, 57)
        self.yolo26pose_infer = AIInferencer(self.model_path, cores=cores, mult_task=mult_task)

    def preprocess(self, color_image:np.ndarray) -> np.ndarray:
        color_float = color_image.astype(np.float32) / 255.0
        return color_float

    def post_process(self, infer_output:list[np.ndarray], scale:tuple[float, float]=(1.0, 1.0), offset:tuple[int, int]=(0, 0)) -> np.ndarray|None:
        # 输出形状为 (1, N, 57)
        output = infer_output[0].reshape(self.output_shape)  # 移除批次维度，形状变为 (N, 57)

        # 分离边界框坐标、类别ID和分数
        boxes = output[:, :4]  # (N, 4)
        scores = output[:, 4]  # (N,)
        class_ids = output[:, 5]  # (N,)
        keypoints = output[:, 6:57]  # 关键点数据 (N, 51) (x1, y1, conf1, x2, y2, conf2, ..., x17, y17, conf17)

        # 应用置信度阈值过滤
        mask = scores > self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_class_ids = class_ids[mask]
        filtered_kpts = keypoints[mask]
        
        if len(filtered_boxes) == 0:
            return None
        
        idxs = np.argsort(filtered_scores, axis=0)[::-1]  # 按置信度降序排序
        filtered_boxes = filtered_boxes[idxs]
        filtered_scores = filtered_scores[idxs]
        filtered_class_ids = filtered_class_ids[idxs]
        filtered_kpts = filtered_kpts[idxs]

        # 移动和缩放边界框坐标
        filtered_boxes[..., 0::2] -= offset[0]
        filtered_boxes[..., 1::2] -= offset[1]
        filtered_boxes[..., 0::2] /= scale[0]
        filtered_boxes[..., 1::2] /= scale[1]

        filtered_kpts[..., 0::3] -= offset[0]
        filtered_kpts[..., 1::3] -= offset[1]
        filtered_kpts[..., 0::3] /= scale[0]
        filtered_kpts[..., 1::3] /= scale[1]

        results = np.hstack((filtered_boxes, np.vstack(filtered_class_ids), np.vstack(filtered_scores), filtered_kpts), dtype=np.float32)

        return results

    def detect(self, color_image:np.ndarray, block:bool=True, scale:tuple[float, float]=(1.0, 1.0), offset:tuple[int, int]=(0, 0)) -> np.ndarray|None:
        """
        Args:
            color_image: np.arraylike(h, w, 3)
            block: bool, if True, block until get result

        Returns:
            np.arraylike(n, 57) [[x1, y1, x2, y2, class, score, kpt1_x, kpt1_y, kpt1_conf ...]...]
        """

        if self.need_preprocess:
            color_image = self.preprocess(color_image)

        input_data = np.expand_dims(color_image, axis=0) # 添加batch维度

        self.yolo26pose_infer.inferfacer.put([input_data])
        outputs = self.yolo26pose_infer.inferfacer.get(block=block)

        detect_result = None
        if outputs is not None:
            detect_result = self.post_process(outputs, scale, offset) # 使用后处理从将推理结果获取检测结果

        return detect_result

    def release(self):
        ret = self.yolo26pose_infer.release()
        if ret:
            print('Yolo26 Pose Released')


def draw_yolo_pose(image:np.ndarray, detect_result:np.ndarray):
    for i in range(detect_result.shape[0]):
        # 解析 BBox 和 基础信息
        x1, y1, x2, y2, class_id, score = detect_result[i, :6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        class_id = min(int(class_id), len(CLASSES) - 1)
        score = float(score)
        class_name = CLASSES[class_id]
        
        bbox_rgb_color = COLOR_LIST[class_id % len(COLOR_LIST)]
        bbox_bgr_color = (bbox_rgb_color[2], bbox_rgb_color[1], bbox_rgb_color[0])

        # 1. 画边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_bgr_color, 2)
        cv2.putText(image, f'{class_name} {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_bgr_color, 2)

        # 2. 解析关键点 (17, 3) -> [x, y, conf]
        kpts = detect_result[i, 6:].reshape(-1, 3)
        
        # 3. 画骨骼连线
        for sk_id, sk in enumerate(SKELETONS):
            pos1_idx, pos2_idx = sk[0], sk[1]
            conf1, conf2 = kpts[pos1_idx, 2], kpts[pos2_idx, 2]
            
            # if conf1 > kpt_thr and conf2 > kpt_thr:
            px1, py1 = int(kpts[pos1_idx, 0]), int(kpts[pos1_idx, 1])
            px2, py2 = int(kpts[pos2_idx, 0]), int(kpts[pos2_idx, 1])

            skeleton_color = LIMBS_COLOR[sk_id % len(KEYPOINTS_COLOR)]
            bgr_color = (skeleton_color[2], skeleton_color[1], skeleton_color[0])
            
            cv2.line(image, (px1, py1), (px2, py2), bgr_color, 2)
        
        # 4. 画关键点圆圈
        for kpt_id in range(len(kpts)):
            conf = kpts[kpt_id, 2]
            # if conf > kpt_thr:
            kx, ky = int(kpts[kpt_id, 0]), int(kpts[kpt_id, 1])

            keypoint_color = KEYPOINTS_COLOR[kpt_id % len(KEYPOINTS_COLOR)]
            bgr_color = (keypoint_color[2], keypoint_color[1], keypoint_color[0])

            cv2.circle(image, (kx, ky), 3, bgr_color, -1)

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
    model_path = str(current_dir / 'models_convert/onnx/yolo26s-pose_[1,3,320,640].onnx')
    video_path = str(current_dir.parent / 'datasets/loco640.mp4')

    yolo26pose = Yolo26Pose(model_path=model_path, need_preprocess=True)
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
        detect_result = yolo26pose.detect(rgb_frame_resized, scale=(scale, scale), offset=offsets)
        end_time = time.time()

        time_array[time_list_idx] = end_time - start_time
        time_list_idx = (time_list_idx + 1) % time_array.size

        if detect_result is not None:
            frame = draw_yolo_pose(frame, detect_result)
        cv2.imshow('Yolo26 Pose', frame)


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
    yolo26pose.release()