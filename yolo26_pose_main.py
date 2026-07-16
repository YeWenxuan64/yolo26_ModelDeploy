import sys
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
    from edge_inferencer.ai_inferencer import AIInferencer, timeit




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
    def __init__(self, model_path:str, model_size:tuple=(640, 320), need_preprocess:bool=False, conf_thresh:float=0.25, cores:tuple[int]=(0,), mult_task:bool=False):
        """
        args:
            model_path: model_path
            cores: cores
        """
        self.model_path = model_path
        self.model_size = model_size  # (width, height)
        self.conf_thresh = conf_thresh # 0.25
        self.need_preprocess = need_preprocess
        self.mult_task = mult_task
        
        self.classes = CLASSES
        self.color_list = COLOR_LIST
        self.keypoints_color = KEYPOINTS_COLOR
        self.limbs_color = LIMBS_COLOR
        self.skeletons = SKELETONS
        
        self.yolo26pose_infer = AIInferencer(self.model_path, cores=cores, mult_task=self.mult_task)

        # 预计算锚点网格
        self.anchor_xy, self.anchor_stride, self.stride_dim_list = self.build_anchor_grids(self.model_size)
        self.total_anchors = self.anchor_xy.shape[0]
        self.class_scores_arange = np.arange(self.total_anchors, dtype=np.uint16)

        # 预期输出形状 (channels, anchors): bbox=4, class=1, kpts=51 (17×3)
        self.output_order = None

    @staticmethod
    def build_anchor_grids(model_size_wh:tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """预生成三个检测头的锚点网格和步长张量"""
        strides = [8, 16, 32]
        w, h = model_size_wh

        grid_xy_list = []
        stride_list = []
        stride_dim_list = []

        for stride in strides:
            fw, fh = w // stride, h // stride  # 特征图宽高
            # 生成网格坐标 (0.5 偏移到 cell 中心)
            gy, gx = np.meshgrid(np.arange(fh, dtype=np.float32) + 0.5,
                                 np.arange(fw, dtype=np.float32) + 0.5, indexing='ij')
            
            grid_xy = np.stack([gx.ravel(), gy.ravel()], axis=1)  # [fw*fh, 2]

            grid_xy_list.append(grid_xy)
            stride_list.append(np.full((fw * fh, 1), stride, dtype=np.float32))
            stride_dim_list.append((fh, fw))

        anchor_xy = np.concatenate(grid_xy_list, axis=0)   # [4200, 2]
        nchor_stride = np.concatenate(stride_list, axis=0) # [4200, 1]

        return anchor_xy, nchor_stride, stride_dim_list
    
    @staticmethod
    def identify_output_order(infer_output:list[np.ndarray], stride_dim_list:list[tuple[int, int]]) -> tuple[int, ...]:
        """根据元素总数匹配实际输出到预期形状，返回重排索引"""
        bbox_shapes = [(4, feature_h, feature_w) for feature_h, feature_w in stride_dim_list]
        cls_shapes = [(1, feature_h, feature_w) for feature_h, feature_w in stride_dim_list]
        kpts_shapes = [(51, feature_h, feature_w) for feature_h, feature_w in stride_dim_list]

        expected_shapes = bbox_shapes + cls_shapes + kpts_shapes

        size_to_idx: dict[int, int] = {shape[0] * shape[1]: i for i, shape in enumerate(expected_shapes)}
        order = list(range(len(expected_shapes)))

        for actual_idx, arr in enumerate(infer_output):
            expected_idx = size_to_idx.get(arr.size)
            if expected_idx is not None:
                order[expected_idx] = actual_idx

        return tuple(order)
    
    @staticmethod
    def preprocess(color_image:np.ndarray) -> np.ndarray:
        color_float = color_image.astype(np.float32) / 255.0
        return color_float

    def bbox_anchor(self, bbox_concat:np.ndarray) -> np.ndarray:
        bbox_concat = bbox_concat.swapaxes(0, 1) # (4, N) -> (N, 4)

        np.subtract(self.anchor_xy, bbox_concat[..., 0:2], out=bbox_concat[..., 0:2])
        np.add(self.anchor_xy, bbox_concat[..., 2:4], out=bbox_concat[..., 2:4])

        np.multiply(bbox_concat, self.anchor_stride, out=bbox_concat)
        
        boxes = bbox_concat
        return boxes

    def kpts_anchor(self, kpts_concat:np.ndarray) -> np.ndarray:
        kpts_raw = kpts_concat.swapaxes(0, 1) # (51, N) -> (N, 51)

        kpts_raw[..., 0::3] += self.anchor_xy[:, 0:1]
        kpts_raw[..., 1::3] += self.anchor_xy[:, 1:2]

        kpts_raw[..., 0::3] *= self.anchor_stride # 乘以步长，转为像素坐标
        kpts_raw[..., 1::3] *= self.anchor_stride

        kpts = kpts_raw.reshape(-1, 17, 3)
        kpts[..., 2] = 1.0 / (1.0 + np.exp(-kpts[..., 2]))  # sigmoid

        return kpts.reshape(-1, 51)

    def post_process(self, infer_output:list[np.ndarray], scale:tuple[float, float]=(1.0, 1.0), offset:tuple[int, int]=(0, 0)) -> np.ndarray|None:
        if self.output_order is None: # 首次推理时识别输出顺序
            self.output_order = self.identify_output_order(infer_output, self.stride_dim_list)
            if self.output_order is None:
                return None

        outputs = [infer_output[i] for i in self.output_order] 

        # 分离边界框坐标、分数、关键点
        bbox_concat = np.concatenate([arr.reshape(4, -1) for arr in outputs[:3]], axis=-1) # (4, N)
        cls_concat = np.concatenate([arr.reshape(1, -1) for arr in outputs[3:6]], axis=-1) # (80, N)
        kpts_concat = np.concatenate([arr.reshape(51, -1) for arr in outputs[6:9]], axis=-1) # (51, N)

        # bbox 解码
        boxes = self.bbox_anchor(bbox_concat)
        keypoints = self.kpts_anchor(kpts_concat)

        # 模型已内置 Sigmoid，class scores 直接使用
        cls_scores = cls_concat

        # 每锚框的最佳类别和分数
        class_ids = np.argmax(cls_scores, axis=0)
        max_scores = cls_scores[class_ids, self.class_scores_arange]

        # 置信度阈值过滤
        mask = max_scores > self.conf_thresh
        boxes = boxes[mask]
        scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]
        keypoints = keypoints[mask]

        if boxes.size == 0:
            return None

        # 按置信度降序
        # order = np.argsort(scores)[::-1][:256]
        # boxes = boxes[order]
        # scores = scores[order]
        # filtered_class_ids = filtered_class_ids[order]
        # keypoints = keypoints[order]

        # 缩放回原始图像坐标
        boxes[..., 0::2] = (boxes[..., 0::2] - offset[0]) / scale[0]
        boxes[..., 1::2] = (boxes[..., 1::2] - offset[1]) / scale[1]
        boxes[..., 0::2] = np.clip(boxes[..., 0::2], 0, self.model_size[0])
        boxes[..., 1::2] = np.clip(boxes[..., 1::2], 0, self.model_size[1])

        keypoints[..., 0::3] = (keypoints[..., 0::3] - offset[0]) / scale[0]
        keypoints[..., 1::3] = (keypoints[..., 1::3] - offset[1]) / scale[1]
        keypoints[..., 0::3] = np.clip(keypoints[..., 0::3], 0, self.model_size[0])
        keypoints[..., 1::3] = np.clip(keypoints[..., 1::3], 0, self.model_size[1])

        results = np.column_stack((boxes, filtered_class_ids, scores, keypoints))
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

        outputs = self.yolo26pose_infer.inferfacer.put([input_data])
        if self.mult_task:
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
            cv2.putText(image, f'{score:.2f}', (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)

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
    yolo26pose.detect = timeit(yolo26pose.detect)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_resized, scale, offsets = resize_image(rgb_frame, (640, 320))

        detect_result = yolo26pose.detect(rgb_frame_resized, scale=(scale, scale), offset=offsets)

        if detect_result is not None:
            frame = draw_yolo_pose(frame, detect_result)
        cv2.imshow('Yolo26 Pose', frame)

        key = cv2.waitKey(1)
        if key == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    yolo26pose.release()