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
    from Edge_Inferencer.ai_inferencer import AIInferencer, timeit




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
        
        self.CLASSES = CLASSES
        self.color_list = COLOR_LIST

        
        self.yolo26_infer = AIInferencer(self.model_path, cores=cores, mult_task=self.mult_task)

        # 预计算锚点网格（用于 bbox 解码）
        self.anchor_xy, self.anchor_stride, self.stride_dim_list = self.build_anchor_grids(self.model_size)
        self.total_anchors = self.anchor_xy.shape[0]

        self.class_scores_arange = np.arange(self.total_anchors, dtype=np.uint16)

        # 预期输出形状（channels, anchors）— 用于识别输出顺序
        self.output_order = None  # 首次推理后由 identify_output_order() 缓存

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

        expected_shapes = bbox_shapes + cls_shapes

        size_to_idx: dict[int, int] = {shape[0] * shape[1]: i for i, shape in enumerate(expected_shapes)}
        order = list(range(len(expected_shapes)))

        for actual_idx, arr in enumerate(infer_output):
            expected_idx = size_to_idx.get(arr.size)
            if expected_idx is not None:
                order[expected_idx] = actual_idx

        return tuple(order)

    @staticmethod
    def contact_outputs(ordered_outputs:list[np.ndarray]) -> list[np.ndarray]:
        bbox_outputs = [arr.reshape(4, -1) for arr in ordered_outputs[:3]]
        cls_outputs = [arr.reshape(80, -1) for arr in ordered_outputs[3:]]

        bbox_output = np.concatenate(bbox_outputs, axis=-1)
        cls_output = np.concatenate(cls_outputs, axis=-1)

        contacted_outputs = [bbox_output, cls_output]
        return contacted_outputs

    @staticmethod
    def preprocess(color_image:np.ndarray) -> np.ndarray:
        color_float = color_image.astype(np.float32) / 255.0
        return color_float

    def bbox_anchor(self, boxes_raw:np.ndarray) -> np.ndarray:
        # x1 = (anchor_x - left) * stride
        # y1 = (anchor_y - top) * stride
        # x2 = (anchor_x + right) * stride
        # y2 = (anchor_y + bottom) * stride
        boxes_raw[:, 0:1] = self.anchor_xy[:, 0:1] - boxes_raw[:, 0:1]
        boxes_raw[:, 1:2] = self.anchor_xy[:, 1:2] - boxes_raw[:, 1:2]
        boxes_raw[:, 2:3] += self.anchor_xy[:, 0:1]
        boxes_raw[:, 3:4] += self.anchor_xy[:, 1:2]

        boxes_raw *= self.anchor_stride # 乘以步长，转为像素坐标
        return boxes_raw

    def post_process(self, infer_output:list[np.ndarray], scale:tuple[float, float]=(1.0, 1.0), offset:tuple[int, int]=(0, 0)) -> np.ndarray|None:
        if self.output_order is None: # 首次推理时识别输出顺序
            self.output_order = self.identify_output_order(infer_output, self.stride_dim_list)
            if self.output_order is None:
                return None

        outputs = [infer_output[i] for i in self.output_order] 
        outputs = self.contact_outputs(outputs) # bbox, class
    
        # 根据预期形状 (channels, anchors) 重塑为 (anchors, channels)
        boxes_raw = np.reshape(outputs[0], (4, self.total_anchors)).swapaxes(0, 1)   # [anchors, 4]
        cls_raw = np.reshape(outputs[1], (80, self.total_anchors)).swapaxes(0, 1)   # [anchors, 80]

        # 模型已内置 Sigmoid，class scores 直接使用
        cls_scores = cls_raw

        # bbox 解码
        boxes = self.bbox_anchor(boxes_raw)

        # 每锚框的最佳类别和分数
        class_ids = np.argmax(cls_scores, axis=1)
        max_scores = cls_scores[self.class_scores_arange, class_ids]

        # 置信度阈值过滤
        mask = max_scores > self.conf_thresh
        boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_class_ids = class_ids[mask]

        if boxes.size == 0:
            return None

        # 按置信度降序
        # order = np.argsort(filtered_scores)[::-1][:256]
        # boxes = boxes[order]
        # filtered_scores = filtered_scores[order]
        # filtered_class_ids = filtered_class_ids[order]

        # 缩放回原始图像坐标
        boxes[..., 0::2] = (boxes[..., 0::2] - offset[0]) / scale[0] # (x1, x2)
        boxes[..., 1::2] = (boxes[..., 1::2] - offset[1]) / scale[1] # (y1, y2)

        boxes[..., 0::2] = np.clip(boxes[..., 0::2], 0, self.model_size[0])
        boxes[..., 1::2] = np.clip(boxes[..., 1::2], 0, self.model_size[1])

        results = np.column_stack([boxes, filtered_class_ids, filtered_scores])
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
    yolo26.detect = timeit(yolo26.detect)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_resized, scale, offsets = resize_image(rgb_frame, (640, 320))

        detect_result = yolo26.detect(rgb_frame_resized, scale=(scale, scale), offset=offsets)

        if detect_result is not None:
            frame = draw_yolo(frame, detect_result)
        cv2.imshow('Yolo26', frame)

        key = cv2.waitKey(1)
        if key == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    yolo26.release()