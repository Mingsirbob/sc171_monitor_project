# ~/sc171_monitor_project/src/ai_processing/yolo_v8_snpe_postprocessor.py
import numpy as np
from typing import List, Dict, Any, Tuple 

# 尝试从config导入配置，如果失败则使用后备值
try:
    from config_sc171 import COCO_CLASSES # YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD 会作为参数传入
except ImportError:
    print("警告 [yolo_v8_snpe_postprocessor]: 无法从config_sc171导入COCO_CLASSES。")
    print("将使用本模块内定义的默认COCO_CLASSES列表进行测试。")
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def xyxy2xywh(box):
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box

def clip_coords(boxes, img_shape):
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])

def scale_coords(model_input_shape, coords, original_img_shape):
    gain_w = original_img_shape[1] / model_input_shape[1]
    gain_h = original_img_shape[0] / model_input_shape[0]
    coords[:, [0, 2]] *= gain_w
    coords[:, [1, 3]] *= gain_h
    clip_coords(coords, original_img_shape)
    return coords

def NMS(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]


def detect_postprocess(prediction, original_shape, model_shape, conf_thres, iou_thres):
    num_classes = len(COCO_CLASSES)
    print(f"原始模型输出形状: {prediction.shape}")

    num_properties = num_classes + 4
    if prediction.ndim == 1:
        num_boxes = prediction.shape[0] // num_properties
        output = prediction.reshape(num_properties, num_boxes).T
    elif prediction.ndim == 2:
        output = prediction.T
    elif prediction.ndim == 3:
        output = np.transpose(prediction, (0, 2, 1))[0]
    else:
        raise ValueError(f"Unsupported prediction shape: {prediction.shape}")
    
    scores = np.max(output[:, 4:], axis=1)
    conf_mask = scores > conf_thres
    
    boxes_filtered_xywh = output[conf_mask, :4]
    scores_filtered = scores[conf_mask]
    class_ids_filtered = np.argmax(output[conf_mask, 4:], axis=1)

    if len(boxes_filtered_xywh) == 0:
        return [[] for _ in range(num_classes)]
    
    # print(f"过滤后边界框boxes_filtered_xywh: {boxes_filtered_xywh}")
    # print(f"过滤后置信度分数scores_filtered: {scores_filtered}")
    # print(f"过滤后类别class_ids_filtered: {class_ids_filtered}")

    boxes_filtered_xyxy = xywh2xyxy(boxes_filtered_xywh)

    final_results = []
    for i in range(num_classes):
        cls_mask = (class_ids_filtered == i)
        # print(f"类别 {i} ({COCO_CLASSES[i]}): 检测到 {cls_mask} 个边界框")
        if not np.any(cls_mask):
            final_results.append([])
            continue

        cls_boxes = boxes_filtered_xyxy[cls_mask]
        cls_scores = scores_filtered[cls_mask]
        
        dets = np.column_stack([cls_boxes, cls_scores])
        keep_dets = NMS(dets, iou_thres)
        
        if len(keep_dets) == 0:
            final_results.append([])
            continue
            
        final_boxes_xyxy = scale_coords(model_shape, keep_dets[:, :4], original_shape)
        final_boxes_xywh = xyxy2xywh(final_boxes_xyxy)
        
        final_results.append(np.column_stack([final_boxes_xywh, keep_dets[:, 4]]))

    return final_results