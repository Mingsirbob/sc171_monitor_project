# ~/sc171_monitor_project/src/ai_processing/yolo_v8_snpe_postprocessor.py
import numpy as np
# 导入配置，主要为了COCO_CLASSES和可能的阈值，但函数设计上应尽量接收这些作为参数
# 假设在调用此模块的脚本中，PYTHONPATH已正确设置，可以找到顶层config
try:
    from config_sc171 import COCO_CLASSES, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD
except ImportError:
    print("警告 [yolo_v8_snpe_postprocessor]: 无法从config_sc171导入配置。")
    print("将使用本模块内定义的默认COCO_CLASSES和阈值进行测试。")
    # 后备定义，主要用于本模块的独立单元测试
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
    YOLO_CONF_THRESHOLD = 0.25 # 默认置信度阈值
    YOLO_IOU_THRESHOLD = 0.45  # 默认IOU阈值


def _xywh_to_xyxy(x_center: np.ndarray, y_center: np.ndarray, width: np.ndarray, height: np.ndarray) -> tuple:
    """将中心点xywh转换为左上角xyxy。所有输入和输出都是NumPy数组。"""
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return x1, y1, x2, y2

def _non_maximum_suppression_per_class(boxes_xyxy_conf_for_class: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    对单个类别的检测框执行NMS。
    Args:
        boxes_xyxy_conf_for_class: NumPy数组，形状 (N, 5)，列为 [x1, y1, x2, y2, confidence_score]。
        iou_threshold: NMS的IoU阈值。
    Returns:
        NMS处理后保留的检测框NumPy数组，形状 (M, 5)。
    """
    if boxes_xyxy_conf_for_class.shape[0] == 0:
        return np.array([]).reshape(0, 5)

    x1 = boxes_xyxy_conf_for_class[:, 0]
    y1 = boxes_xyxy_conf_for_class[:, 1]
    x2 = boxes_xyxy_conf_for_class[:, 2]
    y2 = boxes_xyxy_conf_for_class[:, 3]
    scores = boxes_xyxy_conf_for_class[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep_indices = []
    while order.size > 0:
        i = order[0]
        keep_indices.append(i)

        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection_area = w * h
        
        iou = intersection_area / (areas[i] + areas[order[1:]] - intersection_area + 1e-6)
        
        inds_to_keep_in_order = np.where(iou <= iou_threshold)[0]
        order = order[inds_to_keep_in_order + 1]

    return boxes_xyxy_conf_for_class[keep_indices]


def apply_yolov8_snpe_postprocessing(
    raw_output_data_dict: dict,
    model_output_names: list,
    expected_output_shape: tuple,
    original_image_h: int,
    original_image_w: int,
    model_input_h: int,
    model_input_w: int,
    conf_threshold: float,
    iou_threshold: float,
    class_names: list = None
) -> list:
    """
    对YOLOv8 SNPE模型的原始输出进行后处理。
    Args:
        # ... (参数说明与之前一致) ...
    Returns:
        格式化后的检测结果列表，每个元素是一个字典:
        {'class_id': int, 'class_name': str, 'confidence': float, 
         'bbox_xywh': [x_top_left, y_top_left, width, height]} (坐标相对于原始图像)
    """
    if class_names is None:
        class_names = COCO_CLASSES

    if not raw_output_data_dict or not model_output_names:
        print("错误 [postprocess]: 原始输出字典或模型输出名列表为空。")
        return []

    main_output_node_name = model_output_names[0]
    if main_output_node_name not in raw_output_data_dict:
        print(f"错误 [postprocess]: 在SNPE输出中未找到指定的输出节点 '{main_output_node_name}'。")
        print(f"可用的输出节点: {list(raw_output_data_dict.keys())}")
        return []
    
    flat_output_list = raw_output_data_dict[main_output_node_name]
    if flat_output_list is None:
        print(f"错误 [postprocess]: 输出节点 '{main_output_node_name}' 的数据为None。")
        return []

    try:
        expected_total_elements = np.prod(expected_output_shape)
        if len(flat_output_list) != expected_total_elements:
            print(f"错误 [postprocess]: 原始输出列表长度 ({len(flat_output_list)}) "
                  f"与期望形状 {expected_output_shape} (总元素 {expected_total_elements}) 不匹配。")
            return []
        
        prediction = np.array(flat_output_list, dtype=np.float32).reshape(expected_output_shape)
    except Exception as e:
        print(f"错误 [postprocess]: Reshape原始输出时失败: {e}")
        return []

    if prediction.shape[0] != 1:
        print(f"警告 [postprocess]: Prediction的批次大小不是1 (实际为 {prediction.shape[0]})。只处理第一个批次。")
    
    proposals = prediction.transpose(0, 2, 1)[0] 

    boxes_xywh_model_scale = proposals[:, :4]
    class_scores_model_scale = proposals[:, 4:]

    object_confidences = np.max(class_scores_model_scale, axis=1)
    class_ids = np.argmax(class_scores_model_scale, axis=1)

    valid_indices = object_confidences >= conf_threshold
    
    if not np.any(valid_indices):
        return []

    boxes_xywh_model_scale_filtered = boxes_xywh_model_scale[valid_indices]
    confidences_filtered = object_confidences[valid_indices]
    class_ids_filtered = class_ids[valid_indices]

    x_center, y_center, w, h = boxes_xywh_model_scale_filtered[:,0], boxes_xywh_model_scale_filtered[:,1], \
                               boxes_xywh_model_scale_filtered[:,2], boxes_xywh_model_scale_filtered[:,3]
    x1_model, y1_model, x2_model, y2_model = _xywh_to_xyxy(x_center, y_center, w, h)
    
    boxes_xyxy_model_scale_filtered = np.vstack((x1_model, y1_model, x2_model, y2_model)).T

    final_detections_list = []
    num_defined_classes = len(class_names)

    for class_id_to_process in range(num_defined_classes):
        class_mask = (class_ids_filtered == class_id_to_process)
        if not np.any(class_mask):
            continue

        current_class_boxes_xyxy = boxes_xyxy_model_scale_filtered[class_mask]
        current_class_confidences = confidences_filtered[class_mask]
        
        if current_class_boxes_xyxy.shape[0] == 0:
            continue

        boxes_for_nms = np.hstack((current_class_boxes_xyxy, current_class_confidences[:, np.newaxis]))
        
        kept_boxes_after_nms = _non_maximum_suppression_per_class(boxes_for_nms, iou_threshold)

        if kept_boxes_after_nms.shape[0] > 0:
            scale_x = original_image_w / model_input_w
            scale_y = original_image_h / model_input_h

            for box_info in kept_boxes_after_nms:
                x1_m, y1_m, x2_m, y2_m, conf_m = box_info

                x1_orig = x1_m * scale_x
                y1_orig = y1_m * scale_y
                x2_orig = x2_m * scale_x
                y2_orig = y2_m * scale_y

                x1_orig = np.clip(x1_orig, 0, original_image_w)
                y1_orig = np.clip(y1_orig, 0, original_image_h)
                x2_orig = np.clip(x2_orig, 0, original_image_w)
                y2_orig = np.clip(y2_orig, 0, original_image_h)

                w_orig = x2_orig - x1_orig
                h_orig = y2_orig - y1_orig
                
                if w_orig > 0 and h_orig > 0:
                    final_detections_list.append({
                        'class_id': class_id_to_process,
                        'class_name': class_names[class_id_to_process],
                        'confidence': float(conf_m),
                        'bbox_xywh': [int(round(x1_orig)), int(round(y1_orig)), 
                                      int(round(w_orig)), int(round(h_orig))]
                    })
    
    return final_detections_list


# --- (可选) 模块级测试代码 ---
if __name__ == '__main__':
    print("--- yolo_v8_snpe_postprocessor.py 模块测试 ---")

    num_proposals_test = 100 
    num_classes_test = len(COCO_CLASSES) # 80
    
    # **** 修改点：初始化为低值，然后精确设置强信号 ****
    # 形状 (1, 84, 100) for (Batch, Coords+Classes, Proposals)
    dummy_raw_output_snpe_format = np.full((1, 4 + num_classes_test, num_proposals_test), 0.01, dtype=np.float32) 
    # 随机化坐标部分 (前4个通道)，值在模型输入尺度内 (e.g., 0-639)
    dummy_raw_output_snpe_format[0, :4, :] = np.random.uniform(low=10.0, high=600.0, size=(4, num_proposals_test))

    # 手动设置几个强信号
    # Proposal 0: person (class_id 0), high conf
    dummy_raw_output_snpe_format[0, 0, 0] = 100.0  # x_center (model scale)
    dummy_raw_output_snpe_format[0, 1, 0] = 150.0  # y_center
    dummy_raw_output_snpe_format[0, 2, 0] = 50.0   # width
    dummy_raw_output_snpe_format[0, 3, 0] = 100.0  # height
    dummy_raw_output_snpe_format[0, 4 + 0, 0] = 0.95 # Score for person (class_id 0)

    # Proposal 1: person (class_id 0), high conf, overlapping with proposal 0
    dummy_raw_output_snpe_format[0, 0, 1] = 110.0
    dummy_raw_output_snpe_format[0, 1, 1] = 160.0
    dummy_raw_output_snpe_format[0, 2, 1] = 50.0
    dummy_raw_output_snpe_format[0, 3, 1] = 100.0
    dummy_raw_output_snpe_format[0, 4 + 0, 1] = 0.90 # Score for person

    # Proposal 2: car (class_id 2), high conf, different location
    dummy_raw_output_snpe_format[0, 0, 2] = 300.0
    dummy_raw_output_snpe_format[0, 1, 2] = 350.0
    dummy_raw_output_snpe_format[0, 2, 2] = 80.0
    dummy_raw_output_snpe_format[0, 3, 2] = 60.0
    dummy_raw_output_snpe_format[0, 4 + 2, 2] = 0.88 # Score for car (class_id 2)
    
    # 模拟SNPE Execute返回的字典
    mock_snpe_outputs = {
        "output0": dummy_raw_output_snpe_format.flatten().tolist() 
    }
    
    model_output_names_test = ["output0"] 
    expected_shape_test = (1, 4 + num_classes_test, num_proposals_test) 

    original_h_test, original_w_test = 720, 1280
    model_h_test, model_w_test = 640, 640 

    print(f"\n[测试 apply_yolov8_snpe_postprocessing]")
    # 使用模块内定义的默认阈值进行测试
    # 如果想用config_sc171.py中的值，确保它能被正确导入
    # conf_thresh_to_use = YOLO_CONF_THRESHOLD 
    # iou_thresh_to_use = YOLO_IOU_THRESHOLD
    # 如果导入失败，YOLO_CONF_THRESHOLD 和 YOLO_IOU_THRESHOLD 会是模块顶部的后备值
    
    detections = apply_yolov8_snpe_postprocessing(
        raw_output_data_dict=mock_snpe_outputs,
        model_output_names=model_output_names_test,
        expected_output_shape=expected_shape_test,
        original_image_h=original_h_test,
        original_image_w=original_w_test,
        model_input_h=model_h_test,
        model_input_w=model_w_test,
        conf_threshold=YOLO_CONF_THRESHOLD, # 使用导入或默认的阈值
        iou_threshold=YOLO_IOU_THRESHOLD,
        class_names=COCO_CLASSES 
    )

    print(f"后处理期望检测到: 1 'person' 和 1 'car' (经过NMS和置信度过滤)")
    if detections:
        print(f"后处理实际检测到 {len(detections)} 个对象:")
        persons_found = 0
        cars_found = 0
        for det in detections:
            print(f"  类别: {det['class_name']} (ID: {det['class_id']}), "
                  f"置信度: {det['confidence']:.2f}, "
                  f"BBox_xywh (原始尺寸): {det['bbox_xywh']}")
            if det['class_name'] == 'person':
                persons_found +=1
            if det['class_name'] == 'car':
                cars_found +=1
        print(f"总结: 找到 Person: {persons_found}, Car: {cars_found}")
    else:
        print("后处理未检测到任何对象 (或全部被过滤)。")

    print("\n--- yolo_v8_snpe_postprocessor.py 模块测试完成 ---")