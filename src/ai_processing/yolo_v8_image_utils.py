# ~/sc171_monitor_project/src/ai_processing/yolo_v8_image_utils.py
import cv2
import numpy as np


def preprocess_img(img, target_shape: tuple):
    img_resized = cv2.resize(img, target_shape)
    img_processed = img_resized.astype(np.float32) / 255.0
    return img_processed[None, :]


def draw_detect_res(img, all_boxes):
    final_box_count = sum(len(b) for b in all_boxes)
    print(f"\n--- 绘制 {final_box_count} 个最终检测框 ---")
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
    colors = [(255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 157, 151), 
              (255, 112, 31), (72, 249, 10), (61, 219, 134), (0, 194, 255)]

    img_uint8 = img.astype(np.uint8)
    
    for class_id, boxes in enumerate(all_boxes):
        if len(boxes) == 0:
            continue
        class_name = COCO_CLASSES[class_id]
        for box in boxes:
            x, y, w, h, conf = [int(t) for t in box[:4]] + [box[4]]
            print(f"检测到: {class_name:<12} | 置信度: {conf:.2f} | 边框 (x,y,w,h): ({x}, {y}, {w}, {h})")
            color = colors[class_id % len(colors)]
            cv2.rectangle(img_uint8, (x, y), (x + w, y + h), color, thickness=2)
            label = f'{class_name} {conf:.2f}'
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_uint8, (x, y), (x + text_w, y - text_h - baseline), color, -1)
            cv2.putText(img_uint8, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img_uint8