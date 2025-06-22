# ~/sc171_monitor_project/src/ai_processing/yolo_detector_sc171.py
import numpy as np
import time
import cv2 
import os
from PIL import Image


try:
    from .api_infer_wrapper import SnpeContext, Runtime, PerfProfile, LogLevel
    from .yolo_v8_image_utils import preprocess_img, draw_detect_res
    from .yolo_v8_snpe_postprocessor import detect_postprocess
except ImportError as e: 
    print(f"错误 [YoloDetectorSC171]: 导入本地模块失败: {e}")
    try:
        from src.ai_processing.api_infer_wrapper import SnpeContext, Runtime, PerfProfile, LogLevel
        from src.ai_processing.yolo_v8_image_utils import preprocess_image, draw_detect_res
        from src.ai_processing.yolo_v8_snpe_postprocessor import apply_yolov8_snpe_postprocessing
    except ImportError:
        raise ImportError("无法加载YoloDetectorSC171的依赖模块。") from e


from config_sc171 import YOLO_DLC_PATH, YOLO_MODEL_INPUT_NAME, YOLO_MODEL_OUTPUT_NAMES, COCO_CLASSES, MODEL_INPUT_SHAPE, CONF_THRESHOLD, IOU_THRESHOLD


class YoloDetectorSC171:
    def __init__(self):
        # 初始化YOLO检测器
        self.dlc_path = YOLO_DLC_PATH
        self.model_input_name = YOLO_MODEL_INPUT_NAME
        self.model_output_name = YOLO_MODEL_OUTPUT_NAMES
        self.class_names = COCO_CLASSES
        self.num_classes = len(COCO_CLASSES)
        self.num_properties_per_proposal = self.num_classes + 4
        self.conf_threshold = CONF_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        
        # 初始化 SNPE 推理引擎
        self.snpe_ort = SnpeContext(self.dlc_path, [], Runtime.GPU, PerfProfile.BALANCED, LogLevel.INFO)
        assert self.snpe_ort.Initialize() == 0, "SNPE 引擎初始化失败！"
        print(f"YoloDetectorSC171: SNPE 引擎已成功初始化，使用模型: {self.dlc_path}")

    # --- 返回类型提示已修正 ---
    def detect(self, image):
        self.image = image
        image_shape = self.image.shape
        input_tensor = preprocess_img(self.image, MODEL_INPUT_SHAPE)

        input_feed = {self.model_input_name: input_tensor}
        # 执行推理
        outputs = self.snpe_ort.Execute(YOLO_MODEL_OUTPUT_NAMES, input_feed)
        raw_output = np.array(outputs['output0'])
        # 对推理结果进行后处理
        self.final_detections = detect_postprocess(
            self.class_names,
            raw_output, 
            image_shape, 
            MODEL_INPUT_SHAPE,
            CONF_THRESHOLD, 
            IOU_THRESHOLD
        )
        return self.final_detections
        

    def draw_results(self,save_path=None):
        SAVE_PATH = save_path
        image_to_draw = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        result_image_array = draw_detect_res(image_to_draw, self.final_detections,self.class_names)
        result_image_pil = Image.fromarray(result_image_array)
        result_image_pil.save(SAVE_PATH)

    def save_results(self, find_object):
        result = []
        for class_id, boxes in enumerate(self.final_detections):
            if self.class_names[class_id] in find_object[0]:
                for box in boxes:
                    x, y, w, h = [int(t) for t in box[:4]]
                    conf = box[4]
                    line = f"检测到{self.class_names[class_id]}|置信度{conf:.4f} (x,y,w,h):({x},{y},{w},{h})"
                    print(line)
                    obj = {}
                    obj["class_name"]=self.class_names[class_id]
                    obj["confidence"]=conf
                    result.append(obj)
        return result


    def close(self):
        print("YoloDetectorSC171: 正在释放SNPE上下文...") 
        self.snpe_ort.Release()
        print("YoloDetectorSC171: SNPE上下文已成功释放。")


if __name__ == "__main__":
    YOLO_DLC_PATH = "models/yolov8n.dlc"
    YOLO_MODEL_INPUT_NAME = "images"
    YOLO_MODEL_OUTPUT_NAMES = ["output0"]
    MODEL_INPUT_SHAPE = (640, 640) 
    MODEL_OUTPUT_SHAPE = (1,84,8400)
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
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
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    TRIGGER_CLASSES_FOR_GEMINI = ["person"]
    IMG_PATH = "data/test_images/bus.jpg"
    SAVE_PATH = r"/home/fibo/sc171_monitor_project/data/test_results_detector/detected_bus.jpg"
    YOLO = YoloDetectorSC171()
    img = cv2.imread(IMG_PATH)
    YOLO.detect(img)
    YOLO.draw_results(SAVE_PATH)
    result = YOLO.save_results(TRIGGER_CLASSES_FOR_GEMINI)
    print(result)
    IMG_PATH = "data/test_images/person.jpg"
    SAVE_PATH = r"/home/fibo/sc171_monitor_project/data/test_results_detector/detected_person.jpg"
    img = cv2.imread(IMG_PATH)
    YOLO.detect(img)
    YOLO.draw_results(SAVE_PATH)
    result = YOLO.save_results(TRIGGER_CLASSES_FOR_GEMINI)
    print(result)
    YOLO.close()
    print("检测完成，结果已保存。")