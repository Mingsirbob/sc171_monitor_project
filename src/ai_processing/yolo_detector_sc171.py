# ~/sc171_monitor_project/src/ai_processing/yolo_detector_sc171.py
import numpy as np
import time # 用于计时和可能的调试
import cv2 # 用于测试块中的图像加载和可选的绘制
import os

# 从项目根目录的config_sc171导入配置
# 确保在运行此模块的测试时，PYTHONPATH已正确设置或使用 -m 选项
try:
    from config_sc171 import (
        YOLO_DLC_PATH, YOLO_MODEL_INPUT_NAME, YOLO_MODEL_OUTPUT_NAMES,
        YOLO_MODEL_INPUT_WIDTH, YOLO_MODEL_INPUT_HEIGHT, YOLO_MODEL_INPUT_LAYOUT,
        YOLO_MODEL_OUTPUT_EXPECTED_SHAPE,
        YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, COCO_CLASSES
    )
except ImportError:
    print("错误 [YoloDetectorSC171]: 无法从config_sc171导入配置。")
    print("请确保config_sc171.py在PYTHONPATH中，或者从项目根目录使用 'python3 -m src.ai_processing.yolo_detector_sc171' 运行测试。")
    # 提供后备值以便模块至少可以被导入，但功能会受限
    YOLO_DLC_PATH = "models/yolov8n.dlc" # 示例，实际应来自config
    YOLO_MODEL_INPUT_NAME = "images"
    YOLO_MODEL_OUTPUT_NAMES = ["output0"]
    YOLO_MODEL_INPUT_WIDTH = 640
    YOLO_MODEL_INPUT_HEIGHT = 640
    YOLO_MODEL_INPUT_LAYOUT = "NCHW"
    YOLO_MODEL_OUTPUT_EXPECTED_SHAPE = (1, 84, 8400)
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    COCO_CLASSES = ["person"] # 最小示例

# 从同级包导入我们之前创建的模块
try:
    from .api_infer_wrapper import SnpeContext, Runtime, PerfProfile, LogLevel
    from .yolo_v8_image_utils import preprocess_image
    from .yolo_v8_snpe_postprocessor import apply_yolov8_snpe_postprocessing
except ImportError: # 处理直接运行此文件进行测试时可能发生的相对导入问题
    print("警告 [YoloDetectorSC171]: 尝试使用相对导入失败，可能是因为直接运行此文件。")
    print("将尝试从上一级目录的模块导入（适用于'python3 -m'运行方式）。")
    from ..ai_processing.api_infer_wrapper import SnpeContext, Runtime, PerfProfile, LogLevel
    from ..ai_processing.yolo_v8_image_utils import preprocess_image
    from ..ai_processing.yolo_v8_snpe_postprocessor import apply_yolov8_snpe_postprocessing


class YoloDetectorSC171:
    def __init__(self, 
                 dlc_path: str = YOLO_DLC_PATH,
                 input_name: str = YOLO_MODEL_INPUT_NAME,
                 output_names: list = None, # 允许覆盖配置中的输出名
                 model_input_width: int = YOLO_MODEL_INPUT_WIDTH,
                 model_input_height: int = YOLO_MODEL_INPUT_HEIGHT,
                 input_layout: str = YOLO_MODEL_INPUT_LAYOUT,
                 expected_output_shape: tuple = YOLO_MODEL_OUTPUT_EXPECTED_SHAPE,
                 conf_threshold: float = YOLO_CONF_THRESHOLD,
                 iou_threshold: float = YOLO_IOU_THRESHOLD,
                 class_names: list = None, # 允许覆盖COCO_CLASSES
                 runtime_target: str = Runtime.DSP, # 默认为DSP，可以更改
                 perf_profile: int = PerfProfile.BALANCED,
                 log_level: str = LogLevel.INFO):
        """
        初始化YOLOv8检测器，使用FIBO AISDK (SNPE) 在SC171上运行。
        参数从config_sc171.py获取默认值，但可以被覆盖。
        """
        self.dlc_path = dlc_path
        self.model_input_name = input_name
        self.model_output_names = output_names if output_names is not None else YOLO_MODEL_OUTPUT_NAMES
        self.model_input_width = model_input_width
        self.model_input_height = model_input_height
        self.input_layout = input_layout
        self.expected_output_shape = expected_output_shape
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names if class_names is not None else COCO_CLASSES

        self.snpe_context = None
        self.model_loaded = False

        print(f"YoloDetectorSC171: 尝试初始化SNPE模型...")
        print(f"  DLC路径: {self.dlc_path}")
        print(f"  运行时: {runtime_target}")
        print(f"  输入节点名: {self.model_input_name}")
        print(f"  输出节点名(预期从SNPE获取): {self.model_output_names}")
        print(f"  模型输入尺寸: {self.model_input_width}x{self.model_input_height}")
        print(f"  输入布局: {self.input_layout}")

        try:
            self.snpe_context = SnpeContext(
                dlc_path=self.dlc_path,
                # output_tensors 参数传递给SnpeContext，用于其内部配置或FetchOutputs
                # 如果FIBO SDK的SnpeContext.Initialize()的配置需要知道输出名，
                # 且SnpeContext.Execute()的output_names_to_fetch是独立的，
                # 这里的 self.model_output_names 将主要用于 Execute 和后处理。
                output_tensors=self.model_output_names, 
                runtime=runtime_target,
                profile_level=perf_profile,
                log_level=log_level
            )
            
            if self.snpe_context.Initialize() == 0:
                print("YoloDetectorSC171: SNPE上下文初始化成功。")
                self.model_loaded = True
                # 你可以在这里尝试获取并打印模型的实际输入输出信息（如果FIBO SDK支持）
                # 例如：
                # actual_inputs = self.snpe_context.m_context.GetInputTensorInfo() # 假设有此API
                # actual_outputs = self.snpe_context.m_context.GetOutputTensorInfo()
                # print(f"  SNPE报告的实际输入: {actual_inputs}")
                # print(f"  SNPE报告的实际输出: {actual_outputs}")
            else:
                print(f"错误 [YoloDetectorSC171]: SNPE上下文初始化失败。")
                # 可以在这里尝试获取更详细的FIBO SDK错误

        except FileNotFoundError:
            print(f"错误 [YoloDetectorSC171]: DLC模型文件 '{self.dlc_path}' 未找到。")
        except EnvironmentError as e: # 捕获FIBO_LIB相关的错误
            print(f"错误 [YoloDetectorSC171]: 初始化因环境问题失败: {e}")
        except Exception as e:
            print(f"错误 [YoloDetectorSC171]: 初始化过程中发生未知异常: {e}")
            import traceback
            traceback.print_exc() # 打印完整的堆栈跟踪
        
        if not self.model_loaded:
            print("YoloDetectorSC171: 模型未能成功加载。detect()方法将不可用。")


    def detect(self, image_bgr: np.ndarray) -> list:
        """
        使用加载的SNPE模型对输入的BGR图像帧进行目标检测。
        Args:
            image_bgr: OpenCV读取的BGR格式图像 (H, W, C)。
        Returns:
            检测结果列表。每个检测结果是一个字典，包含:
            {'class_id': int, 'class_name': str, 'confidence': float, 
             'bbox_xywh': [x_top_left, y_top_left, width, height]} (坐标相对于原始图像)。
            如果检测失败或无结果，返回空列表。
        """
        if not self.model_loaded or self.snpe_context is None:
            # print("警告 [YoloDetectorSC171.detect]: 模型未加载，无法执行检测。") # 减少重复打印
            return []

        if image_bgr is None:
            print("错误 [YoloDetectorSC171.detect]: 输入图像为None。")
            return []

        original_height, original_width = image_bgr.shape[:2]

        # 1. 图像预处理
        input_tensor = preprocess_image(
            image_bgr,
            target_width=self.model_input_width,
            target_height=self.model_input_height,
            input_layout=self.input_layout
        )
        if input_tensor is None:
            print("错误 [YoloDetectorSC171.detect]: 图像预处理失败。")
            return []

        # 2. 构造输入feed
        input_feed = {self.model_input_name: input_tensor}

        # 3. 执行SNPE推理
        # SnpeContext.Execute的第二个参数是希望获取的输出节点名列表
        raw_outputs_dict = self.snpe_context.Execute(self.model_output_names, input_feed)

        if raw_outputs_dict is None:
            print("错误 [YoloDetectorSC171.detect]: SNPE模型推理返回None。")
            return []
        
        # 4. 后处理SNPE输出
        detections = apply_yolov8_snpe_postprocessing(
            raw_output_data_dict=raw_outputs_dict,
            model_output_names=self.model_output_names, # 后处理函数会用这个来取数据
            expected_output_shape=self.expected_output_shape,
            original_image_h=original_height,
            original_image_w=original_width,
            model_input_h=self.model_input_height,
            model_input_w=self.model_input_width,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            class_names=self.class_names
        )
        
        return detections

    def close(self):
        """
        释放SNPE上下文资源。
        """
        if self.snpe_context and self.model_loaded:
            print("YoloDetectorSC171: 正在释放SNPE上下文...")
            self.snpe_context.Release()
            self.model_loaded = False # 标记为已释放
            print("YoloDetectorSC171: SNPE上下文已成功释放。")
        # else:
            # print("YoloDetectorSC171: SNPE上下文未初始化或已释放，无需操作。")


# --- 模块级测试代码 ---
if __name__ == '__main__':
    print("--- YoloDetectorSC171 模块测试 ---")
    # 确保FIBO环境脚本已执行!

    # 测试图片路径 (相对于项目根目录的data/test_images/)
    # 使用os.path.abspath和__file__来构建更可靠的路径，
    # 以便直接运行此文件或通过 `python -m` 运行都能找到。
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, '../../')) # 假设此文件在 src/ai_processing/
    
    # 如果直接运行此文件，config_sc171 可能无法通过顶层导入找到，
    # 所以测试时可以直接使用配置值或确保PYTHONPATH正确
    # 我们在类定义中已经处理了config的导入，这里主要用于测试图片路径
    test_image_filename = "bus.jpg" # 确保这张图片在 data/test_images/ 目录下
    test_image_path = os.path.join(project_root_dir, "data", "test_images", test_image_filename)

    if not os.path.exists(YOLO_DLC_PATH):
        print(f"错误：测试前请确保DLC模型文件存在于: {YOLO_DLC_PATH}")
        exit()
    if not os.path.exists(test_image_path):
        print(f"错误：测试前请确保测试图片存在于: {test_image_path}")
        exit()

    print(f"将使用模型: {YOLO_DLC_PATH}")
    print(f"将使用测试图片: {test_image_path}")

    # 初始化检测器 - 先用CPU或GPU测试，确保SNPE流程通畅
    # runtime_for_test = Runtime.CPU
    runtime_for_test = Runtime.DSP # 如果GPU在SC171上配置简单
    # runtime_for_test = Runtime.DSP # 最终目标，但DSP配置和权限可能更复杂

    print(f"\n尝试使用运行时: {runtime_for_test}")
    detector = YoloDetectorSC171(runtime_target=runtime_for_test)

    if not detector.model_loaded:
        print("YOLO检测器未能成功初始化。测试中止。")
    else:
        print("\n加载测试图片...")
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"错误：无法加载测试图片 '{test_image_path}'")
        else:
            print(f"测试图片加载成功，形状: {image.shape}")
            
            print("\n开始第一次检测...")
            start_infer_time = time.perf_counter()
            detections1 = detector.detect(image)
            end_infer_time = time.perf_counter()
            print(f"第一次检测耗时: {(end_infer_time - start_infer_time) * 1000:.2f} ms")

            if detections1:
                print(f"第一次检测到 {len(detections1)} 个对象:")
                for i, det in enumerate(detections1[:5]): # 最多打印前5个
                    print(f"  Obj {i+1}: 类='{det['class_name']}'(ID:{det['class_id']}), "
                          f"Conf={det['confidence']:.2f}, BBoxXYWH={det['bbox_xywh']}")
            else:
                print("第一次检测未找到任何对象。")

            # (可选) 进行第二次检测，观察是否有初始化开销后的性能差异
            if image is not None: # 确保图片仍然加载
                print("\n开始第二次检测 (可能更快，无初始化开销)...")
                start_infer_time_2 = time.perf_counter()
                detections2 = detector.detect(image) # 使用相同的图片再次检测
                end_infer_time_2 = time.perf_counter()
                print(f"第二次检测耗时: {(end_infer_time_2 - start_infer_time_2) * 1000:.2f} ms")
                # 可以比较detections1和detections2是否一致（理论上应该一致）


            # (可选) 可视化结果并保存 (如果需要)
            # from .yolo_v8_image_utils import draw_detections_on_image # 确保导入
            # if detections1:
            #     result_display_image = draw_detections_on_image(image.copy(), detections1, COCO_CLASSES)
            #     save_path = os.path.join(project_root_dir, "data", "yolo_detector_sc171_test_output.jpg")
            #     cv2.imwrite(save_path, result_display_image)
            #     print(f"检测结果图像已保存到: {save_path}")


        print("\n关闭检测器...")
        detector.close()

    print("\n--- YoloDetectorSC171 模块测试完成 ---")