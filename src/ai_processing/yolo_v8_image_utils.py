# ~/sc171_monitor_project/src/ai_processing/yolo_v8_image_utils.py
import cv2
import numpy as np
import os
# 从项目根目录的config_sc171导入COCO_CLASSES
# 为了让这个模块能被其他模块（如测试脚本）正确导入config，
# 通常测试脚本会从项目根目录运行，或者PYTHONPATH包含项目根目录。
# 假设运行环境能找到顶层的config_sc171
try:
    from config_sc171 import COCO_CLASSES
except ImportError:
    # 如果直接运行此文件进行单元测试，可能需要调整路径或提供一个默认的COCO_CLASSES
    print("警告: 无法从config_sc171导入COCO_CLASSES。将使用默认的精简列表进行测试。")
    COCO_CLASSES = ['person', 'car', 'dog'] # 仅用于本文件独立测试时的后备

def preprocess_image(
    image_bgr: np.ndarray, 
    target_width: int, 
    target_height: int, 
    input_layout: str = "NCHW" # YOLOv8 ONNX通常是NCHW
) -> np.ndarray or None:
    """
    为YOLOv8模型预处理输入图像。
    Args:
        image_bgr: OpenCV读取的BGR格式图像 (H, W, C)。
        target_width: 模型期望的输入宽度。
        target_height: 模型期望的输入高度。
        input_layout: 模型期望的输入布局 ("NCHW" 或 "NHWC")。
    Returns:
        预处理后的图像NumPy数组 (float32)，准备好作为模型输入。
        如果发生错误则返回None。
    """
    if image_bgr is None:
        print("错误 [preprocess_image]: 输入图像为None。")
        return None
    if not isinstance(image_bgr, np.ndarray):
        print(f"错误 [preprocess_image]: 输入图像类型不是NumPy数组，而是{type(image_bgr)}。")
        return None

    # print(f"DEBUG [preprocess_image]: 原始图像形状: {image_bgr.shape}")

    # 1. Resize图像到模型输入尺寸
    # cv2.resize 的 dsize 参数是 (width, height)
    try:
        resized_image = cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"错误 [preprocess_image]: cv2.resize失败: {e}")
        return None
    # print(f"DEBUG [preprocess_image]: Resize后形状: {resized_image.shape}")


    # 2. BGR -> RGB (YOLO模型通常在RGB图像上训练)
    try:
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"错误 [preprocess_image]: BGR到RGB转换失败: {e}")
        return None

    # 3. 归一化到0-1范围并将数据类型转换为float32
    normalized_image = rgb_image.astype(np.float32) / 255.0

    # 4. 根据指定的input_layout调整维度顺序并增加批次维度
    if input_layout.upper() == "NCHW":
        # 从 HWC (Height, Width, Channels) 转换为 CHW (Channels, Height, Width)
        chw_image = np.transpose(normalized_image, (2, 0, 1))
        # 增加批次维度 NCHW (1, Channels, Height, Width)
        input_tensor = np.expand_dims(chw_image, axis=0)
    elif input_layout.upper() == "NHWC":
        # 直接增加批次维度 NHWC (1, Height, Width, Channels)
        input_tensor = np.expand_dims(normalized_image, axis=0)
    else:
        print(f"错误 [preprocess_image]: 不支持的输入布局 '{input_layout}'。请使用 'NCHW' 或 'NHWC'。")
        return None
    
    # print(f"DEBUG [preprocess_image]: 最终输入张量形状: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    return input_tensor.astype(np.float32) # 再次确保是float32


def draw_detections_on_image(
    image_bgr: np.ndarray, 
    detections: list, 
    class_names: list = None # 可选，如果为None则只显示class_id
) -> np.ndarray:
    """
    在图像上绘制检测到的边界框和标签。
    Args:
        image_bgr: 原始BGR图像 (H, W, C)。
        detections: 检测结果列表，每个元素是一个字典，期望包含：
                    'class_id': int,
                    'class_name': str (可选，如果class_names提供了，会用这个),
                    'confidence': float,
                    'bbox_xywh': [x_top_left, y_top_left, width, height]
        class_names: COCO类别名称列表，用于通过class_id查找名称。
    Returns:
        绘制了检测结果的图像副本。
    """
    display_image = image_bgr.copy()
    if class_names is None:
        class_names = COCO_CLASSES # 使用模块内定义的后备列表

    for det in detections:
        try:
            class_id = int(det.get('class_id', -1))
            confidence = float(det.get('confidence', 0.0))
            bbox = det.get('bbox_xywh')

            if bbox is None or len(bbox) != 4:
                # print(f"警告 [draw_detections]: 检测结果缺少有效的bbox: {det}")
                continue

            x, y, w, h = map(int, bbox) # 确保是整数像素坐标

            # 获取类别名称
            label_name = str(class_id) # 默认显示ID
            if 'class_name' in det:
                label_name = det['class_name']
            elif 0 <= class_id < len(class_names):
                label_name = class_names[class_id]
            
            label = f"{label_name}: {confidence:.2f}"

            # 绘制边界框
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制标签背景和文本
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display_image, (x, y - label_height - baseline), (x + label_width, y), (0, 255, 0), cv2.FILLED)
            cv2.putText(display_image, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        except Exception as e:
            print(f"警告 [draw_detections]: 绘制检测结果时发生错误 for det={det}: {e}")
            continue
            
    return display_image


# --- (可选) 模块级测试代码 ---
if __name__ == '__main__':
    print("--- yolo_v8_image_utils.py 模块测试 ---")

    # 1. 测试 preprocess_image
    print("\n[测试 preprocess_image]")
    dummy_bgr_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) # HWC, BGR
    
    target_w, target_h = 320, 320
    
    # 测试 NCHW 布局
    preprocessed_nchw = preprocess_image(dummy_bgr_image, target_w, target_h, input_layout="NCHW")
    if preprocessed_nchw is not None:
        print(f"  NCHW 输出形状: {preprocessed_nchw.shape}, dtype: {preprocessed_nchw.dtype}")
        assert preprocessed_nchw.shape == (1, 3, target_h, target_w), "NCHW shape 错误"
        assert preprocessed_nchw.dtype == np.float32, "NCHW dtype 错误"
        assert np.max(preprocessed_nchw) <= 1.0 and np.min(preprocessed_nchw) >= 0.0, "NCHW 值范围错误"
        print("  NCHW 预处理测试通过。")
    else:
        print("  NCHW 预处理失败。")

    # 测试 NHWC 布局
    preprocessed_nhwc = preprocess_image(dummy_bgr_image, target_w, target_h, input_layout="NHWC")
    if preprocessed_nhwc is not None:
        print(f"  NHWC 输出形状: {preprocessed_nhwc.shape}, dtype: {preprocessed_nhwc.dtype}")
        assert preprocessed_nhwc.shape == (1, target_h, target_w, 3), "NHWC shape 错误"
        assert preprocessed_nhwc.dtype == np.float32, "NHWC dtype 错误"
        assert np.max(preprocessed_nhwc) <= 1.0 and np.min(preprocessed_nhwc) >= 0.0, "NHWC 值范围错误"
        print("  NHWC 预处理测试通过。")
    else:
        print("  NHWC 预处理失败。")

    # 2. 测试 draw_detections_on_image
    print("\n[测试 draw_detections_on_image]")
    # 创建一个用于绘制的虚拟图像副本
    draw_test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    dummy_detections = [
        {'class_id': 0, 'class_name': 'person', 'confidence': 0.95, 'bbox_xywh': [50, 50, 100, 200]},
        {'class_id': 2, 'confidence': 0.80, 'bbox_xywh': [200, 100, 80, 80]}, # 无class_name，应使用COCO_CLASSES
        {'class_id': 99, 'confidence': 0.75, 'bbox_xywh': [300, 150, 60, 120]},# class_id超出范围
        {'bbox_xywh': [400, 200, 50, 50]}, # 缺少其他信息
        {'class_id': 1, 'confidence': 0.85} # 缺少bbox
    ]
    
    # 使用COCO_CLASSES (通过模块顶部的导入或后备列表)
    result_image = draw_detections_on_image(draw_test_image, dummy_detections) 
    
    if result_image is not None:
        print("  draw_detections_on_image 执行完成。")
        # 在实际SC171上，imshow可能无法工作，除非有X11转发或桌面环境
        # 可以选择将结果保存到文件进行检查
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/test_draw_output.jpg")
        # 确保data目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            cv2.imwrite(save_path, result_image)
            print(f"  绘制结果已保存到: {save_path}")
        except Exception as e:
            print(f"  保存绘制结果失败: {e}")
    else:
        print("  draw_detections_on_image 返回 None。")

    print("\n--- yolo_v8_image_utils.py 模块测试完成 ---")