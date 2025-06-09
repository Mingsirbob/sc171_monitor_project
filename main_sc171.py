# ~/sc171_monitor_project/main_sc171.py
import cv2
import time
import os
import uuid # 用于生成Event ID
from datetime import datetime, timezone # 用于Event时间戳

# --- 1. 导入配置和自定义模块 ---
try:
    import config_sc171 as cfg
    from src.video_io.camera_handler_sc171 import CameraHandlerSC171
    from src.ai_processing.yolo_detector_sc171 import YoloDetectorSC171
    from src.ai_processing.gemini_analyzer_cloud import GeminiAnalyzerCloud # <--- 新增导入
    from src.data_management.event_models_shared import Event, DetectedObject # <--- 新增导入
    from src.ai_processing.yolo_v8_image_utils import draw_detections_on_image
    # 从api_infer_wrapper导入Runtime枚举类，以便指定GPU
    from src.ai_processing.api_infer_wrapper import Runtime 

except ImportError as e:
    print(f"关键导入错误: {e}") # ... (错误提示保持不变) ...
    exit()

# --- 全局控制参数 ---
SAVE_RESULT_FRAMES = True
SAVE_FRAME_INTERVAL = 30 
FRAMES_OUTPUT_DIR = os.path.join(cfg.DATA_DIR, "main_output_frames_v2") # 版本2的输出目录
SNAPSHOTS_DIR = os.path.join(cfg.DATA_DIR, "event_snapshots") # 保存Gemini触发快照的目录

# Gemini触发相关配置 (可以从config_sc171.py读取，这里为方便演示先定义)
TRIGGER_CLASSES_FOR_GEMINI = getattr(cfg, 'TRIGGER_CLASSES_FOR_GEMINI', ["person", "fire"]) # 默认为person和fire
MIN_CONFIDENCE_FOR_GEMINI_TRIGGER = getattr(cfg, 'MIN_CONFIDENCE_FOR_GEMINI_TRIGGER', 0.65)
GEMINI_COOLDOWN_SECONDS = getattr(cfg, 'GEMINI_COOLDOWN_SECONDS', 30) # 默认30秒冷却

def main():
    print("--- SC171监控智能识别项目 - 主程序启动 (V2 - 集成Gemini与Event) ---")
    
    if SAVE_RESULT_FRAMES:
        os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
        print(f"带检测框的结果帧将保存到: {FRAMES_OUTPUT_DIR}")
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True) # 确保快照目录存在
    print(f"事件快照将保存到: {SNAPSHOTS_DIR}")


    # --- 2. 初始化核心模块 ---
    print("\n[初始化摄像头模块...]")
    camera = CameraHandlerSC171(camera_source=cfg.SC171_CAMERA_SOURCE, desired_fps=cfg.DESIRED_FPS)
    if not camera.open(): # ... (摄像头错误处理保持不变) ...
        return
    actual_cam_w, actual_cam_h = camera.get_resolution()
    actual_cam_fps = camera.get_fps()
    print(f"摄像头实际参数: {actual_cam_w}x{actual_cam_h} @ {actual_cam_fps:.2f} FPS")

    print("\n[初始化YOLO检测器模块 (GPU模式)...]")
    yolo_detector = YoloDetectorSC171(runtime_target=Runtime.GPU)
    if not yolo_detector.model_loaded: # ... (YOLO错误处理保持不变) ...
        camera.close(); return

    # --- 新增：初始化Gemini分析器 ---
    print("\n[初始化Gemini分析器模块...]")
    gemini_analyzer = GeminiAnalyzerCloud()
    if not gemini_analyzer.is_initialized:
        print("警告：Gemini分析器初始化失败。云分析功能将不可用。")
        # 程序可以继续运行，但不会进行Gemini分析

    print("\n--- 所有核心模块初始化完成，开始主循环 ---")
    frame_counter = 0
    last_gemini_trigger_time = 0.0 # 初始化冷却计时器

    try:
        while True:
            ret, frame_bgr = camera.read_frame()
            if not ret or frame_bgr is None: # ... (读取帧错误处理保持不变) ...
                break
            frame_counter += 1
            current_process_time_start = time.perf_counter()

            # --- YOLO目标检测 ---
            yolo_detections = yolo_detector.detect(frame_bgr) # 返回的是字典列表

            # --- Gemini触发与事件创建逻辑 ---
            trigger_gemini_now = False
            yolo_objects_for_event = [] # 存储当前帧所有YOLO检测对象，用于Event记录
            triggering_yolo_objects_summary = [] # 存储触发Gemini的对象的简要信息，用于Prompt

            if yolo_detections:
                for det in yolo_detections:
                    # 将YOLO检测结果转换为DetectedObject实例并存入列表
                    try:
                        detected_obj = DetectedObject(
                            class_id=det['class_id'],
                            class_name=det['class_name'],
                            confidence=det['confidence'],
                            bbox_xywh=det['bbox_xywh']
                        )
                        yolo_objects_for_event.append(detected_obj)

                        # 检查是否满足Gemini触发条件
                        if det['class_name'] in TRIGGER_CLASSES_FOR_GEMINI and \
                           det['confidence'] >= MIN_CONFIDENCE_FOR_GEMINI_TRIGGER:
                            trigger_gemini_now = True
                            triggering_yolo_objects_summary.append(f"{det['class_name']} (conf: {det['confidence']:.2f})")
                    except KeyError as ke:
                        print(f"警告: YOLO检测结果字典缺少键: {ke} - {det}")
                    except Exception as e_obj:
                        print(f"警告: 处理YOLO检测对象时出错: {e_obj} - {det}")
            
            current_timestamp = time.time()
            event_instance = None # 初始化event_instance

            if trigger_gemini_now and \
               (current_timestamp - last_gemini_trigger_time > GEMINI_COOLDOWN_SECONDS) and \
               gemini_analyzer.is_initialized: # 确保Gemini分析器已初始化
                
                print(f"\n帧 {frame_counter}: **触发Gemini分析** (检测到: {', '.join(triggering_yolo_objects_summary)})")
                last_gemini_trigger_time = current_timestamp # 更新冷却时间戳

                # 1. 保存图像快照 (推荐)
                snapshot_filename = f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
                snapshot_saved_successfully = False
                try:
                    if cv2.imwrite(snapshot_path, frame_bgr): # 保存原始BGR帧
                        print(f"  图像快照已保存到: {snapshot_path}")
                        snapshot_saved_successfully = True
                    else:
                        print(f"  错误：保存图像快照到 {snapshot_path} 失败 (cv2.imwrite返回False)。")
                        snapshot_path = None # 如果保存失败，路径设为None
                except Exception as e_snap:
                    print(f"  错误：保存图像快照时发生异常: {e_snap}")
                    snapshot_path = None

                # 2. 构造Prompt (可以更智能地基于triggering_yolo_objects_summary)
                prompt_text_for_gemini = (
                    f"分析以下图片中的场景。图片中可能检测到了以下值得注意的对象：{', '.join(triggering_yolo_objects_summary)}。\n"
                    "评估当前场景的潜在风险，并给出风险评级。\n"
                    "同时，简要描述图片中的主要事件内容。\n"
                    "请严格以JSON格式返回结果，该JSON对象应包含两个键：\n"
                    "1. 'risk_level': 字符串，值为'低'、'中'或'高'中的一个 (如果无法判断或无风险，则为'未知'或'低')。\n"
                    "2. 'description': 字符串，为事件的详细文本描述。\n"
                    "不要包含任何JSON之外的解释性文字或markdown标记。"
                )
                # print(f"  发送给Gemini的Prompt: {prompt_text_for_gemini}") # 调试时取消注释

                # 3. 调用Gemini分析
                gemini_risk, gemini_desc = gemini_analyzer.analyze_image(frame_bgr, prompt_text_for_gemini)
                gemini_call_ts = datetime.now(timezone.utc).isoformat()

                # 4. 创建Event对象
                cam_id_str = f"cam_{cfg.SC171_CAMERA_SOURCE}" if isinstance(cfg.SC171_CAMERA_SOURCE, int) else str(cfg.SC171_CAMERA_SOURCE)
                event_instance = Event(
                    camera_id=cam_id_str,
                    detected_yolo_objects=yolo_objects_for_event, # 包含当前帧所有YOLO检测
                    triggering_image_snapshot_path=snapshot_path if snapshot_saved_successfully else None,
                    gemini_analysis_prompt=prompt_text_for_gemini,
                    gemini_risk_level=gemini_risk if gemini_risk else "分析失败",
                    gemini_description=gemini_desc if gemini_desc else "未能获取描述",
                    gemini_api_call_timestamp_utc=gemini_call_ts
                )
                print(f"  Gemini分析结果 - 风险: {event_instance.gemini_risk_level}, 描述 (部分): {str(event_instance.gemini_description)[:80]}...")
                print(f"  新事件已创建: ID {event_instance.event_id}")
            
            # --- 打印单帧处理总耗时 ---
            current_process_time_end = time.perf_counter()
            total_frame_processing_time_ms = (current_process_time_end - current_process_time_start) * 1000
            if not trigger_gemini_now or not gemini_analyzer.is_initialized or (current_timestamp - last_gemini_trigger_time <= GEMINI_COOLDOWN_SECONDS and last_gemini_trigger_time != 0.0) : # 如果没有触发Gemini或在冷却期
                 print(f"帧 {frame_counter}: YOLO检测到 {len(yolo_detections)} 个对象 (总耗时: {total_frame_processing_time_ms:.2f} ms)")


            # --- 可视化与保存结果帧 (与之前类似) ---
            display_frame = frame_bgr.copy()
            if yolo_detections:
                class_names_to_use = cfg.COCO_CLASSES if hasattr(cfg, 'COCO_CLASSES') else class_names_from_util
                display_frame = draw_detections_on_image(display_frame, yolo_detections, class_names_to_use)
            
            # 添加Gemini风险等级到显示帧 (如果分析了)
            if event_instance and event_instance.gemini_risk_level != "未分析":
                cv2.putText(display_frame, f"Gemini Risk: {event_instance.gemini_risk_level}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if SAVE_RESULT_FRAMES and (frame_counter % SAVE_FRAME_INTERVAL == 0):
                # ... (保存逻辑与之前一致) ...
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(FRAMES_OUTPUT_DIR, f"output_frame_{timestamp_str}_{frame_counter}.jpg")
                try:
                    cv2.imwrite(save_path, display_frame)
                    # print(f"结果帧已保存到: {save_path}") # 减少打印
                except Exception as e_save:
                    print(f"保存结果帧失败: {e_save}")
            
            # cv2.imshow / cv2.waitKey (在SC171上通常不直接使用)

    except KeyboardInterrupt: # ... (异常处理保持不变) ...
        print("\n用户通过Ctrl+C请求中断程序...")
    except Exception as e:
        print(f"\n主循环中发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
    finally: # ... (资源释放保持不变，确保所有模块的close都被调用) ...
        print("\n开始清理和释放资源...")
        if 'yolo_detector' in locals() and hasattr(yolo_detector, 'model_loaded') and yolo_detector.model_loaded:
            yolo_detector.close()
        if 'camera' in locals() and hasattr(camera, 'is_opened') and camera.is_opened:
            camera.close()
        # GeminiAnalyzerCloud 目前没有需要显式关闭的资源
        print("--- SC171监控智能识别项目 - 主程序结束 ---")

if __name__ == '__main__':
    if not os.getenv("FIBO_LIB"): # ... (FIBO_LIB检查保持不变) ...
        pass
    if hasattr(cfg, 'ensure_directories_exist'): # ... (目录确保保持不变) ...
        cfg.ensure_directories_exist()
    main()