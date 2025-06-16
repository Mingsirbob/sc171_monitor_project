# ~/sc171_monitor_project/main_sc171.py
import cv2
import time
import os
import uuid 
from datetime import datetime, timezone

# --- 1. 导入配置和自定义模块 ---
try:
    import config_sc171 as cfg
    from src.video_io.camera_handler_sc171 import CameraHandlerSC171
    from src.video_io.video_buffer_sc171 import VideoBufferSC171
    from src.ai_processing.yolo_detector_sc171 import YoloDetectorSC171
    from src.ai_processing.gemini_analyzer_cloud import GeminiAnalyzerCloud
    from src.data_management.event_models_shared import Event, SimplifiedDetectedObject
    from src.data_management.event_logger_local import EventLoggerLocal
    # from src.communication_protocols.server_publisher_http import ServerPublisherHTTP # <--- 注释掉或删除
    from src.ai_processing.yolo_v8_image_utils import draw_detections_on_image
    from src.ai_processing.api_infer_wrapper import Runtime 

except ImportError as e:
    print(f"关键导入错误: {e}")
    exit()

# --- 全局控制参数 ---
SAVE_RESULT_FRAMES = True
SAVE_FRAME_INTERVAL = 90
FRAMES_OUTPUT_DIR = os.path.join(cfg.DATA_DIR, "main_output_frames_v2.3_no_publish") # 更新输出目录
SNAPSHOTS_DIR = os.path.join(cfg.DATA_DIR, "event_snapshots")

TRIGGER_CLASSES_FOR_GEMINI = getattr(cfg, 'TRIGGER_CLASSES_FOR_GEMINI', ["person", "fire"])
MIN_CONFIDENCE_FOR_GEMINI_TRIGGER = getattr(cfg, 'MIN_CONFIDENCE_FOR_GEMINI_TRIGGER', 0.65)
GEMINI_COOLDOWN_SECONDS = getattr(cfg, 'GEMINI_COOLDOWN_SECONDS', 30)
# PUSH_RISK_LEVELS_TO_SERVER = getattr(cfg, 'PUSH_RISK_LEVELS_TO_SERVER', ["高", "中"]) # <--- 注释掉或删除

def main():
    print("--- SC171监控智能识别项目 - 主程序启动 (V2.3 - 无服务器推送) ---")
    
    if SAVE_RESULT_FRAMES: # ... (目录创建不变)
        os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    # --- 2. 初始化核心模块 ---
    print("\n[初始化摄像头模块...]") # ... (摄像头初始化不变)
    camera = CameraHandlerSC171(camera_source=cfg.SC171_CAMERA_SOURCE, desired_fps=cfg.DESIRED_FPS)
    if not camera.open(): return
    actual_cam_w, actual_cam_h = camera.get_resolution()
    actual_cam_fps = camera.get_fps()
    if not actual_cam_w or not actual_cam_h or actual_cam_fps <= 0: camera.close(); return
    print(f"摄像头实际参数: {actual_cam_w}x{actual_cam_h} @ {actual_cam_fps:.2f} FPS")

    print("\n[初始化YOLO检测器模块 (GPU模式)...]") # ... (YOLO初始化不变)
    yolo_detector = YoloDetectorSC171(runtime_target=Runtime.GPU)
    if not yolo_detector.model_loaded: camera.close(); return

    print("\n[初始化Gemini分析器模块...]") # ... (Gemini初始化不变)
    gemini_analyzer = GeminiAnalyzerCloud()
    if not gemini_analyzer.is_initialized: print("警告：Gemini分析器初始化失败。云分析功能将不可用。")

    print("\n[初始化本地事件记录器模块...]") # ... (EventLogger初始化不变)
    event_logger = EventLoggerLocal(log_directory=cfg.LOG_DIR)
    if event_logger.log_directory is None: print("警告：本地事件记录器未能正确初始化日志目录。事件可能无法记录。")

    # --- 移除ServerPublisherHTTP的初始化 ---
    # print("\n[初始化服务器事件推送器模块...]")
    # server_publisher = ServerPublisherHTTP(
    #     server_url=cfg.SERVER_EVENT_PUBLISH_ENDPOINT, 
    #     auth_token=cfg.SERVER_AUTH_TOKEN           
    # )
    # if not server_publisher.is_configured:
    #     print("警告：服务器推送器URL未配置。事件将无法推送到服务器。")
    # ------------------------------------

    print("\n[初始化视频缓冲模块...]") # ... (VideoBuffer初始化不变)
    cam_id_str_for_buffer_and_event = f"cam_{cfg.SC171_CAMERA_SOURCE}" if isinstance(cfg.SC171_CAMERA_SOURCE, int) else str(cfg.SC171_CAMERA_SOURCE).replace('/','_').replace('\\','_')
    video_buffer = VideoBufferSC171(
        camera_id=cam_id_str_for_buffer_and_event, 
        cache_duration_seconds=cfg.VIDEO_CACHE_DURATION_MINUTES * 60,
        output_directory_root=cfg.VIDEO_CACHE_DIR,
        fps=actual_cam_fps, frame_width=actual_cam_w, frame_height=actual_cam_h
    )
    if not video_buffer.current_video_writer: camera.close(); yolo_detector.close(); return

    print("\n--- 所有核心模块初始化完成，开始主循环 ---")
    frame_counter = 0
    last_gemini_trigger_time = 0.0

    try:
        while True:
            ret, frame_bgr = camera.read_frame() # ... (循环和帧处理逻辑基本不变) ...
            if not ret or frame_bgr is None: break
            if frame_bgr.shape[1] != actual_cam_w or frame_bgr.shape[0] != actual_cam_h:
                try: frame_bgr = cv2.resize(frame_bgr, (actual_cam_w, actual_cam_h))
                except Exception as e_resize_main: print(f"错误: resize帧失败: {e_resize_main}"); continue
            frame_counter += 1
            current_process_time_start = time.perf_counter()

            completed_segment_path = video_buffer.add_frame(frame_bgr.copy())
            if completed_segment_path: # ... (视频片段保存打印不变) ...
                print(f"\n========= 视频片段已完成并保存: {os.path.basename(completed_segment_path)} =========")
                print(f"  完整路径: {completed_segment_path}")
                print("=======================================================================\n")

            yolo_detections = yolo_detector.detect(frame_bgr)
            
            trigger_gemini_now = False
            yolo_objects_for_event = []
            triggering_yolo_objects_summary = []
            if yolo_detections: # ... (YOLO对象处理和Gemini触发判断不变) ...
                for det in yolo_detections:
                    try:
                        simplified_obj = SimplifiedDetectedObject(class_name=det['class_name'], confidence=det['confidence'])
                        yolo_objects_for_event.append(simplified_obj)
                        if det['class_name'] in TRIGGER_CLASSES_FOR_GEMINI and det['confidence'] >= MIN_CONFIDENCE_FOR_GEMINI_TRIGGER:
                            trigger_gemini_now = True; triggering_yolo_objects_summary.append(f"{det['class_name']} (conf: {det['confidence']:.2f})")
                    except KeyError as ke: print(f"警告: YOLO字典键错误: {ke} - {det}")
                    except Exception as e_obj: print(f"警告: 处理YOLO对象时出错: {e_obj} - {det}")

            current_timestamp_for_logic = time.time()
            event_instance_created_this_frame = None 

            if trigger_gemini_now and \
               (current_timestamp_for_logic - last_gemini_trigger_time > GEMINI_COOLDOWN_SECONDS) and \
               gemini_analyzer.is_initialized:
                # ... (Gemini分析、快照保存、Event创建的逻辑不变) ...
                print(f"\n帧 {frame_counter}: **触发Gemini分析** (检测到: {', '.join(triggering_yolo_objects_summary)})")
                last_gemini_trigger_time = current_timestamp_for_logic
                snapshot_filename = f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.jpg"; snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename); snapshot_saved_successfully = False
                try:
                    if cv2.imwrite(snapshot_path, frame_bgr): snapshot_saved_successfully = True
                    else: print(f"  错误：保存快照到 {snapshot_path} 失败。"); snapshot_path = None
                except Exception as e_snap: print(f"  错误：保存快照时异常: {e_snap}"); snapshot_path = None
                prompt_text_for_gemini = (f"分析以下图片中的场景。图片中可能检测到了以下值得注意的对象：{', '.join(triggering_yolo_objects_summary)}。\n评估当前场景的潜在风险，并给出风险评级。\n同时，简要描述图片中的主要事件内容。\n请严格以JSON格式返回结果，该JSON对象应包含两个键：\n1. 'risk_level': 字符串，值为'低'、'中'或'高'中的一个 (如果无法判断或无风险，则为'未知'或'低')。\n2. 'description': 字符串，为事件的详细文本描述。\n不要包含任何JSON之外的解释性文字或markdown标记.")
                gemini_risk, gemini_desc = gemini_analyzer.analyze_image(frame_bgr, prompt_text_for_gemini)
                gemini_call_ts = datetime.now(timezone.utc).isoformat()
                event_instance_created_this_frame = Event(
                    camera_id=cam_id_str_for_buffer_and_event, detected_yolo_objects=yolo_objects_for_event,
                    triggering_image_snapshot_path=snapshot_path if snapshot_saved_successfully else None,
                    gemini_analysis_prompt=prompt_text_for_gemini,
                    gemini_risk_level=gemini_risk if gemini_risk else "分析失败",
                    gemini_description=gemini_desc if gemini_desc else "未能获取描述",
                    gemini_api_call_timestamp_utc=gemini_call_ts
                )
                print(f"  Gemini分析结果 - 风险: {event_instance_created_this_frame.gemini_risk_level}, 描述 (部分): {str(event_instance_created_this_frame.gemini_description)[:80]}...")
                
                if event_logger.log_directory: # 日志记录逻辑不变
                    if event_logger.log_event(event_instance_created_this_frame):
                        print(f"  事件 {event_instance_created_this_frame.event_id} 已成功记录到本地日志。")
                    else:
                        print(f"  错误：事件 {event_instance_created_this_frame.event_id} 记录到本地日志失败。")
                
                # --- 移除服务器推送逻辑 ---
                # if server_publisher.is_configured and \
                #    event_instance_created_this_frame.gemini_risk_level in PUSH_RISK_LEVELS_TO_SERVER:
                #     print(f"  事件风险为 '{event_instance_created_this_frame.gemini_risk_level}'，准备推送到服务器...")
                #     # ... 推送调用 ...
                # elif server_publisher.is_configured:
                #     print(f"  事件风险为 '{event_instance_created_this_frame.gemini_risk_level}'，不推送到服务器。")
                # --------------------------
            
            current_process_time_end = time.perf_counter() # ... (耗时打印逻辑不变) ...
            total_frame_processing_time_ms = (current_process_time_end - current_process_time_start) * 1000
            if event_instance_created_this_frame: print(f"帧 {frame_counter}: Gemini分析完成 (总耗时含Gemini: {total_frame_processing_time_ms:.2f} ms)")
            elif yolo_detections: print(f"帧 {frame_counter}: YOLO检测到 {len(yolo_detections)} 个对象 (总耗时: {total_frame_processing_time_ms:.2f} ms)")
            else: print(f"帧 {frame_counter}: 未检测到对象 (总耗时: {total_frame_processing_time_ms:.2f} ms)")

            display_frame = frame_bgr.copy() # ... (可视化与保存结果帧逻辑不变) ...
            if yolo_detections:
                class_names_to_use = cfg.COCO_CLASSES if hasattr(cfg, 'COCO_CLASSES') else class_names_from_util
                display_frame = draw_detections_on_image(display_frame, yolo_detections, class_names_to_use)
            if event_instance_created_this_frame and event_instance_created_this_frame.gemini_risk_level != "未分析":
                 cv2.putText(display_frame, f"Gemini Risk: {event_instance_created_this_frame.gemini_risk_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if SAVE_RESULT_FRAMES and (frame_counter % SAVE_FRAME_INTERVAL == 0):
                timestamp_str = time.strftime("%Y%m%d_%H%M%S"); save_path = os.path.join(FRAMES_OUTPUT_DIR, f"output_frame_{timestamp_str}_{frame_counter}.jpg")
                try: cv2.imwrite(save_path, display_frame)
                except Exception as e_save: print(f"保存结果帧失败: {e_save}")
            
    except KeyboardInterrupt: # ...
        print("\n用户通过Ctrl+C请求中断程序...")
    except Exception as e: # ...
        print(f"\n主循环中发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
    finally: # ... (资源释放不变) ...
        print("\n开始清理和释放资源...")
        if 'video_buffer' in locals() and hasattr(video_buffer, 'current_video_writer') and video_buffer.current_video_writer: video_buffer.close(); print("视频缓冲区已关闭。")
        if 'yolo_detector' in locals() and hasattr(yolo_detector, 'model_loaded') and yolo_detector.model_loaded: yolo_detector.close()
        if 'camera' in locals() and hasattr(camera, 'is_opened') and camera.is_opened: camera.close()
        print("--- SC171监控智能识别项目 - 主程序结束 ---")

if __name__ == '__main__':
    if not os.getenv("FIBO_LIB"): # ...
        pass
    if hasattr(cfg, 'ensure_directories_exist'): # ...
        cfg.ensure_directories_exist()
        if hasattr(cfg, 'VIDEO_CACHE_DIR'): os.makedirs(cfg.VIDEO_CACHE_DIR, exist_ok=True); print(f"视频缓存根目录 '{cfg.VIDEO_CACHE_DIR}' 已确保存在。")
    main()