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
    # 导入 GeminiAnalyzerCloud 和新的 Pydantic 模型 (如果 Event 模型需要它)
    from src.ai_processing.gemini_analyzer_cloud import GeminiAnalyzerCloud, GeminiAnalysisResult 
    # 导入 Event 和 SimplifiedDetectedObject
    from src.data_management.event_models_shared import Event, SimplifiedDetectedObject 
    from src.data_management.event_logger_local import EventLoggerLocal
    from src.ai_processing.yolo_v8_image_utils import draw_detections_on_image, COCO_CLASSES as class_names_from_util
    from src.ai_processing.api_infer_wrapper import Runtime 

except ImportError as e:
    print(f"关键导入错误: {e}") 
    exit()

# --- 全局控制参数 (与之前类似) ---
SAVE_RESULT_FRAMES = True
SAVE_FRAME_INTERVAL = 90 
FRAMES_OUTPUT_DIR = os.path.join(cfg.DATA_DIR, "main_output_frames_v2.4_gemini_struct") # 更新版本号
SNAPSHOTS_DIR = os.path.join(cfg.DATA_DIR, "event_snapshots")

TRIGGER_CLASSES_FOR_GEMINI = getattr(cfg, 'TRIGGER_CLASSES_FOR_GEMINI', ["person", "fire"])
MIN_CONFIDENCE_FOR_GEMINI_TRIGGER = getattr(cfg, 'MIN_CONFIDENCE_FOR_GEMINI_TRIGGER', 0.65)
GEMINI_COOLDOWN_SECONDS = getattr(cfg, 'GEMINI_COOLDOWN_SECONDS', 30)

def main():
    print("--- SC171监控智能识别项目 - 主程序启动 (V2.4 - 集成结构化Gemini输出) ---")
    
    if SAVE_RESULT_FRAMES: os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    # ... (其他初始化打印和目录创建与之前类似) ...

    print("\n[初始化摄像头模块...]")
    camera = CameraHandlerSC171(camera_source=cfg.SC171_CAMERA_SOURCE, desired_fps=cfg.DESIRED_FPS)
    # ... (摄像头初始化和参数获取与之前类似) ...
    if not camera.open(): return
    actual_cam_w, actual_cam_h = camera.get_resolution()
    actual_cam_fps = camera.get_fps()
    if not actual_cam_w or not actual_cam_h or actual_cam_fps <= 0: camera.close(); return
    print(f"摄像头实际参数: {actual_cam_w}x{actual_cam_h} @ {actual_cam_fps:.2f} FPS")


    print("\n[初始化YOLO检测器模块 (GPU模式)...]")
    yolo_detector = YoloDetectorSC171(runtime_target=Runtime.GPU)
    # ... (YOLO初始化与之前类似) ...
    if not yolo_detector.model_loaded: camera.close(); return


    print("\n[初始化Gemini分析器模块...]")
    gemini_analyzer = GeminiAnalyzerCloud() # 初始化不变
    if not gemini_analyzer.is_initialized: print("警告：Gemini分析器初始化失败。云分析功能将不可用。")


    print("\n[初始化本地事件记录器模块...]")
    event_logger = EventLoggerLocal(log_directory=cfg.LOG_DIR) # 初始化不变
    if event_logger.log_directory is None: print("警告：本地事件记录器未能正确初始化日志目录。")


    print("\n[初始化视频缓冲模块...]") # ... (VideoBuffer初始化不变) ...
    cam_id_str_for_buffer_and_event = f"cam_{cfg.SC171_CAMERA_SOURCE}" if isinstance(cfg.SC171_CAMERA_SOURCE, int) else str(cfg.SC171_CAMERA_SOURCE).replace('/','_').replace('\\','_')
    video_buffer = VideoBufferSC171(
        camera_id=cam_id_str_for_buffer_and_event, cache_duration_seconds=cfg.VIDEO_CACHE_DURATION_MINUTES * 60,
        output_directory_root=cfg.VIDEO_CACHE_DIR, fps=actual_cam_fps, 
        frame_width=actual_cam_w, frame_height=actual_cam_h
    )
    if not video_buffer.current_video_writer: camera.close(); yolo_detector.close(); return


    print("\n--- 所有核心模块初始化完成，开始主循环 ---")
    frame_counter = 0
    last_gemini_trigger_time = 0.0

    try:
        while True:
            ret, frame_bgr = camera.read_frame()
            if not ret or frame_bgr is None: break
            # ... (帧尺寸检查与resize不变) ...
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
            yolo_objects_for_event_creation = [] # 用于创建Event对象的SimplifiedDetectedObject列表
            triggering_yolo_objects_summary_for_prompt = [] # 用于构建Gemini Prompt的字符串列表
            
            if yolo_detections:
                for det in yolo_detections:
                    try:
                        simplified_obj = SimplifiedDetectedObject(
                            class_name=det['class_name'], confidence=det['confidence']
                        )
                        yolo_objects_for_event_creation.append(simplified_obj)
                        if det['class_name'] in TRIGGER_CLASSES_FOR_GEMINI and \
                           det['confidence'] >= MIN_CONFIDENCE_FOR_GEMINI_TRIGGER:
                            trigger_gemini_now = True
                            triggering_yolo_objects_summary_for_prompt.append(f"{det['class_name']} (conf: {det['confidence']:.2f})")
                    except KeyError as ke: print(f"警告: YOLO字典键错误: {ke} - {det}")
                    except Exception as e_obj: print(f"警告: 处理YOLO对象时出错: {e_obj} - {det}")

            current_timestamp_for_logic = time.time()
            event_instance_this_frame: Optional[Event] = None # 初始化为None

            if trigger_gemini_now and \
               (current_timestamp_for_logic - last_gemini_trigger_time > GEMINI_COOLDOWN_SECONDS) and \
               gemini_analyzer.is_initialized:
                
                print(f"\n帧 {frame_counter}: **触发Gemini分析** (检测到: {', '.join(triggering_yolo_objects_summary_for_prompt)})")
                last_gemini_trigger_time = current_timestamp_for_logic

                snapshot_path = None # 初始化快照路径
                snapshot_saved_successfully = False
                # ... (快照保存逻辑不变) ...
                snapshot_filename = f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                temp_snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
                try:
                    if cv2.imwrite(temp_snapshot_path, frame_bgr): 
                        snapshot_path = temp_snapshot_path # 只有成功才赋值
                        snapshot_saved_successfully = True
                        # print(f"  图像快照已保存到: {snapshot_path}")
                    else: print(f"  错误：保存快照到 {temp_snapshot_path} 失败。")
                except Exception as e_snap: print(f"  错误：保存快照时异常: {e_snap}")
                
                # 调用Gemini分析，现在期望返回 GeminiAnalysisResult 对象或 None
                gemini_analysis_data: Optional[GeminiAnalysisResult] = gemini_analyzer.analyze_image(
                    frame_bgr, 
                    yolo_objects_summary=", ".join(triggering_yolo_objects_summary_for_prompt) # 传递YOLO摘要
                )
                # gemini_api_call_ts_for_event = datetime.now(timezone.utc).isoformat() # 这个可以移到Event内部或不单独记录

                event_instance_this_frame = Event(
                    camera_id=cam_id_str_for_buffer_and_event,
                    detected_yolo_objects=yolo_objects_for_event_creation, # 包含当前帧所有简化的YOLO检测
                    triggering_image_snapshot_path=snapshot_path, # 使用已确定的快照路径
                    
                    # --- 将GeminiAnalysisResult对象直接赋给Event的新字段 ---
                    gemini_analysis=gemini_analysis_data 
                    # 之前分散的 gemini_risk_level, gemini_description 等字段已移入 Event.gemini_analysis
                )
                
                if gemini_analysis_data:
                    print(f"  Gemini分析完成 - 总体风险: {gemini_analysis_data.risk_level}")
                    print(f"    火灾: {gemini_analysis_data.specific_events_detected.fire.detected} "
                          f"(详情: {gemini_analysis_data.specific_events_detected.fire.details or 'N/A'})")
                    # 可以打印其他特定事件
                else:
                    print(f"  Gemini分析失败或未返回有效结构。Event对象中的gemini_analysis将为None。")
                
                print(f"  新事件已创建: ID {event_instance_this_frame.event_id}")
                
                if event_logger.log_directory:
                    if event_logger.log_event(event_instance_this_frame): # log_event现在会调用新的to_custom_dict
                        print(f"  事件 {event_instance_this_frame.event_id} 已成功记录到本地日志。")
                    # ... (日志记录失败的打印)

            # ... (单帧耗时打印逻辑与之前类似，可根据event_instance_this_frame调整) ...
            current_process_time_end = time.perf_counter(); total_frame_processing_time_ms = (current_process_time_end - current_process_time_start) * 1000
            if event_instance_this_frame and event_instance_this_frame.gemini_analysis: print(f"帧 {frame_counter}: Gemini分析完成 (总耗时含Gemini: {total_frame_processing_time_ms:.2f} ms)")
            elif yolo_detections: print(f"帧 {frame_counter}: YOLO检测到 {len(yolo_detections)} 个对象 (总耗时: {total_frame_processing_time_ms:.2f} ms)")
            else: print(f"帧 {frame_counter}: 未检测到对象 (总耗时: {total_frame_processing_time_ms:.2f} ms)")


            # --- 可视化与保存结果帧 ---
            display_frame = frame_bgr.copy()
            if yolo_detections: # ... (绘制YOLO检测框不变) ...
                class_names_to_use = cfg.COCO_CLASSES if hasattr(cfg, 'COCO_CLASSES') else class_names_from_util
                display_frame = draw_detections_on_image(display_frame, yolo_detections, class_names_to_use)
            
            # 从Event对象中获取Gemini风险信息来绘制
            if event_instance_this_frame and event_instance_this_frame.gemini_analysis:
                 risk_to_display = event_instance_this_frame.gemini_analysis.risk_level
                 cv2.putText(display_frame, f"Gemini Risk: {risk_to_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 # 可以再绘制特定事件的检测状态
                 fire_detected = event_instance_this_frame.gemini_analysis.specific_events_detected.fire.detected
                 fire_status_text = f"Fire: {'Yes' if fire_detected else 'No'}"
                 cv2.putText(display_frame, fire_status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if fire_detected else (0,100,0), 2)


            if SAVE_RESULT_FRAMES and (frame_counter % SAVE_FRAME_INTERVAL == 0): # ... (保存结果帧不变) ...
                timestamp_str = time.strftime("%Y%m%d_%H%M%S"); save_path = os.path.join(FRAMES_OUTPUT_DIR, f"output_frame_{timestamp_str}_{frame_counter}.jpg")
                try: cv2.imwrite(save_path, display_frame)
                except Exception as e_save: print(f"保存结果帧失败: {e_save}")
            
    except KeyboardInterrupt: # ...
        print("\n用户通过Ctrl+C请求中断程序...")
    except Exception as e: # ...
        print(f"\n主循环中发生未捕获的异常: {e}")
        import traceback; traceback.print_exc()
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