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
    from src.video_io.video_buffer_sc171 import VideoBufferSC171 # 导入视频缓存模块
    # from src.video_io.video_uploader_sc171 import VideoUploaderSC171 # 下一步集成
    from src.ai_processing.yolo_detector_sc171 import YoloDetectorSC171
    from src.ai_processing.gemini_analyzer_cloud import GeminiAnalyzerCloud
    from src.data_management.event_models_shared import Event, DetectedObject
    from src.data_management.event_logger_local import EventLoggerLocal
    from src.ai_processing.yolo_v8_image_utils import draw_detections_on_image
    # 从api_infer_wrapper导入Runtime枚举类，以便指定GPU/DSP
    from src.ai_processing.api_infer_wrapper import Runtime, PerfProfile, LogLevel

except ImportError as e:
    print(f"关键导入错误: {e}")
    print("请确保：")
    print("1. 你在项目的根目录下运行此脚本 (例如 'python3 main_sc171.py')。")
    print("2. 或者，项目的根目录已添加到PYTHONPATH环境变量中。")
    print("3. 所有必要的模块文件都存在于正确的路径下。")
    print("4. FIBO环境脚本已正确执行 (对于api_infer_wrapper中的底层导入)。")
    exit()

# --- 全局控制参数 ---
SAVE_RESULT_FRAMES = True  # 是否保存带有检测结果的帧图像 (用于调试)
SAVE_FRAME_INTERVAL = getattr(cfg, 'SAVE_FRAME_INTERVAL', 60) # 每隔多少帧保存一次，默认60帧
FRAMES_OUTPUT_DIR = os.path.join(cfg.DATA_DIR, "main_output_frames_v2.2")
SNAPSHOTS_DIR = os.path.join(cfg.DATA_DIR, "event_snapshots")

# 从配置读取或使用默认值
TRIGGER_CLASSES_FOR_GEMINI = getattr(cfg, 'TRIGGER_CLASSES_FOR_GEMINI', ["person"]) # 默认只关注 "person"
MIN_CONFIDENCE_FOR_GEMINI_TRIGGER = getattr(cfg, 'MIN_CONFIDENCE_FOR_GEMINI_TRIGGER', 0.70) # 稍微提高阈值
GEMINI_COOLDOWN_SECONDS = getattr(cfg, 'GEMINI_COOLDOWN_SECONDS', 60) # 默认60秒冷却
VIDEO_CACHE_DURATION_MIN = getattr(cfg, 'VIDEO_CACHE_DURATION_MINUTES', 1) # 测试时用1分钟，实际用15


def main():
    print(f"--- SC171监控智能识别项目 - 主程序启动 (V2.2 - 集成VideoBuffer) ---")
    
    if SAVE_RESULT_FRAMES:
        os.makedirs(FRAMES_OUTPUT_DIR, exist_ok=True)
        print(f"带检测框的结果帧将保存到: {FRAMES_OUTPUT_DIR}")
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    print(f"事件快照将保存到: {SNAPSHOTS_DIR}")
    # VideoBufferSC171 会在其构造函数中创建 camera_id 子目录

    # --- 2. 初始化核心模块 ---
    print("\n[初始化摄像头模块...]")
    camera = CameraHandlerSC171(
        camera_source=cfg.SC171_CAMERA_SOURCE,
        desired_fps=cfg.DESIRED_FPS 
    )
    if not camera.open():
        print("错误：摄像头打开失败。程序退出。")
        return
    
    actual_cam_w, actual_cam_h = camera.get_resolution()
    actual_cam_fps = camera.get_fps()
    if not actual_cam_w or not actual_cam_h or actual_cam_fps <= 0:
        print("错误：未能从摄像头获取有效的帧尺寸或FPS。程序退出。")
        camera.close()
        return
    print(f"摄像头实际参数: {actual_cam_w}x{actual_cam_h} @ {actual_cam_fps:.2f} FPS")

    # 构造 camera_id 字符串
    cam_id_str = f"cam_{cfg.SC171_CAMERA_SOURCE}" if isinstance(cfg.SC171_CAMERA_SOURCE, int) else str(cfg.SC171_CAMERA_SOURCE).replace('/','_').replace('\\','_')

    print("\n[初始化视频缓存模块...]")
    video_buffer = VideoBufferSC171(
        camera_id=cam_id_str,
        cache_duration_seconds=VIDEO_CACHE_DURATION_MIN * 60,
        output_directory_root=cfg.VIDEO_CACHE_DIR,
        fps=actual_cam_fps, 
        frame_width=actual_cam_w,
        frame_height=actual_cam_h
        # fourcc_str='XVID' # 如果mp4v有问题，可以尝试XVID (输出.avi)
    )
    if not video_buffer.is_writer_open: # 修改检查条件为 is_writer_open
        print("错误: 视频缓存模块未能初始化VideoWriter。程序退出。")
        camera.close()
        return
    print(f"视频缓存模块初始化完成。视频片段将缓存 {VIDEO_CACHE_DURATION_MIN} 分钟。")

    print("\n[初始化YOLO检测器模块 (GPU模式)...]")
    yolo_detector = YoloDetectorSC171(runtime_target=Runtime.GPU) 
    if not yolo_detector.model_loaded:
        if hasattr(video_buffer, 'close'): video_buffer.close()
        camera.close()
        return

    print("\n[初始化Gemini分析器模块...]")
    gemini_analyzer = GeminiAnalyzerCloud()
    if not gemini_analyzer.is_initialized:
        print("警告：Gemini分析器初始化失败。云分析功能将不可用。")

    print("\n[初始化本地事件记录器模块...]")
    event_logger = EventLoggerLocal(log_directory=cfg.LOG_DIR)
    if event_logger.log_directory is None:
        print("警告：本地事件记录器未能正确初始化日志目录。事件可能无法记录。")


    print("\n--- 所有核心模块初始化完成，开始主循环 ---")
    frame_counter = 0
    last_gemini_trigger_time = 0.0 
    # 初始化为0.0，确保第一次符合条件时能触发，或 time.time() - 很大的数，
    # 或者在循环外第一次检查时不判断冷却。这里0.0配合 current_time - 0.0 > cooldown 即可。

    try:
        while True:
            ret, frame_bgr = camera.read_frame()
            if not ret or frame_bgr is None:
                print("无法从摄像头读取帧，可能视频流结束或发生错误。")
                break
            
            frame_counter += 1
            current_process_time_start = time.perf_counter()

            # --- 视频缓存：将原始帧添加到缓冲区 ---
            if video_buffer and video_buffer.is_writer_open: # 再次检查writer是否有效
                completed_segment_path = video_buffer.add_frame(frame_bgr) # 使用原始 frame_bgr
                if completed_segment_path:
                    print(f"\n========= 视频片段已录制完成并保存 =========")
                    print(f"  路径: {completed_segment_path}")
                    print(f"  (下一步：将调用上传器上传此文件，目前仅打印信息)")
                    print("================================================\n")
            
            # 为YOLO和Gemini创建副本，避免修改正在被缓存的原始帧
            frame_for_ai = frame_bgr.copy()

            # --- YOLO目标检测 ---
            yolo_detections = yolo_detector.detect(frame_for_ai)
            
            # --- Gemini触发与事件创建逻辑 ---
            trigger_gemini_now = False
            yolo_objects_for_event = []
            triggering_yolo_objects_summary = []

            if yolo_detections:
                for det in yolo_detections:
                    try:
                        detected_obj = DetectedObject(
                            class_id=det['class_id'], class_name=det['class_name'],
                            confidence=det['confidence'], bbox_xywh=det['bbox_xywh']
                        )
                        yolo_objects_for_event.append(detected_obj)
                        if det['class_name'] in TRIGGER_CLASSES_FOR_GEMINI and \
                           det['confidence'] >= MIN_CONFIDENCE_FOR_GEMINI_TRIGGER:
                            trigger_gemini_now = True
                            triggering_yolo_objects_summary.append(f"{det['class_name']}({det['confidence']:.2f})")
                    except KeyError as ke: print(f"警告: YOLO检测结果字典缺少键: {ke} - {det}")
                    except Exception as e_obj: print(f"警告: 处理YOLO检测对象时出错: {e_obj} - {det}")
            
            current_timestamp = time.time() # 获取当前精确时间
            event_instance_created_this_frame = None # 用于存储本帧创建的Event对象

            if trigger_gemini_now and \
               (current_timestamp - last_gemini_trigger_time > GEMINI_COOLDOWN_SECONDS) and \
               gemini_analyzer.is_initialized:
                
                print(f"\n帧 {frame_counter}: **触发Gemini分析** (检测到: {', '.join(triggering_yolo_objects_summary)})")
                
                # 1. 保存图像快照
                snapshot_filename = f"snapshot_{cam_id_str}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
                snapshot_saved_successfully = False
                try:
                    if cv2.imwrite(snapshot_path, frame_for_ai): # 使用frame_for_ai进行快照
                        print(f"  图像快照已保存到: {snapshot_path}")
                        snapshot_saved_successfully = True
                    else: print(f"  错误：保存图像快照到 {snapshot_path} 失败。"); snapshot_path = None
                except Exception as e_snap: print(f"  错误：保存图像快照时发生异常: {e_snap}"); snapshot_path = None

                # 2. 构造Prompt
                prompt_text_for_gemini = (
                    f"分析以下图片中的场景。图片中可能检测到了以下值得注意的对象：{', '.join(triggering_yolo_objects_summary)}。\n"
                    "评估当前场景的潜在风险，并给出风险评级。\n"
                    "同时，简要描述图片中的主要事件内容。\n"
                    "请严格以JSON格式返回结果，该JSON对象应包含两个键：\n"
                    "1. 'risk_level': 字符串，值为'低'、'中'或'高'中的一个 (如果无法判断或无风险，则为'未知'或'低')。\n"
                    "2. 'description': 字符串，为事件的详细文本描述。\n"
                    "3. 'event_name':字符串，值为：'火灾监控'，'打架监测'，'摔倒检测'。\n"
                    "不要包含任何JSON之外的解释性文字或markdown标记。"
                )
                
                # 3. 调用Gemini分析
                gemini_risk, gemini_desc = gemini_analyzer.analyze_image(frame_for_ai, prompt_text_for_gemini) # 使用frame_for_ai
                gemini_call_ts = datetime.now(timezone.utc).isoformat()
                last_gemini_trigger_time = current_timestamp # 在API调用后更新冷却时间戳

                # 4. 创建Event对象
                event_instance_created_this_frame = Event(
                    camera_id=cam_id_str,
                    detected_yolo_objects=yolo_objects_for_event, 
                    triggering_image_snapshot_path=snapshot_path if snapshot_saved_successfully else None,
                    gemini_analysis_prompt=prompt_text_for_gemini,
                    gemini_risk_level=gemini_risk if gemini_risk else "分析失败",
                    gemini_description=gemini_desc if gemini_desc else "未能获取描述",
                    gemini_api_call_timestamp_utc=gemini_call_ts
                )
                print(f"  Gemini分析结果 - 风险: {event_instance_created_this_frame.gemini_risk_level}, 描述 (部分): {str(event_instance_created_this_frame.gemini_description)[:80]}...")
                
                # 5. 记录事件到本地日志
                if event_logger and event_logger.log_directory: 
                    if event_logger.log_event(event_instance_created_this_frame):
                        print(f"  事件 {event_instance_created_this_frame.event_id} 已成功记录到本地日志。")
                    else:
                        print(f"  错误：事件 {event_instance_created_this_frame.event_id} 记录到本地日志失败。")
                else:
                    print("  警告：本地事件记录器未正确配置，跳过日志记录。")
            
            current_process_time_end = time.perf_counter()
            total_frame_processing_time_ms = (current_process_time_end - current_process_time_start) * 1000
            
            if event_instance_created_this_frame:
                print(f"帧 {frame_counter}: Gemini分析完成 (总耗时含Gemini: {total_frame_processing_time_ms:.2f} ms)")
            elif yolo_detections:
                print(f"帧 {frame_counter}: YOLO检测到 {len(yolo_objects_for_event)} 个对象 (总耗时: {total_frame_processing_time_ms:.2f} ms)")
            else:
                print(f"帧 {frame_counter}: 未检测到对象 (总耗时: {total_frame_processing_time_ms:.2f} ms)")

            # --- 可视化与保存结果帧 ---
            if SAVE_RESULT_FRAMES: # 只在需要保存时才进行绘制操作
                display_output_frame = frame_bgr.copy() # 用一个新的副本来绘制最终的输出帧
                if yolo_detections:
                    class_names_to_use = cfg.COCO_CLASSES if hasattr(cfg, 'COCO_CLASSES') else class_names_from_util
                    display_output_frame = draw_detections_on_image(display_output_frame, yolo_detections, class_names_to_use)
                
                if event_instance_created_this_frame and hasattr(event_instance_created_this_frame, 'gemini_risk_level'):
                    if event_instance_created_this_frame.gemini_risk_level not in ["未分析", "分析失败"]: # 只显示有效的风险等级
                        cv2.putText(display_output_frame, f"Gemini Risk: {event_instance_created_this_frame.gemini_risk_level}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                if frame_counter % SAVE_FRAME_INTERVAL == 0:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(FRAMES_OUTPUT_DIR, f"output_frame_{cam_id_str}_{timestamp_str}_{frame_counter}.jpg")
                    try: 
                        cv2.imwrite(save_path, display_output_frame)
                        # print(f"结果帧已保存到: {save_path}") # 减少打印
                    except Exception as e_save: print(f"保存结果帧失败: {e_save}")
            
    except KeyboardInterrupt:
        print("\n用户通过Ctrl+C请求中断程序...")
    except Exception as e:
        print(f"\n主循环中发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n开始清理和释放资源...")
        if 'yolo_detector' in locals() and hasattr(yolo_detector, 'model_loaded') and yolo_detector.model_loaded:
            yolo_detector.close()
        if 'video_buffer' in locals() and hasattr(video_buffer, 'is_writer_open') and video_buffer.is_writer_open:
            video_buffer.close() # 确保最后一个视频片段被保存
            print("视频缓存模块已关闭。")
        if 'camera' in locals() and hasattr(camera, 'is_opened') and camera.is_opened:
            camera.close()
        print("--- SC171监控智能识别项目 - 主程序结束 ---")

if __name__ == '__main__':
    if not os.getenv("FIBO_LIB"):
        print("警告：FIBO_LIB环境变量未检测到。请确保在运行此脚本前已正确source FIBO环境脚本。")
        # exit(1) # 对于关键依赖，可以选择退出
    
    # 确保项目根目录下的配置能被正确加载，并创建必要的目录
    # 这是通过 main_sc171.py 顶部的 try-except ImportError 和 cfg.ensure_directories_exist() 来处理的
    # 但如果 cfg 本身导入失败，ensure_directories_exist 就不会被调用。
    # 更好的做法是在这里直接调用，或者在main()的开头。
    if hasattr(cfg, 'PROJECT_ROOT'): # 确保cfg已成功导入.
        # 创建所有在config中定义的目录
        dirs_to_create = [cfg.LOG_DIR, cfg.DATA_DIR, cfg.VIDEO_CACHE_DIR, 
                          cfg.TEST_IMAGES_DIR, cfg.MODELS_DIR, 
                          FRAMES_OUTPUT_DIR, SNAPSHOTS_DIR]
        for d in dirs_to_create:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e_dir:
                print(f"创建目录 {d} 失败: {e_dir}")
        print("必要的项目目录已检查/创建。")

    main()