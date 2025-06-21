# ~/sc171_monitor_project/main_sc171.py
import cv2
import time
import os
from datetime import datetime, timezone

# --- 1. 导入 ---
try:
    import config_sc171 as cfg # 用于应用级配置
    # YoloDetectorSC171 内部会处理其模型相关的配置
    from src.ai_processing.yolo_detector_sc171 import YoloDetectorSC171
    from src.video_io.camera_handler_sc171 import CameraHandlerSC171
    from src.video_io.video_buffer_sc171 import VideoBufferSC171
    from src.ai_processing.gemini_analyzer_cloud import GeminiAnalyzerCloud, GeminiAnalysisResult
    from src.data_management.event_models_shared import Event, SimplifiedDetectedObject
    from src.data_management.event_logger_local import EventLoggerLocal
    # draw_detections_on_image 现在由 YoloDetectorSC171.draw_results 内部调用
    from src.ai_processing.api_infer_wrapper import Runtime # YoloDetectorSC171内部可能用到
except ImportError as e:
    print(f"关键导入错误: {e}")
    exit()

# --- 2. 全局应用级配置读取 ---
SAVE_DEBUG_FRAMES = getattr(cfg, 'SAVE_RESULT_FRAMES', False)
SAVE_FRAME_INTERVAL = getattr(cfg, 'SAVE_FRAME_INTERVAL', 300)
DEBUG_FRAMES_DIR = os.path.join(cfg.DATA_DIR, "main_output_frames_v2.5_new_yolo") # 更新版本
EVENT_SNAPSHOTS_DIR = os.path.join(cfg.DATA_DIR, "event_snapshots")

TRIGGER_CLASSES = getattr(cfg, 'TRIGGER_CLASSES_FOR_GEMINI', ["person"]) # 保持不变
# MIN_CONFIDENCE_FOR_GEMINI_TRIGGER 现在由 YoloDetectorSC171 内部的 CONF_THRESHOLD 控制
GEMINI_COOLDOWN = getattr(cfg, 'GEMINI_COOLDOWN_SECONDS', 30)

def initialize_modules(config_module):
    print("\n[模块初始化阶段]")
    camera = CameraHandlerSC171(camera_source=config_module.SC171_CAMERA_SOURCE, desired_fps=config_module.DESIRED_FPS)
    if not camera.open(): print("错误：摄像头打开失败。"); return None, None, None, None, None, None
    cam_w, cam_h = camera.get_resolution(); cam_fps = camera.get_fps()
    if not cam_w or not cam_h or cam_fps <= 0: print("错误：未能从摄像头获取有效参数。"); camera.close(); return None, None, None, None, None, None
    print(f"  摄像头: {cam_w}x{cam_h} @ {cam_fps:.2f} FPS")

    # YoloDetectorSC171 现在从其模块内部获取配置
    yolo = YoloDetectorSC171()
    if not hasattr(yolo, 'snpe_ort') or yolo.snpe_ort is None : # 简单检查是否初始化成功
        print("错误：YOLO检测器初始化失败。")
        camera.close(); return None, None, None, None, None, None
    print("  YOLO检测器: 初始化成功")

    gemini = GeminiAnalyzerCloud() # 不变
    if not gemini.is_initialized: print("警告：Gemini分析器初始化失败。")
    else: print("  Gemini分析器: 初始化成功")

    logger = EventLoggerLocal(log_directory=config_module.LOG_DIR) # 不变
    if logger.log_directory is None: print("警告：本地事件记录器目录无效。")
    else: print(f"  本地事件记录器: 日志目录 '{logger.log_directory}'")
    
    cam_id_str = f"cam_{config_module.SC171_CAMERA_SOURCE}".replace('/','_').replace('\\','_')
    buffer = VideoBufferSC171( # 不变
        camera_id=cam_id_str, cache_duration_seconds=config_module.VIDEO_CACHE_DURATION_MINUTES * 60,
        output_directory_root=config_module.VIDEO_CACHE_DIR, fps=cam_fps,
        frame_width=cam_w, frame_height=cam_h
    )
    if not buffer.current_video_writer:
        print("错误：视频缓冲器未能初始化写入器。")
        camera.close(); yolo.close(); return None, None, None, None, None, None
    print(f"  视频缓冲器 (cam_id: {cam_id_str}): 初始化成功")
    
    return camera, yolo, gemini, logger, buffer, cam_id_str

def process_frame_logic(frame_bgr, yolo_detector, gemini_analyzer, event_logger, video_buffer, cam_id, frame_num, last_gemini_time):
    current_time = time.time()
    event_this_frame = None
    debug_frame_save_path = None # 用于返回保存的调试帧路径

    completed_video_path = video_buffer.add_frame(frame_bgr.copy()) # 不变
    if completed_video_path:
        print(f"信息: 视频片段 '{os.path.basename(completed_video_path)}' 已保存。")

    # 1. YOLO检测 (现在 detect 不返回值)
    yolo_detector.detect(frame_bgr) 
    
    # 2. 使用 save_results 获取关心的对象信息，并判断是否触发Gemini
    # TRIGGER_CLASSES 是全局的，YoloDetectorSC171内部的 CONF_THRESHOLD 会做第一轮过滤
    # save_results 的 find_object 参数期望是一个列表的列表，例如 [["person", "fire"]]
    triggering_objects_text_list = yolo_detector.save_results([TRIGGER_CLASSES])
    
    trigger_gemini = bool(triggering_objects_text_list) # 如果列表不为空，则触发
    yolo_summary_for_prompt = "; ".join(triggering_objects_text_list) if triggering_objects_text_list else "未检测到明确触发对象"

    # 将 yolo_detector.final_detections (原始的、按类别组织的) 转换为 SimplifiedDetectedObject 列表
    yolo_objects_for_event_creation = []
    if hasattr(yolo_detector, 'final_detections') and yolo_detector.final_detections:
        for class_id, boxes_in_class in enumerate(yolo_detector.final_detections):
            if len(boxes_in_class) > 0:
                class_name = yolo_detector.class_names[class_id] # 使用检测器内部的类别名
                for box in boxes_in_class: # box 是 [x,y,w,h,conf]
                    # 只有通过了YOLO内部CONF_THRESHOLD的才会在这里
                    s_obj = SimplifiedDetectedObject(class_name=class_name, confidence=box[4])
                    yolo_objects_for_event_creation.append(s_obj)
    
    if trigger_gemini and (current_time - last_gemini_time > GEMINI_COOLDOWN) and gemini_analyzer.is_initialized:
        print(f"信息: 帧 {frame_num} - 触发Gemini分析 (基于: {yolo_summary_for_prompt})")
        last_gemini_time = current_time
        
        snapshot_file = None # ... (快照保存逻辑不变) ...
        try:
            snap_fname = f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.jpg"; snap_fpath = os.path.join(EVENT_SNAPSHOTS_DIR, snap_fname)
            if cv2.imwrite(snap_fpath, frame_bgr): snapshot_file = snap_fpath
        except Exception as e_snap: print(f"错误: 保存快照失败: {e_snap}")

        prompt = (f"场景分析。YOLO初步判断: [{yolo_summary_for_prompt}]. " # 使用新的summary
                  "请严格按JSON格式返回风险评估('risk_level'), 描述('description'), "
                  "及特定事件('specific_events_detected': {'fire':{'detected':bool,...}, ...}).")
        
        gemini_data = gemini_analyzer.analyze_image(frame_bgr, yolo_summary_for_prompt)

        event_this_frame = Event(
            camera_id=cam_id,
            detected_yolo_objects=yolo_objects_for_event_creation, # 使用转换后的列表
            triggering_image_snapshot_path=snapshot_file,
            gemini_analysis=gemini_data
        )
        
        if gemini_data: print(f"  Gemini风险: {gemini_data.risk_level}")
        else: print("  Gemini分析失败或无有效结果。")

        if event_logger.log_directory:
            if event_logger.log_event(event_this_frame):
                print(f"  事件 {event_this_frame.event_id} 已记录。")
    
    # 调试帧保存 (现在调用YoloDetectorSC171的draw_results)
    if SAVE_DEBUG_FRAMES and (frame_num % SAVE_FRAME_INTERVAL == 0):
        if hasattr(yolo_detector, 'final_detections') and yolo_detector.final_detections: # 确保有检测结果才绘制
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            debug_frame_save_path = os.path.join(DEBUG_FRAMES_DIR, f"debug_{ts_str}_{frame_num}.jpg")
            try:
                # yolo_detector.draw_results 现在负责绘制和保存
                # 它内部会使用 self.image (原始帧) 和 self.final_detections
                # 我们需要确保 self.image 是在 detect() 中被正确设置的原始帧
                yolo_detector.draw_results(save_path=debug_frame_save_path) 
                # 如果要在上面再绘制Gemini信息，需要先获取绘制了YOLO结果的图，再绘制，再保存
                # 或者修改YoloDetectorSC171.draw_results使其能接收额外的文本来绘制
                print(f"调试帧已保存到: {debug_frame_save_path}")
            except Exception as e: print(f"错误: 保存调试帧失败: {e}")

    return event_this_frame, last_gemini_time # 返回是否有事件创建，以及更新的冷却时间

def main_loop(camera, yolo, gemini, logger, buffer, cam_id):
    frame_num = 0
    last_gemini_call_time = 0.0
    
    while True:
        ret, frame = camera.read_frame()
        if not ret or frame is None: break
        frame_num += 1
        
        # 注意：process_frame_logic 现在返回 event_this_frame 和 last_gemini_call_time
        _, last_gemini_call_time = process_frame_logic( # 我们不直接用这里的event_this_frame做主要判断
            frame, yolo, gemini, logger, buffer, cam_id, frame_num, last_gemini_call_time
        )
        
        # print(f"Processed frame {frame_num}")
        # 通过某种方式退出 (例如，特定数量的帧后用于测试，或Ctrl+C)
        # if frame_num > 10: break # 短暂测试

def cleanup_modules(camera, yolo, buffer): # 不变
    print("\n[资源清理阶段]")
    if buffer and hasattr(buffer, 'current_video_writer') and buffer.current_video_writer: buffer.close(); print("  视频缓冲器: 已关闭")
    if yolo and hasattr(yolo, 'snpe_ort'): yolo.close(); print("  YOLO检测器: 已关闭") # 检查snpe_ort是否存在
    if camera and hasattr(camera, 'is_opened') and camera.is_opened: camera.close(); print("  摄像头: 已关闭")

def run_application(): # 不变
    print("--- SC171监控应用启动 (简化版 V2.5 - 新YOLO Detector) ---")
    if hasattr(cfg, 'ensure_project_directories_exist'): cfg.ensure_project_directories_exist()
    if hasattr(cfg, 'VIDEO_CACHE_DIR'): os.makedirs(cfg.VIDEO_CACHE_DIR, exist_ok=True)
    if SAVE_DEBUG_FRAMES: os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
    os.makedirs(EVENT_SNAPSHOTS_DIR, exist_ok=True)

    modules = initialize_modules(cfg)
    if modules[0] is None: print("核心模块初始化失败，程序无法启动。"); return
    camera, yolo, gemini, logger, video_buffer, cam_id_str = modules
    try:
        main_loop(camera, yolo, gemini, logger, video_buffer, cam_id_str)
    except KeyboardInterrupt: print("\n用户请求中断程序...")
    except Exception as e: print(f"\n主程序发生未捕获的严重异常: {e}"); import traceback; traceback.print_exc()
    finally: cleanup_modules(camera, yolo, video_buffer)
    print("--- SC171监控应用结束 ---")

if __name__ == '__main__':
    if not os.getenv("FIBO_LIB"):print("警告：FIBO_LIB环境变量未设置。")
    run_application()