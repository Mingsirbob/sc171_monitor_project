# ~/sc171_monitor_project/main_sc171.py
import cv2
import time
import os
import threading
from datetime import datetime, timezone

# --- 1. 导入 ---
try:
    import config_sc171 as cfg # 用于应用级配置
    from src.video_io.camera_handler_sc171 import CameraHandler
    from src.video_io.video_buffer_sc171 import FrameStack, FrameQueue
    from src.ai_processing.yolo_detector_sc171 import YoloDetectorSC171
    from src.ai_processing.gemini_analyzer_cloud import GeminiAnalyzerCloud, GeminiAnalysisResult
    from src.video_io.video_saver_sc171 import VideoSaver
    from src.data_management.event_models_shared import Event, SimplifiedDetectedObject
    from src.data_management.event_logger_local import EventLoggerLocal
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

def gemini_analyze(gemini:GeminiAnalyzerCloud, image, yolo_result):
    prompt = (f"场景分析。YOLO初步判断: {yolo_result}. "
                "请严格按JSON格式返回风险评估('risk_level'), 描述('description'), "
                "及特定事件('specific_events_detected': {'fire':{'detected':bool,...}, ...}).")
    gemini_data = gemini.analyze_image(image, prompt)
    return gemini_data

def initialize_modules(config_module):
    print("\n[模块初始化阶段]")
    print("摄像头初始化")
    camera = CameraHandler(camera_source=config_module.SC171_CAMERA_SOURCE, width=config_module.WIDTH, height=config_module.HEIGHT, fps=config_module.DESIRED_FPS)
    if not camera.is_opened(): print("错误：摄像头打开失败。"); return None, None, None, None, None
    camera.print_info()
    actual_width, actual_height, actual_fps = camera.get_actual_params()

    print("Yolo检测器初始化")
    yolo = YoloDetectorSC171()
    print("Yolo检测器初始化成功")

    print("Gemini模型初始化")
    gemini = GeminiAnalyzerCloud() 
    if not gemini.is_initialized: print("警告：Gemini分析器初始化失败。")
    else: print("Gemini模型: 初始化成功")

    logger = EventLoggerLocal(log_directory=config_module.LOG_DIR) 
    if logger.log_directory is None: print("警告：本地事件记录器目录无效。")
    else: print(f"  本地事件记录器: 日志目录 '{logger.log_directory}'")
    print("本地事件记录器: 初始化成功")

    print("VideoSaver初始化")
    video_saver = VideoSaver(config_module.VIDEO_FILE_NAME, actual_width, actual_height, actual_fps)
    print("VideoSaver初始化成功")
    
    return camera, yolo, gemini, video_saver, logger

def camera_producer_worker(camera:CameraHandler, frame_stack:FrameStack, frame_queue:FrameQueue, stop_event:threading.Event):
    while not stop_event.is_set():
        ret, frame = camera.read_frame()
        if not ret:
            print("错误：从摄像头读取帧失败。")
            stop_event.set()
            break
        frame_stack.push(frame)
        frame_queue.put(frame)

def analysis_consumer_worker(yolo:YoloDetectorSC171, gemini:GeminiAnalyzerCloud, frame_stack:FrameStack, logger:EventLoggerLocal, stop_event:threading.Event):
    time.sleep(5)
    yolo_result = None
    while not stop_event.is_set():
        latest_frame = frame_stack.get_latest()
        frame_to_analyze = latest_frame.copy()

        raw_yolo_result = yolo.detect(frame_to_analyze)
        yolo.draw_results('./data/test_results_detector/test.jpg')
        yolo_result = yolo.save_results(["person"])
        if yolo_result:
            print("检测到目标")
            gemini_data = gemini_analyze(gemini, frame_to_analyze, yolo_result)
            
        event_this_frame = Event(
            camera_id = cfg.SC171_CAMERA_SOURCE,
            detected_yolo_objects=yolo_result,
            triggering_image_snapshot_path='./data/test_results_detector/test.jpg',
            gemini_analysis=gemini_data,
        )

        yolo_result = None

    yolo.close()







def video_saver_consumer_worker(video_saver:VideoSaver, frame_queue:FrameQueue, stop_event:threading.Event):
    time.sleep(5)
    frame_count = 0
    while not stop_event.is_set() or not frame_queue.is_empty():
        frame = frame_queue.get()
        video_saver.write(frame)
        frame_count += 1
    video_saver.close()
        


def run_application(): # 不变
    # --- 1. 初始化共享资源 ---
    raw_frame_stack = FrameStack(max_size=cfg.VIDEO_STACK_MAX_SIZE)
    raw_frame_queue = FrameQueue(max_size=cfg.VIDEO_QUEUE_MAX_SIZE)
    stop_event = threading.Event()

    # --- 2. 初始化模块实例 ---
    model = initialize_modules(cfg)

    camera, yolo, gemini, video_saver, logger = model

    camera_producer_t = threading.Thread(target=camera_producer_worker,args=(camera,raw_frame_stack,raw_frame_queue,stop_event))
    analysis_consumer_t = threading.Thread(target=analysis_consumer_worker,args=(yolo,gemini,raw_frame_stack,logger,stop_event))
    video_saver_consumer_t = threading.Thread(target=video_saver_consumer_worker,args=(video_saver,raw_frame_queue,stop_event))

    print("线程启动")
    camera_producer_t.start()
    analysis_consumer_t.start()
    video_saver_consumer_t.start()

    try:
        camera_producer_t.join()
        analysis_consumer_t.join()
        video_saver_consumer_t.join()
    except KeyboardInterrupt:
        print("主线程: 捕获到键盘中断，正在清理...")
        stop_event.set()
        
    print("线程结束")

if __name__ == '__main__':
    if not os.getenv("FIBO_LIB"):print("警告：FIBO_LIB环境变量未设置。")
    run_application()