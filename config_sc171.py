# ~/sc171_monitor_project/config_sc171.py
import os
from dotenv import load_dotenv # 确保在文件顶部导入

# --- 1. 项目基础路径设置 ---
# 假设此config_sc171.py文件位于项目的根目录下
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 2. 加载 .env 文件 (应在所有 os.getenv 调用之前) ---
# 这会加载项目根目录下的 .env 文件中的环境变量
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"信息 [config_sc171]: 已从 '{dotenv_path}' 加载环境变量。")
else:
    print(f"警告 [config_sc171]: .env 文件未找到于 '{dotenv_path}'。依赖环境变量的服务可能无法工作。")

# --- 3. 核心目录路径定义 ---
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VIDEO_CACHE_DIR = os.path.join(DATA_DIR, "video_cache") # VideoBufferSC171的根输出目录
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images") # 测试图片存放目录
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")       # AI模型文件存放目录
SNAPSHOTS_OUTPUT_DIR = os.path.join(DATA_DIR, "event_snapshots") # 事件快照保存目录

# --- 4. 摄像头配置 (SC171特定) ---
SC171_CAMERA_SOURCE = 2 # 摄像头ID
WIDTH = 640
HEIGHT = 480
DESIRED_FPS = 20.0      # 你期望的摄像头捕获帧率

# --- 5. YOLOv8 SNPE 模型配置 ---
# !! 请务必使用 `snpe-dlc-info your_model.dlc` 命令检查并替换以下与你的模型匹配的值 !!
YOLO_DLC_NAME = "yolov8n.dlc" # 模型文件名，方便更换
YOLO_DLC_PATH = os.path.join(MODELS_DIR, YOLO_DLC_NAME) 

YOLO_MODEL_INPUT_NAME = "images"
MODEL_INPUT_SHAPE = (640, 640) 

YOLO_MODEL_OUTPUT_NAMES = ["output0"] 
MODEL_OUTPUT_SHAPE = (1,84,8400) 

# --- 6. YOLOv8 后处理参数 ---
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

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


# --- 7. Gemini API (通过OpenAI库访问) 配置 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # 从.env文件加载
if not GEMINI_API_KEY:
    print("警告 [config_sc171]: GEMINI_API_KEY 未在环境变量中设置。Gemini云分析将不可用。")

# Google为Gemini提供的OpenAI兼容端点 (根据你测试成功的配置)
GEMINI_OPENAI_COMPATIBLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" 
# 用于Gemini API调用的模型ID
GEMINI_MODEL_ID_FOR_VISION = "gemini-2.5-flash" 
GEMINI_API_TIMEOUT_SECONDS = 60  # API调用超时时间

# --- 8. 视频缓存配置 ---
VIDEO_CACHE_DURATION_MINUTES = 0.25 # 视频片段缓存时长 (分钟)
VIDEO_FILE_NAME = "./data/video_cache/video_cache.mp4"

# --- 9. 事件处理与Gemini触发配置 ---
TRIGGER_CLASSES_FOR_GEMINI = ["person"] # 哪些YOLO类别触发Gemini
MIN_CONFIDENCE_FOR_GEMINI_TRIGGER = 0.65      # 触发Gemini的最小YOLO置信度
GEMINI_COOLDOWN_SECONDS = 30                  # Gemini调用冷却时间 (秒)

# --- 10. FrameStack与FrameQueue配置 ---
VIDEO_STACK_MAX_SIZE = 50 # 实时帧缓冲器最大深度
VIDEO_QUEUE_MAX_SIZE = 100 # 视频缓冲队列最大深度


# --- 11. 辅助函数：确保目录存在 ---
# 这个函数在其他模块导入此config时不会自动执行，需要在主程序显式调用
def ensure_project_directories_exist():
    """创建项目中需要用到的核心数据和日志目录。"""
    directories_to_create = [
        LOG_DIR,
        DATA_DIR,
        VIDEO_CACHE_DIR,
        TEST_IMAGES_DIR,
        MODELS_DIR,
        SNAPSHOTS_OUTPUT_DIR
    ]
    for directory in directories_to_create:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"警告 [config_sc171]: 创建目录 '{directory}' 失败: {e}")
    print("信息 [config_sc171]: 项目核心目录已检查/创建。")


# --- 12. (可选) 模块自测试/配置打印 ---
# 当直接运行 `python3 config_sc171.py` 时执行
if __name__ == '__main__':
    print("--- 配置信息预览 (config_sc171.py) ---")
    print(f"  项目根目录 (PROJECT_ROOT): {PROJECT_ROOT}")
    print(f"  日志目录 (LOG_DIR): {LOG_DIR}")
    print(f"  数据根目录 (DATA_DIR): {DATA_DIR}")
    print(f"  视频缓存目录 (VIDEO_CACHE_DIR): {VIDEO_CACHE_DIR}")
    print(f"  模型目录 (MODELS_DIR): {MODELS_DIR}")
    print(f"  事件快照目录 (SNAPSHOTS_OUTPUT_DIR): {SNAPSHOTS_OUTPUT_DIR}")
    
    print(f"\n  YOLO DLC路径 (YOLO_DLC_PATH): {YOLO_DLC_PATH}")
    print(f"  YOLO 输入节点名 (YOLO_MODEL_INPUT_NAME): {YOLO_MODEL_INPUT_NAME}")
    print(f"  YOLO 输入布局 (YOLO_MODEL_INPUT_LAYOUT): {YOLO_MODEL_INPUT_LAYOUT}")
    print(f"  YOLO 输出节点名 (YOLO_MODEL_OUTPUT_NAMES): {YOLO_MODEL_OUTPUT_NAMES}")
    
    print(f"\n  摄像头源 (SC171_CAMERA_SOURCE): {SC171_CAMERA_SOURCE}")
    
    print(f"\n  Gemini API Key (GEMINI_API_KEY): {'已设置' if GEMINI_API_KEY else '未设置或加载失败'}")
    print(f"  Gemini Base URL: {GEMINI_OPENAI_COMPATIBLE_BASE_URL}")
    print(f"  Gemini 模型ID: {GEMINI_MODEL_ID_FOR_VISION}")

    print(f"\n  视频缓存时长 (分钟): {VIDEO_CACHE_DURATION_MINUTES}")
    print(f"  触发Gemini的类别: {TRIGGER_CLASSES_FOR_GEMINI}")

    print("\n  调用 ensure_project_directories_exist()...")
    ensure_project_directories_exist()
    print("--- 配置预览结束 ---")