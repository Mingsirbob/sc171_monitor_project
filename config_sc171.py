# ~/sc171_monitor_project/config_sc171.py
import os
from dotenv import load_dotenv


# --- 项目路径设置 ---
# 假设此config_sc171.py文件位于项目的根目录下
# 如果不是，你可能需要调整PROJECT_ROOT的获取方式
# 例如，如果它在 src/config/ 目录下，则 PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 日志与数据缓存路径 ---
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VIDEO_CACHE_DIR = os.path.join(DATA_DIR, "video_cache")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models") # DLC模型存放目录

# --- YOLOv8 SNPE 模型配置 ---
# !! 请务必使用 `snpe-dlc-info your_model.dlc` 命令检查并替换以下值 !!
YOLO_DLC_PATH = os.path.join(MODELS_DIR, "yolov8n.dlc")  # SC171上yolov8n.dlc的绝对路径

YOLO_MODEL_INPUT_WIDTH = 640
YOLO_MODEL_INPUT_HEIGHT = 640
YOLO_MODEL_INPUT_CHANNELS = 3 # 通常是3 (RGB)

# 模型期望的输入节点名称 (通过 snpe-dlc-info 获取)
# 常见的YOLOv8 ONNX导出后的输入名是 "images"
YOLO_MODEL_INPUT_NAME = "images" # <--- 替换为你的DLC实际输入名

# 模型期望的输入数据布局: "NCHW" 或 "NHWC"
# YOLOv8 ONNX导出通常是NCHW (1, Channels, Height, Width)
# SNPE可能对特定后端有偏好，需要确认DLC转换时的设置或DLC本身的信息
YOLO_MODEL_INPUT_LAYOUT = "NCHW" # <--- 根据你的DLC确认或修改

# 模型主要的输出节点名称列表 (通过 snpe-dlc-info 获取)
# YOLOv8可能只有一个主要输出，或者有多个检测头对应多个输出
# FIBO的YOLOv5示例用的是 "StatefulPartitionedCall:0"
# 你需要看你的yolov8n.dlc的实际输出名
YOLO_MODEL_OUTPUT_NAMES = ["output0"] # <--- 替换为你的DLC实际输出名(们)

# 主要输出张量被reshape后的期望形状
# 对于YOLOv8 (COCO 80类): (Batch, NumClasses+Coords, NumProposals) -> (1, 80+4, 8400) = (1, 84, 8400)
YOLO_MODEL_OUTPUT_EXPECTED_SHAPE = (1, 84, 8400) # <--- 根据你的DLC确认或修改

# --- YOLOv8 后处理参数 ---
YOLO_CONF_THRESHOLD = 0.30  # 检测结果的置信度阈值 (可调整)
YOLO_IOU_THRESHOLD = 0.45   # NMS (非极大值抑制) 的IOU阈值 (可调整)

# COCO类别名称列表 (确保与你的yolov8n模型训练时使用的类别一致)
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

# --- SNPE 运行时配置 (可以由主程序或YoloDetector动态选择) ---
# 导入FIBO SDK定义的枚举类，如果它们在你的api_infer_wrapper.py中
# from src.ai_processing.api_infer_wrapper import Runtime, PerfProfile, LogLevel # 假设路径
# DEFAULT_SNPE_RUNTIME = Runtime.DSP # 最终目标
# DEFAULT_SNPE_PERF_PROFILE = PerfProfile.BURST
# DEFAULT_SNPE_LOG_LEVEL = LogLevel.INFO
# 注意：如果 api_infer_wrapper.py 尚未创建，或者为了解耦，
# 也可以在YoloDetector中直接使用字符串 "DSP", "CPU", "GPU" 等，
# 然后在SnpeContext初始化时转换为FIBO SDK的枚举值。
# 为简单起见，我们暂时不在config中定义这些，让YoloDetector的构造函数参数来决定。

# --- 摄像头配置 (SC171特定) ---
# 摄像头ID或设备路径 (例如，'/dev/video0')
# 这需要根据SC171的具体情况来确定
SC171_CAMERA_SOURCE = 0 # 或者 "/dev/video0", "/dev/video1" 等

# 期望的捕获帧率 (如果摄像头支持设置)
# 如果CameraHandlerSC171内部能获取实际帧率，这个可以作为参考或后备
DESIRED_FPS = 20.0

# --- Gemini API 配置 (从.env文件加载) ---
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # 确保.env文件和python-dotenv已设置

# --- 服务器通信配置 ---
# SERVER_EVENT_PUBLISH_ENDPOINT = "https://httpbin.org/post" # 测试用
# SERVER_VIDEO_UPLOAD_ENDPOINT = "https://httpbin.org/post"  # 测试用
# SERVER_AUTH_TOKEN = os.getenv("YOUR_SERVER_AUTH_TOKEN") # 如果服务器需要认证

# --- 视频缓存配置 ---
VIDEO_CACHE_DURATION_MINUTES = 15 # 视频片段缓存时长 (分钟)

# --- 事件处理与推送配置 ---
# TRIGGER_CLASSES_FOR_GEMINI = ["fire", "person"] # 哪些YOLO类别触发Gemini
# MIN_CONFIDENCE_FOR_GEMINI_TRIGGER = 0.6 # 触发Gemini的最小YOLO置信度
# PUSH_RISK_LEVELS_TO_SERVER = ["高", "中"] # 哪些风险级别推送到服务器
# GEMINI_COOLDOWN_SECONDS = 30 # Gemini调用冷却时间

# --- 确保目录存在 ---
# 在程序启动时，可以有一个初始化函数来创建这些目录
def ensure_directories_exist():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)
    # os.makedirs(os.path.join(VIDEO_CACHE_DIR, "snapshots"), exist_ok=True) # 如果使用快照
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("必要的项目目录已检查/创建。")

# 你可以在你的 main_sc171.py 的开头调用 ensure_directories_exist()
# 或者，每个模块在需要时自行创建其所需的目录 (os.makedirs(..., exist_ok=True))


# ~/sc171_monitor_project/config_sc171.py
# ...其他配置...

# --- 摄像头配置 (SC171特定) ---
# 摄像头ID或设备路径 (例如，0, 1, '/dev/video0', '/dev/video1')
# 你需要根据SC171的实际情况来确定
SC171_CAMERA_SOURCE = 2 # 默认尝试第一个摄像头
# 或者 SC171_CAMERA_SOURCE = "/dev/video0"

# 期望的捕获帧率 (如果摄像头支持设置)
# 如果CameraHandlerSC171内部能获取实际帧率，这个可以作为参考或后备
DESIRED_FPS = 20.0 # 你期望的FPS

dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print(f"警告 [config_sc171]: .env 文件未找到于 {dotenv_path}。依赖环境变量的服务可能无法工作。")

# --- Gemini API (通过OpenAI库访问) 配置 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("警告 [config_sc171]: GEMINI_API_KEY 未在环境变量中设置。Gemini分析将不可用。")

# Google为Gemini提供的OpenAI兼容端点
GEMINI_OPENAI_COMPATIBLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" 
# 注意：上面的URL后面可能还需要加上 "/openai" 或类似的路径，
# 例如 "https://generativelanguage.googleapis.com/v1beta/openai"
# 或者直接在OpenAI client初始化时用 "https://generativelanguage.googleapis.com/v1beta/models"
# 然后 model 参数用 "gemini-pro-vision"
# 你提供的示例是 client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
# 和 model="gemini-1.5-flash-latest"
# 我们将遵循你的示例。确保你的API Key支持这个模型和端点。

# 用于Gemini API调用的模型ID (确保你的API Key有权访问此模型)
# "gemini-pro-vision" 是一个常见的视觉模型
# 你提供的示例用了 "gemini-1.5-flash-latest"
GEMINI_MODEL_ID_FOR_VISION = "gemini-2.5-flash-preview-05-20" 

# (可选) Gemini API调用的默认超时时间 (秒)
GEMINI_API_TIMEOUT_SECONDS = 60 

# 示例：打印一些配置，用于快速验证
if __name__ == '__main__':
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"DLC模型路径: {YOLO_DLC_PATH}")
    print(f"模型输入节点名: {YOLO_MODEL_INPUT_NAME}")
    print(f"模型输出节点名: {YOLO_MODEL_OUTPUT_NAMES}")
    print(f"日志目录: {LOG_DIR}")
    ensure_directories_exist()