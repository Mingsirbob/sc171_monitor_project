# ~/sc171_monitor_project/src/data_management/event_logger_local.py
import json
import os
from datetime import datetime, timezone
from typing import Optional, List 

# 导入Event模型和配置
try:
    from .event_models_shared import Event, SimplifiedDetectedObject 
    from config_sc171 import LOG_DIR
except ImportError:
    print("警告 [EventLoggerLocal]: 无法从config_sc171或同级event_models_shared导入。")
    print("将使用后备定义（主要用于本模块独立测试）。")
    LOG_DIR = "temp_logs_event_logger_test" # 临时目录
    from dataclasses import dataclass, field, asdict # asdict用于后备的to_custom_dict
    # --- 后备定义 START ---
    @dataclass
    class SimplifiedDetectedObject: # <--- 后备定义中包含此类
        class_name: str
        confidence: float
        # bbox_xywh: Optional[List[int]] = None # 根据需要

    @dataclass
    class Event:
        camera_id: str
        event_id: str = field(default_factory=lambda: "test_uuid_logger_fallback")
        timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        detected_yolo_objects: List[SimplifiedDetectedObject] = field(default_factory=list)
        gemini_risk_level: str = "未分析"
        gemini_description: Optional[str] = None
        # 为测试添加一个 to_custom_dict 的简单实现
        def to_custom_dict(self): 
            # 注意：这里的asdict(o) 对于后备的SimplifiedDetectedObject是有效的
            return {
                "camera_id": self.camera_id, "event_id": self.event_id, 
                "timestamp_utc": self.timestamp_utc,
                "detected_yolo_objects": [asdict(o) for o in self.detected_yolo_objects],
                "gemini_risk_level": self.gemini_risk_level,
                "gemini_description": self.gemini_description
            }
    # --- 后备定义 END ---


class EventLoggerLocal:
    # ... (构造函数和方法保持不变，它们依赖于正确导入或后备定义的Event) ...
    def __init__(self, log_directory: str = None, log_file_prefix: str = "events_sc171"):
        self.log_directory = log_directory if log_directory is not None else LOG_DIR
        self.log_file_prefix = log_file_prefix
        try:
            os.makedirs(self.log_directory, exist_ok=True)
        except OSError as e:
            print(f"错误 [EventLoggerLocal]: 创建日志目录 '{self.log_directory}' 失败: {e}")
            self.log_directory = None

    def _get_daily_log_file_path(self) -> Optional[str]:
        if not self.log_directory: return None
        today_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_directory, f"{self.log_file_prefix}_{today_str}.jsonl")

    def log_event(self, event: Event) -> bool: # event类型应匹配导入或后备的Event
        if not isinstance(event, Event):
            print("错误 [EventLoggerLocal.log_event]: 输入的不是一个有效的Event对象。")
            return False
        log_file_path = self._get_daily_log_file_path()
        if not log_file_path:
            print("错误 [EventLoggerLocal.log_event]: 日志目录无效，无法记录事件。")
            return False
        try:
            event_dict = event.to_custom_dict() 
            json_line = json.dumps(event_dict, ensure_ascii=False)
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            return True
        except AttributeError as ae:
            print(f"错误 [EventLoggerLocal.log_event]: Event对象缺少 to_custom_dict() 方法或该方法出错: {ae}")
            return False
        except IOError as e:
            print(f"错误 [EventLoggerLocal.log_event]: 写入日志文件 '{log_file_path}' 时发生IO错误: {e}")
            return False
        except Exception as e:
            print(f"错误 [EventLoggerLocal.log_event]: 记录事件时发生未知错误: {e}")
            return False

# --- 模块级测试代码 ---
if __name__ == '__main__':
    print("--- EventLoggerLocal 模块测试 (使用自定义日志格式) ---")
    
    # 当直接运行此脚本时，顶层的try-except会捕获ImportError，
    # 此时 LOG_DIR, Event, SimplifiedDetectedObject 会使用后备定义。
    # 我们需要确保这些后备定义是可用的。
    
    # 确保测试时日志目录存在 (使用顶层定义的LOG_DIR，它可能是真实的或后备的)
    if LOG_DIR: # 检查LOG_DIR是否有效 (不是None)
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"EventLoggerLocal: 日志目录 '{LOG_DIR}' 已确保存在 (用于测试)。")
    else:
        print("错误：日志目录在测试开始前无效，无法继续测试。")
        exit()

    logger = EventLoggerLocal() # 它会使用顶层定义的LOG_DIR
    
    print(f"日志将尝试写入文件，例如: {logger._get_daily_log_file_path()}")

    # 创建使用 SimplifiedDetectedObject 的事件 (现在这些类应该从后备定义中可用)
    test_yolo_objects_simplified = [
        SimplifiedDetectedObject(class_name="person", confidence=0.81234),
        SimplifiedDetectedObject(class_name="car", confidence=0.76543)
    ]
    event1 = Event( # 这个Event也是后备定义中的Event
        camera_id="test_cam_001_custom_log_fallback", 
        detected_yolo_objects=test_yolo_objects_simplified,
        gemini_risk_level="中",
        gemini_description="自定义日志格式测试事件1 (后备)"
    )
    # event1.event_id = "evt_custom_fallback_001" 

    print("\n记录第一个自定义格式事件...")
    if logger.log_event(event1):
        print(f"  事件 {event1.event_id} 记录成功。")
    else:
        print(f"  事件 {event1.event_id} 记录失败。")

    log_file_to_check = logger._get_daily_log_file_path()
    if log_file_to_check and os.path.exists(log_file_to_check):
        print(f"\n请检查日志文件内容: {log_file_to_check}")
        try:
            with open(log_file_to_check, "r", encoding="utf-8") as f:
                print("文件最新一行内容预览:")
                last_line = None
                for line in f: 
                    last_line = line
                if last_line:
                    print(f"  {last_line.strip()}")
        except Exception as e:
            print(f"读取日志文件预览失败: {e}")
    else:
        print(f"日志文件 {log_file_to_check} 未找到或无法访问。")
        
    print("\n--- EventLoggerLocal 模块测试完成 ---")