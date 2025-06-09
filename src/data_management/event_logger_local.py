# ~/sc171_monitor_project/src/data_management/event_logger_local.py
import json
import os
from datetime import datetime, timezone
from typing import Optional, List # 确保List也被导入，因为后备定义中会用到

# 导入Event模型和配置
try:
    from .event_models_shared import Event, DetectedObject # <--- 同时导入 DetectedObject
    from config_sc171 import LOG_DIR
except ImportError:
    print("警告 [EventLoggerLocal]: 无法从config_sc171或同级event_models_shared导入。")
    print("将使用后备定义（主要用于本模块独立测试）。")
    LOG_DIR = "temp_logs_event_logger_test"
    from dataclasses import dataclass, field, asdict
    # from typing import List # 已在顶部导入
    @dataclass
    class DetectedObject: class_id: int; class_name: str; confidence: float; bbox_xywh: List[int]
    @dataclass
    class Event:
        camera_id: str
        event_id: str = field(default_factory=lambda: "test_uuid")
        timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        detected_yolo_objects: List[DetectedObject] = field(default_factory=list)
        gemini_risk_level: str = "未分析"
        gemini_description: Optional[str] = None
        def to_dict(self): return asdict(self)


class EventLoggerLocal:
    # ... (类定义保持不变) ...
    def __init__(self, log_directory: str = None, log_file_prefix: str = "events_sc171"):
        self.log_directory = log_directory if log_directory is not None else LOG_DIR
        self.log_file_prefix = log_file_prefix
        try:
            os.makedirs(self.log_directory, exist_ok=True)
            print(f"EventLoggerLocal: 日志目录 '{self.log_directory}' 已确保存在。")
        except OSError as e:
            print(f"错误 [EventLoggerLocal]: 创建日志目录 '{self.log_directory}' 失败: {e}")
            self.log_directory = None

    def _get_daily_log_file_path(self) -> Optional[str]:
        if not self.log_directory:
            return None
        today_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_directory, f"{self.log_file_prefix}_{today_str}.jsonl")

    def log_event(self, event: Event) -> bool:
        if not isinstance(event, Event):
            print("错误 [EventLoggerLocal.log_event]: 输入的不是一个有效的Event对象。")
            return False
        log_file_path = self._get_daily_log_file_path()
        if not log_file_path:
            print("错误 [EventLoggerLocal.log_event]: 日志目录无效，无法记录事件。")
            return False
        try:
            event_dict = event.to_dict()
            json_line = json.dumps(event_dict, ensure_ascii=False)
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
            return True
        except AttributeError: # ... (异常处理保持不变) ...
            print(f"错误 [EventLoggerLocal.log_event]: 输入的Event对象缺少 to_dict() 方法。")
            return False
        except IOError as e:
            print(f"错误 [EventLoggerLocal.log_event]: 写入日志文件 '{log_file_path}' 时发生IO错误: {e}")
            return False
        except Exception as e:
            print(f"错误 [EventLoggerLocal.log_event]: 记录事件时发生未知错误: {e}")
            return False

# --- 模块级测试代码 ---
if __name__ == '__main__':
    print("--- EventLoggerLocal 模块测试 ---")
    
    # 确保测试时LOG_DIR存在
    # 如果顶部的from config_sc171 import LOG_DIR失败，LOG_DIR会是后备的 "temp_logs_event_logger_test"
    if not os.path.exists(LOG_DIR): # 检查并创建，以防万一
        print(f"测试块：创建临时日志目录 {LOG_DIR}")
        os.makedirs(LOG_DIR, exist_ok=True)

    logger = EventLoggerLocal() 
    
    print(f"日志将尝试写入文件，例如: {logger._get_daily_log_file_path()}")

    # --- 修改点：确保DetectedObject已定义 ---
    # 尝试从已导入的模块获取DetectedObject，如果导入失败，则测试块无法使用它除非本地定义
    # 但因为我们已经在顶部的try-except中也导入了DetectedObject (如果成功)
    # 或者在except块中定义了它，所以这里应该总是能找到 DetectedObject
    # 但为了更明确，我们可以再次检查
    if 'DetectedObject' not in globals():
        print("错误：DetectedObject 未在当前作用域定义，测试无法进行。")
        # 这通常不应该发生，因为顶部的导入或后备定义应该处理了
    else:
        test_yolo_objects = [
            DetectedObject(class_id=0, class_name="person", confidence=0.8, bbox_xywh=[10,10,20,50])
        ]
        # 同样，Event类也应该在全局作用域内可用
        event1 = Event(
            camera_id="test_cam_001", 
            detected_yolo_objects=test_yolo_objects,
            gemini_risk_level="中",
            gemini_description="测试事件描述1"
        )
        event2 = Event(
            camera_id="test_cam_002",
            gemini_risk_level="高",
            gemini_description="这是一个高风险事件的描述，包含中文。"
        )

        print("\n记录第一个事件...")
        if logger.log_event(event1): # ... (后续测试逻辑保持不变) ...
            print(f"  事件 {event1.event_id} 记录成功。")
        else:
            print(f"  事件 {event1.event_id} 记录失败。")

        print("\n记录第二个事件...")
        if logger.log_event(event2):
            print(f"  事件 {event2.event_id} 记录成功。")
        else:
            print(f"  事件 {event2.event_id} 记录失败。")

        log_file_to_check = logger._get_daily_log_file_path()
        if log_file_to_check and os.path.exists(log_file_to_check):
            print(f"\n请检查日志文件内容: {log_file_to_check}")
            try:
                with open(log_file_to_check, "r", encoding="utf-8") as f:
                    print("文件内容预览 (前5行):")
                    for i, line in enumerate(f):
                        if i >= 5: break
                        print(f"  {line.strip()}")
            except Exception as e: print(f"读取日志文件预览失败: {e}")
        else: print(f"日志文件 {log_file_to_check} 未找到或无法访问。")
        
    print("\n--- EventLoggerLocal 模块测试完成 ---")