# ~/sc171_monitor_project/src/data_management/event_models_shared.py
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid
from typing import List, Dict, Any, Optional

@dataclass
class DetectedObject:
    """
    描述YOLO检测到的单个对象的信息。
    """
    class_id: int
    class_name: str
    confidence: float
    bbox_xywh: List[int] # [x_top_left, y_top_left, width, height]

@dataclass
class Event:
    """
    数据类，用于表示一个检测到的并可能经过分析的事件。
    """
    # --- 将没有默认值的字段放在最前面 ---
    camera_id: str # 发生事件的摄像头ID (例如 "cam_01", "sc171_front_door")

    # --- 带有默认值或default_factory的字段放在后面 ---
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    detected_yolo_objects: List[DetectedObject] = field(default_factory=list) 

    triggering_image_snapshot_path: Optional[str] = None 
    
    gemini_analysis_prompt: Optional[str] = None
    gemini_risk_level: str = "未分析" 
    gemini_description: Optional[str] = None
    gemini_api_call_timestamp_utc: Optional[str] = None

    # ... (to_dict 和 from_dict 方法保持不变) ...
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'Event':
        detected_objects_data = data_dict.get("detected_yolo_objects", [])
        yolo_objects = [DetectedObject(**obj_data) for obj_data in detected_objects_data]
        
        init_data = {k: v for k, v in data_dict.items() if k != "detected_yolo_objects"}
        init_data["detected_yolo_objects"] = yolo_objects
        
        return cls(**init_data)

# --- 模块级测试代码 (保持不变，但创建Event实例时要注意参数顺序或使用关键字参数) ---
if __name__ == '__main__':
    print("--- event_models_shared.py 模块测试 ---")

    # 1. 测试 DetectedObject
    print("\n[测试 DetectedObject]")
    obj1 = DetectedObject(class_id=0, class_name="person", confidence=0.88, bbox_xywh=[10, 20, 50, 120])
    print(f"  创建的DetectedObject: {obj1}")
    print(f"  转为字典: {asdict(obj1)}")

    # 2. 测试 Event 对象的创建
    print("\n[测试 Event 创建]")
    yolo_detections_for_event = [
        DetectedObject(class_id=0, class_name="person", confidence=0.92, bbox_xywh=[100, 150, 30, 80]),
        DetectedObject(class_id=7, class_name="truck", confidence=0.75, bbox_xywh=[200, 200, 150, 100])
    ]

    # 创建Event实例时，camera_id是第一个位置参数，或者使用关键字参数
    event1 = Event(
        camera_id="SC171_CAM_01_LIVING_ROOM", # camera_id 现在是必需的第一个参数
        # 其他参数可以使用关键字参数，顺序不重要
        detected_yolo_objects=yolo_detections_for_event,
        triggering_image_snapshot_path="/data/snapshots/evt_abc_frame_123.jpg",
        gemini_analysis_prompt="图片中有什么潜在危险？请评估风险并描述。",
        gemini_risk_level="高",
        gemini_description="检测到人员在禁止区域内，且附近有快速移动的卡车，存在碰撞风险。",
        gemini_api_call_timestamp_utc=datetime.now(timezone.utc).isoformat()
    )
    print(f"  创建的Event对象 (event1):\n    ID: {event1.event_id}\n    Camera: {event1.camera_id}\n    Timestamp: {event1.timestamp_utc}")
    print(f"    YOLO 检测数量: {len(event1.detected_yolo_objects)}")
    if event1.detected_yolo_objects:
        print(f"    第一个YOLO对象: {event1.detected_yolo_objects[0]}")
    print(f"    Gemini风险: {event1.gemini_risk_level}")
    print(f"    Gemini描述 (部分): {event1.gemini_description[:50]}...")

    event2 = Event(
        camera_id="SC171_CAM_02_GARAGE", # 必需
        detected_yolo_objects=[
            DetectedObject(class_id=2, class_name="car", confidence=0.99, bbox_xywh=[30, 40, 200, 100])
        ]
    )
    print(f"\n  创建的Event对象 (event2 - 仅YOLO):\n    ID: {event2.event_id}\n    Camera: {event2.camera_id}\n    Timestamp: {event2.timestamp_utc}")
    print(f"    Gemini风险: {event2.gemini_risk_level}")


    # 3. 测试 to_dict() 和 from_dict()
    print("\n[测试 to_dict() 和 from_dict()]")
    event1_dict = event1.to_dict()
    print(f"  Event1 to_dict() 输出 (部分):")
    print(f"    event_id: {event1_dict['event_id']}")
    print(f"    detected_yolo_objects (第一个): {event1_dict['detected_yolo_objects'][0] if event1_dict['detected_yolo_objects'] else 'N/A'}")
    
    event1_restored = Event.from_dict(event1_dict)
    print(f"\n  从字典恢复的Event对象 (event1_restored):")
    print(f"    ID: {event1_restored.event_id}")
    print(f"    Camera: {event1_restored.camera_id}")
    assert event1.event_id == event1_restored.event_id
    assert event1.detected_yolo_objects[0].class_name == event1_restored.detected_yolo_objects[0].class_name
    assert isinstance(event1_restored.detected_yolo_objects[0], DetectedObject)
    print("  to_dict() 和 from_dict() 基本功能测试通过。")

    print("\n--- event_models_shared.py 模块测试完成 ---")