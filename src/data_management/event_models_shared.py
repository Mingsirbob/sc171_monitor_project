# ~/sc171_monitor_project/src/data_management/event_models_shared.py
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid
from typing import List, Dict, Any, Optional

@dataclass
class SimplifiedDetectedObject:
    """
    描述YOLO检测到的单个对象的简化信息 (符合你期望的日志格式)。
    """
    class_name: str
    confidence: float
    # bbox_xywh: Optional[List[int]] = None # 如果确定日志中不需要bbox，则注释或删除此行

@dataclass
class Event:
    """
    数据类，用于表示一个检测到的并可能经过分析的事件。
    """
    camera_id: str 
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())) # 仍然建议使用UUID作为事件ID
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # 使用新的简化对象列表
    detected_yolo_objects: List[SimplifiedDetectedObject] = field(default_factory=list) 

    triggering_image_snapshot_path: Optional[str] = None 
    gemini_analysis_prompt: Optional[str] = None
    gemini_risk_level: str = "未分析" 
    gemini_description: Optional[str] = None
    gemini_api_call_timestamp_utc: Optional[str] = None

    def to_custom_dict(self) -> Dict[str, Any]:
        """
        将 Event 对象转换为你期望的自定义字典格式，用于日志记录。
        """
        # 先处理嵌套的 detected_yolo_objects
        yolo_objects_for_log = []
        for obj in self.detected_yolo_objects:
            yolo_objects_for_log.append({
                "class_name": obj.class_name,
                "confidence": round(obj.confidence, 4) # 可以控制小数位数
                # 如果需要bbox: "bbox_xywh": obj.bbox_xywh 
            })

        return {
            "camera_id": self.camera_id,
            "event_id": self.event_id, # 通常 event_id 是一个UUID，而不是“事件名称”
                                      # 如果“事件名称”是其他含义，需要额外字段
            "timestamp_utc": self.timestamp_utc,
            "detected_yolo_objects": yolo_objects_for_log,
            "gemini_risk_level": self.gemini_risk_level,
            "gemini_description": self.gemini_description
            # 注意：triggering_image_snapshot_path, gemini_analysis_prompt, 
            # gemini_api_call_timestamp_utc 没有在你期望的格式中，所以这里不包含。
            # 如果需要，可以添加。
        }

    # from_dict 方法如果需要，也需要相应调整以匹配自定义格式，但对于日志记录主要是to_dict
    # 为了简单，这里暂时不修改from_dict，因为日志主要是写入。
    # 如果你需要从这种自定义格式的日志中读回Event对象，则from_dict也需要适配。

    # 保留标准的asdict，如果其他地方需要完整数据
    def to_full_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- 模块级测试代码 ---
if __name__ == '__main__':
    print("--- event_models_shared.py 模块测试 (自定义日志格式) ---")

    print("\n[测试 SimplifiedDetectedObject]")
    s_obj1 = SimplifiedDetectedObject(class_name="person", confidence=0.8876543)
    print(f"  创建的SimplifiedDetectedObject: {s_obj1}")
    print(f"  转为字典 (asdict): {asdict(s_obj1)}") # dataclass的asdict会包含所有字段

    print("\n[测试 Event 创建与 to_custom_dict()]")
    yolo_detections_for_event_simplified = [
        SimplifiedDetectedObject(class_name="person", confidence=0.92123),
        SimplifiedDetectedObject(class_name="truck", confidence=0.75987)
    ]

    event1 = Event(
        camera_id="SC171_CAM_01_LIVING_ROOM",
        detected_yolo_objects=yolo_detections_for_event_simplified,
        gemini_risk_level="高",
        gemini_description="这是一个测试事件描述。"
    )
    
    custom_dict_output = event1.to_custom_dict()
    print("\n  Event1 to_custom_dict() 输出:")
    import json
    print(json.dumps(custom_dict_output, indent=2, ensure_ascii=False))
    
    # 验证字段是否符合期望
    assert "camera_id" in custom_dict_output
    assert "event_id" in custom_dict_output
    assert "timestamp_utc" in custom_dict_output
    assert "detected_yolo_objects" in custom_dict_output
    assert "gemini_risk_level" in custom_dict_output
    assert "gemini_description" in custom_dict_output
    # 确保不期望的字段不在里面 (除非你修改了to_custom_dict来包含它们)
    assert "triggering_image_snapshot_path" not in custom_dict_output 
    assert "gemini_analysis_prompt" not in custom_dict_output

    if custom_dict_output["detected_yolo_objects"]:
        first_yolo_obj_log = custom_dict_output["detected_yolo_objects"][0]
        assert "class_name" in first_yolo_obj_log
        assert "confidence" in first_yolo_obj_log
        assert "bbox_xywh" not in first_yolo_obj_log # 因为我们在SimplifiedDetectedObject中没加
        assert "class_id" not in first_yolo_obj_log # 因为我们只选了class_name
    
    print("\n  自定义字典格式测试通过。")
    print("\n--- event_models_shared.py 模块测试完成 ---")