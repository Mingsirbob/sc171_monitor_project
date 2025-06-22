# ~/sc171_monitor_project/src/data_management/event_models_shared.py
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel , Field

# --- Pydantic模型定义 (用于Gemini分析结果的结构) ---
# 这些模型将由 GeminiAnalyzerCloud 返回，并作为 Event.gemini_analysis 的类型
class SpecificEventDetail(BaseModel):
    detected: bool = Field(description="是否检测到此特定事件")
    confidence: Optional[float] = Field(None, description="对此判断的置信度 (0.0-1.0)，如果模型能提供")
    details: Optional[str] = Field(None, description="关于此特定事件的额外描述或证据")

class SpecificEventsDetected(BaseModel):
    fire: SpecificEventDetail = Field(default_factory=lambda: SpecificEventDetail(detected=False))
    fall_down: SpecificEventDetail = Field(default_factory=lambda: SpecificEventDetail(detected=False))
    fighting: SpecificEventDetail = Field(default_factory=lambda: SpecificEventDetail(detected=False))

class GeminiAnalysisResult(BaseModel):
    risk_level: str = Field(description="总体风险等级 (例如: 低, 中, 高, 未知)")
    description: str = Field(description="事件的综合文本描述")
    specific_events_detected: SpecificEventsDetected = Field(default_factory=SpecificEventsDetected)

# --- 项目核心Dataclass ---
@dataclass
class SimplifiedDetectedObject:
    class_name: str
    confidence: float

@dataclass
class Event:
    camera_id: str 
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    detected_yolo_objects: List[SimplifiedDetectedObject] = field(default_factory=list) 
    triggering_image_snapshot_path: Optional[str] = None 
    gemini_analysis: Optional[GeminiAnalysisResult] = None # 类型是上面定义的Pydantic模型

    def to_custom_dict(self) -> Dict[str, Any]:
        yolo_objects_for_log = [{"class_name": obj.class_name, "confidence": round(obj.confidence, 4)} 
                                for obj in self.detected_yolo_objects]
        gemini_data_for_log = None
        if self.gemini_analysis:
            try: gemini_data_for_log = self.gemini_analysis.model_dump(mode='json', exclude_none=True) # Pydantic V2
            except AttributeError:
                try: gemini_data_for_log = self.gemini_analysis.dict(exclude_none=True) # Pydantic V1
                except AttributeError:
                    gemini_data_for_log = {"error": "gemini_analysis 无法序列化"}
                    print(f"警告: Event.to_custom_dict() 无法序列化 gemini_analysis: {type(self.gemini_analysis)}")
        return {
            "camera_id": self.camera_id, "event_id": self.event_id,
            "timestamp_utc": self.timestamp_utc,
            "detected_yolo_objects": yolo_objects_for_log,
            "gemini_analysis": gemini_data_for_log,
            "triggering_image_snapshot_path": self.triggering_image_snapshot_path
        }

    # from_dict 暂时简化或不实现，主要用于写入
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'Event':
        yolo_objects_data = data_dict.get("detected_yolo_objects", [])
        yolo_objects = [SimplifiedDetectedObject(**obj_data) for obj_data in yolo_objects_data]
        init_data = {k: v for k, v in data_dict.items() if k not in ["detected_yolo_objects", "gemini_analysis"]}
        init_data["detected_yolo_objects"] = yolo_objects
        # 简单处理gemini_analysis的反序列化，如果需要Pydantic对象则需要更复杂逻辑
        gemini_raw = data_dict.get("gemini_analysis")
        if gemini_raw and isinstance(gemini_raw, dict):
            try:
                init_data["gemini_analysis"] = GeminiAnalysisResult(**gemini_raw)
            except Exception: # ValidationError等
                init_data["gemini_analysis"] = gemini_raw # 回退到字典
        else:
            init_data["gemini_analysis"] = gemini_raw
        return cls(**init_data)


# --- 模块级测试代码 ---
if __name__ == '__main__':
    print("--- event_models_shared.py 模块测试 (更新Event结构) ---")
    
    # SimplifiedDetectedObject 和 Event 现在应该直接从本模块的全局作用域获取
    s_obj1 = SimplifiedDetectedObject(class_name="person", confidence=0.88) # 直接使用
    print(f"\n  SimplifiedDetectedObject: {s_obj1}")

    yolo_sim_list = [s_obj1]
    
    # 创建模拟的Gemini分析结果对象，使用本模块定义的Pydantic模型
    mock_fire_detail = SpecificEventDetail(detected=True, confidence=0.9, details="火焰明显")
    mock_fall_detail = SpecificEventDetail(detected=False)
    mock_fight_detail = SpecificEventDetail(detected=False)
    
    mock_specific_events = SpecificEventsDetected(
        fire=mock_fire_detail, 
        fall_down=mock_fall_detail, 
        fighting=mock_fight_detail
    )
    mock_gemini_analysis = GeminiAnalysisResult(
        risk_level="高",
        description="检测到火灾风险。",
        specific_events_detected=mock_specific_events
    )

    event1 = Event( # 直接使用
        camera_id="CAM_TEST_001",
        detected_yolo_objects=yolo_sim_list,
        gemini_analysis=mock_gemini_analysis, 
        triggering_image_snapshot_path="/path/to/snap.jpg"
    )
    print(f"\n  创建的Event对象 (event1):\n    ID: {event1.event_id}\n    Camera: {event1.camera_id}")
    if event1.gemini_analysis:
        print(f"    Gemini Risk: {event1.gemini_analysis.risk_level}")
        print(f"    Gemini Fire Detected: {event1.gemini_analysis.specific_events_detected.fire.detected}")

    print("\n  Event1 to_custom_dict() 输出:")
    import json # 确保导入json
    custom_dict = event1.to_custom_dict()
    print(json.dumps(custom_dict, indent=2, ensure_ascii=False))
    
    assert "gemini_analysis" in custom_dict
    if custom_dict.get("gemini_analysis"):
        assert "specific_events_detected" in custom_dict["gemini_analysis"]
        assert "fire" in custom_dict["gemini_analysis"]["specific_events_detected"]
        assert custom_dict["gemini_analysis"]["specific_events_detected"]["fire"]["detected"] is True
    else:
        print("警告：custom_dict中未找到gemini_analysis数据。")
    
    print("\n--- event_models_shared.py 模块测试完成 ---")