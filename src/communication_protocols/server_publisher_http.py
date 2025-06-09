# ~/sc171_monitor_project/src/communication_protocols/server_publisher_http.py
import requests
import json
import os # 用于测试时构造路径
from typing import Optional, Dict, Any # 用于类型提示

# 导入配置和Event模型
try:
    # 假设config_sc171.py在项目根目录
    from config_sc171 import SERVER_EVENT_PUBLISH_ENDPOINT, SERVER_AUTH_TOKEN
    # 假设event_models_shared.py在 src/data_management/
    from src.data_management.event_models_shared import Event
except ImportError:
    print("警告 [ServerPublisherHTTP]: 无法从config_sc171或event_models_shared导入。")
    print("将使用后备定义（主要用于本模块独立测试）。")
    SERVER_EVENT_PUBLISH_ENDPOINT = "https://httpbin.org/post" # 测试用
    SERVER_AUTH_TOKEN = None # 测试用
    # 后备Event定义，以便单元测试能运行
    from dataclasses import dataclass, field, asdict
    from typing import List # List也需要从typing导入 (Python 3.8)
    from datetime import datetime, timezone
    @dataclass
    class DetectedObject: class_id: int; class_name: str; confidence: float; bbox_xywh: List[int]
    @dataclass
    class Event:
        camera_id: str
        event_id: str = field(default_factory=lambda: "test_uuid_publisher")
        timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        detected_yolo_objects: List[DetectedObject] = field(default_factory=list)
        gemini_risk_level: str = "未分析"
        gemini_description: Optional[str] = None
        def to_dict(self) -> Dict[str, Any]: return asdict(self)


class ServerPublisherHTTP:
    def __init__(self, 
                 server_url: Optional[str] = None, 
                 auth_token: Optional[str] = None, 
                 timeout: int = 20): # 增加默认超时到20秒，因为网络可能不稳定
        """
        初始化HTTP服务器推送器。
        Args:
            server_url: 服务器接收事件的API端点。如果为None，则从config获取。
            auth_token: (可选) 用于服务器认证的Token。如果为None，则从config获取。
            timeout: 请求超时时间（秒）。
        """
        self.server_url = server_url if server_url is not None else SERVER_EVENT_PUBLISH_ENDPOINT
        self.auth_token = auth_token if auth_token is not None else SERVER_AUTH_TOKEN
        self.timeout = timeout
        
        self.headers = {"Content-Type": "application/json"}
        if self.auth_token:
            # 根据你的服务器认证方案调整，Bearer Token是常见的
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
            print(f"ServerPublisherHTTP: 将使用认证Token (前缀: {self.auth_token[:5]}...) 进行推送。")
        
        if not self.server_url:
            print("错误 [ServerPublisherHTTP]: 服务器URL未配置。推送功能将不可用。")
            self.is_configured = False
        else:
            self.is_configured = True
            print(f"ServerPublisherHTTP: 初始化完成。事件将推送到: {self.server_url}")

    def publish_event_data(self, event_payload_dict: Dict[str, Any], event_id_for_log: str = "N/A") -> bool:
        """
        将事件数据 (字典格式) 推送到服务器。
        Args:
            event_payload_dict: 包含事件信息的字典。
            event_id_for_log: 用于日志记录的事件ID。
        Returns:
            True 如果推送成功 (服务器返回2xx状态码)，否则 False。
        """
        if not self.is_configured:
            print(f"错误 [ServerPublisherHTTP.publish_event_data]: 服务器URL未配置，无法推送事件 {event_id_for_log}。")
            return False

        # print(f"DEBUG [ServerPublisherHTTP]: 准备推送到服务器的数据 (事件 {event_id_for_log}):\n{json.dumps(event_payload_dict, indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                self.server_url,
                headers=self.headers,
                json=event_payload_dict, # requests库会自动将字典转换为JSON字符串
                timeout=self.timeout
            )

            if 200 <= response.status_code < 300:
                print(f"信息 [ServerPublisherHTTP]: 事件 {event_id_for_log} 成功推送到服务器。响应: {response.status_code}")
                # print(f"  服务器响应内容 (部分): {response.text[:200]}...")
                return True
            else:
                print(f"错误 [ServerPublisherHTTP]: 推送事件 {event_id_for_log} 到服务器失败。")
                print(f"  URL: {self.server_url}")
                print(f"  状态码: {response.status_code}")
                print(f"  服务器响应 (部分): {response.text[:500]}...")
                return False
        except requests.exceptions.Timeout:
            print(f"错误 [ServerPublisherHTTP]: 推送事件 {event_id_for_log} 时请求超时 ({self.timeout}秒)。")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"错误 [ServerPublisherHTTP]: 推送事件 {event_id_for_log} 时发生连接错误: {e}")
            return False
        except requests.exceptions.RequestException as e: # 捕获其他requests相关的网络错误
            print(f"错误 [ServerPublisherHTTP]: 推送事件 {event_id_for_log} 时发生网络请求错误: {e}")
            return False
        except Exception as e: # 捕获其他未知错误
            print(f"错误 [ServerPublisherHTTP]: 推送事件 {event_id_for_log} 时发生未知错误: {e}")
            return False

    def publish_event_object(self, event: Event) -> bool:
        """
        将一个Event对象序列化为JSON并推送到服务器。
        这是一个便利方法。
        Args:
            event: 要推送的Event对象。
        Returns:
            True 如果推送成功，否则 False。
        """
        if not isinstance(event, Event):
            print("错误 [ServerPublisherHTTP.publish_event_object]: 输入的不是一个有效的Event对象。")
            return False
        
        try:
            payload = event.to_dict()
            return self.publish_event_data(payload, event_id_for_log=event.event_id)
        except AttributeError:
            print(f"错误 [ServerPublisherHTTP.publish_event_object]: 输入的Event对象缺少 to_dict() 方法。")
            return False
        except Exception as e:
            print(f"错误 [ServerPublisherHTTP.publish_event_object]: 准备Event对象推送时出错: {e}")
            return False


# --- 模块级测试代码 ---
if __name__ == '__main__':
    print("--- ServerPublisherHTTP 模块测试 ---")

    # 确保config_sc171.py可访问，或使用后备值
    # 测试时，建议将SERVER_EVENT_PUBLISH_ENDPOINT设置为 "https://httpbin.org/post"
    test_url = SERVER_EVENT_PUBLISH_ENDPOINT
    if test_url != "https://httpbin.org/post":
        print(f"警告: 为了获得明确的测试回显，建议在config_sc171.py中将")
        print(f"      SERVER_EVENT_PUBLISH_ENDPOINT 设置为 'https://httpbin.org/post'")
        print(f"      当前值为: {test_url}")
    
    # 可以测试有无Auth Token的情况
    # publisher_no_auth = ServerPublisherHTTP(server_url=test_url, auth_token=None)
    publisher_with_auth = ServerPublisherHTTP(server_url=test_url, auth_token="dummy_test_token_123") # 使用虚拟token测试头部

    if not publisher_with_auth.is_configured:
        print("服务器URL未有效配置，测试中止。")
    else:
        # 创建一个示例Event对象
        test_event_obj = Event(
            camera_id="sc171_test_cam_publish",
            detected_yolo_objects=[
                DetectedObject(class_id=0, class_name="person", confidence=0.91, bbox_xywh=[15,25,35,45])
            ],
            gemini_risk_level="高",
            gemini_description="通过ServerPublisherHTTP模块测试推送的高风险事件。"
        )
        # 手动设置一个event_id以便追踪
        test_event_obj.event_id = "evt_test_push_001" 

        print(f"\n尝试使用publish_event_object推送事件 {test_event_obj.event_id}...")
        success1 = publisher_with_auth.publish_event_object(test_event_obj)
        if success1:
            print("publish_event_object 测试推送（模拟）成功。")
        else:
            print("publish_event_object 测试推送（模拟）失败。")

        # 测试直接推送字典数据
        simple_dict_payload = {
            "alertType": "manual_test",
            "message": "这是一个简单的字典推送测试",
            "severity": 5,
            "source": "test_script"
        }
        print(f"\n尝试使用publish_event_data推送字典数据...")
        success2 = publisher_with_auth.publish_event_data(simple_dict_payload, event_id_for_log="dict_test_002")
        if success2:
            print("publish_event_data 测试推送（模拟）成功。")
        else:
            print("publish_event_data 测试推送（模拟）失败。")
            
    print("\n--- ServerPublisherHTTP 模块测试完成 ---")