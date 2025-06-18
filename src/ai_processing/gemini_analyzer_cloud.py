# ~/sc171_monitor_project/src/ai_processing/gemini_analyzer_cloud.py
import base64
import cv2 
import numpy as np
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Tuple, List
import os
import json 

try:
    from config_sc171 import (
        GEMINI_API_KEY, 
        GEMINI_OPENAI_COMPATIBLE_BASE_URL,
        GEMINI_MODEL_ID_FOR_VISION, 
        GEMINI_API_TIMEOUT_SECONDS
    )
except ImportError:
    print("警告 [GeminiAnalyzerCloud]: 无法从config_sc171导入Gemini配置。将使用后备值。")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_FALLBACK"
    GEMINI_OPENAI_COMPATIBLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    GEMINI_MODEL_ID_FOR_VISION = "gemini-1.5-flash-latest"
    GEMINI_API_TIMEOUT_SECONDS = 60

# --- Pydantic模型定义 (保持不变) ---
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

class GeminiAnalyzerCloud:
    def __init__(self):
        self.client = None
        self.is_initialized = False
        self.model_id = GEMINI_MODEL_ID_FOR_VISION
        self.timeout = float(GEMINI_API_TIMEOUT_SECONDS)

        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_FALLBACK":
            print("错误 [GeminiAnalyzerCloud]: GEMINI_API_KEY 未正确配置。")
            return
        try:
            self.client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_OPENAI_COMPATIBLE_BASE_URL)
            self.is_initialized = True
            print("GeminiAnalyzerCloud: OpenAI客户端初始化成功。")
            print(f"  使用模型: {self.model_id}, Base URL: {self.client.base_url}, Timeout: {self.timeout}s")
        except Exception as e:
            print(f"错误 [GeminiAnalyzerCloud]: 初始化OpenAI客户端失败: {e}")

    def _encode_frame_to_base64(self, frame_bgr: np.ndarray, image_format: str = ".jpg") -> Optional[str]:
        if frame_bgr is None: return None
        try:
            s, b = cv2.imencode(image_format, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            return base64.b64encode(b).decode('utf-8') if s else None
        except Exception as e: print(f"错误 [_encode_frame_to_base64]: Base64编码失败: {e}"); return None

    def _extract_json_from_string(self, text: str) -> Optional[str]:
        """尝试从可能包含Markdown代码块的文本中提取JSON字符串。"""
        if not text: return None
        # 检查是否有markdown代码块 ```json ... ```
        if text.strip().startswith("```json") and text.strip().endswith("```"):
            text = text.strip()[7:-3].strip() # 移除 ```json 和 ```
        elif text.strip().startswith("```") and text.strip().endswith("```"): # 有些模型可能只返回 ``` ... ```
             text = text.strip()[3:-3].strip()

        # 找到第一个 '{' 和最后一个 '}'
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return text[json_start : json_end+1]
        return None # 未找到有效的JSON结构

    def analyze_image(self, frame_bgr: np.ndarray, yolo_objects_summary: str = "未知物体") -> Optional[GeminiAnalysisResult]:
        if not self.is_initialized or self.client is None:
            print("错误 [GeminiAnalyzerCloud.analyze_image]: 分析器未初始化。")
            return None

        base64_image = self._encode_frame_to_base64(frame_bgr)
        if base64_image is None:
            print("错误 [GeminiAnalyzerCloud.analyze_image]: 图像编码为Base64失败。")
            return None

        mime_type = "image/jpeg"
        
        # Prompt工程是这里的核心，明确要求JSON输出并描述结构
        prompt_text_for_gemini = (
            "你是一个专业的图像风险评估助手。请严格按照以下JSON格式提供你的分析结果，不要包含任何额外的解释或markdown标记。JSON对象必须包含以下顶级键：'risk_level', 'description', 'specific_events_detected'。\n"
            "'specific_events_detected' 对象必须包含 'fire', 'fall_down', 'fighting' 这三个键，每个键对应的值是一个对象，该对象必须包含 'detected' (布尔型)键，并可选包含 'confidence' (浮点型0.0-1.0) 和 'details' (字符串)键。\n"
            "--- 分析任务开始 ---\n"
            f"所附图片中可能存在以下由YOLO检测到的对象：[{yolo_objects_summary}]。\n"
            "请分析图片：\n"
            "1. 给出总体风险评估 (risk_level: '低', '中', '高', 或 '未知')。\n"
            "2. 提供综合文本描述 (description)。\n"
            "3. 判断是否存在特定事件 (specific_events_detected):\n"
            "   - 'fire' (火焰)\n"
            "   - 'fall_down' (人员摔倒)\n"
            "   - 'fighting' (打架斗殴)\n"
            "--- 分析任务结束，请提供严格的JSON输出 ---"
        )
        
        messages_payload = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text_for_gemini},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
            ]}
        ]
        
        print(f"DEBUG [GeminiAnalyzerCloud]: 发送给Gemini的Prompt (部分): {prompt_text_for_gemini[:200]}...")
        content_str = None # 初始化

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages_payload,
                timeout=self.timeout,
                # temperature=0.2, # (可选) 降低温度可能使输出更可预测、更接近JSON格式
                # max_tokens=1024  # (可选) 确保有足够空间返回完整JSON
            )

            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                content_str = completion.choices[0].message.content
                print(f"****** GEMINI RAW RESPONSE START ******\n{content_str}\n****** GEMINI RAW RESPONSE END ******")
                
                json_to_parse = self._extract_json_from_string(content_str)
                
                if json_to_parse:
                    try:
                        parsed_data = GeminiAnalysisResult.model_validate_json(json_to_parse) # Pydantic V2
                        # parsed_data = GeminiAnalysisResult.parse_raw(json_to_parse) # Pydantic V1
                        return parsed_data
                    except ValidationError as ve:
                        print(f"错误 [GeminiAnalyzerCloud]: Pydantic模型验证失败: {ve}")
                        print(f"  尝试解析的JSON字符串: {json_to_parse}")
                        return None # 或者返回一个包含错误信息的默认对象
                    except json.JSONDecodeError as je:
                        print(f"错误 [GeminiAnalyzerCloud]: JSON解析失败 (在Pydantic之前): {je}")
                        print(f"  尝试解析的字符串 (提取后): {json_to_parse}")
                        return None
                else:
                    print(f"警告 [GeminiAnalyzerCloud]: 未能从Gemini返回的内容中提取有效的JSON结构。原始返回: {content_str[:300]}")
                    return None
            else: # API调用成功但没有内容
                error_message = "Gemini API响应 (.create) 中没有有效内容。"
                if completion.choices and completion.choices[0].finish_reason:
                     error_message += f" Finish reason: {completion.choices[0].finish_reason}."
                print(f"错误 [GeminiAnalyzerCloud]: {error_message}")
                return None

        except APITimeoutError: # ... (错误处理与之前类似) ...
            print(f"错误 [GeminiAnalyzerCloud]: 调用Gemini API超时 ({self.timeout}s)。"); return None
        except APIConnectionError as e: print(f"错误 [GeminiAnalyzerCloud]: 调用Gemini API连接失败: {e}"); return None
        except APIError as e: print(f"错误 [GeminiAnalyzerCloud]: 调用Gemini API时返回API错误: {e}"); return None
        except Exception as e: 
            print(f"严重错误 [GeminiAnalyzerCloud]: 调用Gemini API或解析时发生未知异常: {e}")
            import traceback; traceback.print_exc(); return None

# --- 模块级测试代码  ---
if __name__ == '__main__':
    print("--- GeminiAnalyzerCloud 模块测试 (使用OpenAI库和Pydantic结构化输出 - 简化版) ---")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_FALLBACK":
        print("错误：GEMINI_API_KEY未配置。测试中止。")
    else:
        analyzer = GeminiAnalyzerCloud()
        if not analyzer.is_initialized:
            print("Gemini分析器初始化失败。测试中止。")
        else:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root_dir = os.path.abspath(os.path.join(current_script_dir, '../../'))
            test_image_path = os.path.join(project_root_dir, "data", "test_images", "smoke.jpg")

            if not os.path.exists(test_image_path): print(f"测试图片 {test_image_path} 未找到。"); exit()
            test_frame = cv2.imread(test_image_path)
            if test_frame is None: print("无法加载测试图片。"); exit()
            
            yolo_summary_test = "公交车, 多名行人, 交通信号灯" 
            
            print(f"\n调用 analyze_image (模型: {analyzer.model_id})...")
            analysis_result = analyzer.analyze_image(test_frame, yolo_objects_summary=yolo_summary_test)

            print("\n--- Gemini 分析结果 (Pydantic对象) ---")
            if analysis_result:
                print(f"  风险评级: {analysis_result.risk_level}")
                print(f"  事件描述: {analysis_result.description}")
                print(f"  特定事件检测:")
                print(f"    火焰: Detected={analysis_result.specific_events_detected.fire.detected}, Conf={analysis_result.specific_events_detected.fire.confidence}, Details='{analysis_result.specific_events_detected.fire.details}'")
                print(f"    摔倒: Detected={analysis_result.specific_events_detected.fall_down.detected}, Conf={analysis_result.specific_events_detected.fall_down.confidence}, Details='{analysis_result.specific_events_detected.fall_down.details}'")
                print(f"    打架: Detected={analysis_result.specific_events_detected.fighting.detected}, Conf={analysis_result.specific_events_detected.fighting.confidence}, Details='{analysis_result.specific_events_detected.fighting.details}'")
                try:
                    print("\n  完整JSON输出 (Pydantic V2):"); print(analysis_result.model_dump_json(indent=2))
                except AttributeError:
                    try: print("\n  完整JSON输出 (Pydantic V1):"); print(analysis_result.json(indent=2))
                    except AttributeError: print("无法调用 .model_dump_json() 或 .json()。")
            else:
                print("  未能从Gemini获取或解析有效的结构化分析结果。")
    print("\n--- GeminiAnalyzerCloud 模块测试完成 ---")