# ~/sc171_monitor_project/src/ai_processing/gemini_analyzer_cloud.py
import base64
import cv2 # 用于图像编码
import numpy as np
from openai import OpenAI # 使用OpenAI库
from pydantic import BaseModel, Field, ValidationError # 用于结构化输出
from typing import Optional, Tuple, List # 用于类型提示
import os # 用于测试时加载图片

# 尝试从config导入配置
try:
    from config_sc171 import GEMINI_API_KEY, GEMINI_OPENAI_COMPATIBLE_BASE_URL, \
                               GEMINI_MODEL_ID_FOR_VISION, GEMINI_API_TIMEOUT_SECONDS
except ImportError:
    print("警告 [GeminiAnalyzerCloud]: 无法从config_sc171导入配置。将使用硬编码的占位符值。")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_FALLBACK" # 仅为模块能加载，实际无法工作
    GEMINI_OPENAI_COMPATIBLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai" # 示例
    GEMINI_MODEL_ID_FOR_VISION = "gemini-1.5-flash-latest" # 示例
    GEMINI_API_TIMEOUT_SECONDS = 60


# 定义期望的结构化输出模型
class GeminiStructuredResponse(BaseModel):
    risk_level: str = Field(..., description="事件的风险等级 (例如: 低, 中, 高, 未知)")
    description: str = Field(..., description="对事件内容的详细文本描述")

class GeminiAnalyzerCloud:
    def __init__(self):
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_FALLBACK":
            print("错误 [GeminiAnalyzerCloud]: GEMINI_API_KEY 未正确配置。无法初始化OpenAI客户端。")
            self.client = None
            self.is_initialized = False
            return

        try:
            self.client = OpenAI(
                api_key=GEMINI_API_KEY,
                base_url=GEMINI_OPENAI_COMPATIBLE_BASE_URL 
                # 示例中是 "https://generativelanguage.googleapis.com/v1beta/openai/"
                # 如果Google的OpenAI兼容层要求在base_url后加 /v1 或其他，需要调整
                # 通常，OpenAI库会自动附加 /chat/completions 等路径
            )
            self.model_id = GEMINI_MODEL_ID_FOR_VISION
            self.timeout = GEMINI_API_TIMEOUT_SECONDS
            self.is_initialized = True
            print("GeminiAnalyzerCloud: OpenAI客户端初始化成功 (用于访问Gemini)。")
            print(f"  使用模型: {self.model_id}")
            print(f"  Base URL: {self.client.base_url}") # 确认base_url
        except Exception as e:
            print(f"错误 [GeminiAnalyzerCloud]: 初始化OpenAI客户端失败: {e}")
            self.client = None
            self.is_initialized = False

    def _encode_frame_to_base64(self, frame_bgr: np.ndarray, image_format: str = ".jpg") -> Optional[str]:
        """将OpenCV图像帧编码为Base64字符串。"""
        if frame_bgr is None:
            return None
        try:
            success, encoded_image_bytes = cv2.imencode(image_format, frame_bgr)
            if not success:
                print(f"错误 [GeminiAnalyzerCloud]: cv2.imencode 图像到 '{image_format}' 失败。")
                return None
            base64_image_string = base64.b64encode(encoded_image_bytes).decode('utf-8')
            return base64_image_string
        except Exception as e:
            print(f"错误 [GeminiAnalyzerCloud]: 图像Base64编码失败: {e}")
            return None

    def analyze_image(self, frame_bgr: np.ndarray, prompt_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        使用Gemini (通过OpenAI库) 分析图像帧。
        Args:
            frame_bgr: OpenCV BGR图像帧。
            prompt_text: 指导Gemini分析的文本提示，应引导其输出符合GeminiStructuredResponse的JSON。
        Returns:
            一个元组 (risk_level, description)。如果失败则为 (None, None)。
        """
        if not self.is_initialized or self.client is None:
            print("错误 [GeminiAnalyzerCloud.analyze_image]: 分析器未初始化。")
            return None, None

        base64_image = self._encode_frame_to_base64(frame_bgr)
        if base64_image is None:
            return "分析失败", "图像编码错误" # 返回具体的错误信息

        mime_type = "image/jpeg" if ".jpg" in self._encode_frame_to_base64.__defaults__[0] else "image/png"
        
        messages_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]
        # 为了获取结构化JSON，我们可以在prompt中强烈要求，并尝试解析
        # 如果你的OpenAI库版本支持 response_model 或类似 tool_choice 的功能，可以尝试使用
        # 这里我们先用直接解析JSON字符串的方式，并在prompt中要求JSON输出

        print(f"DEBUG [GeminiAnalyzerCloud]: 发送给Gemini的Prompt (部分): {prompt_text[:100]}...")
        # print(f"DEBUG [GeminiAnalyzerCloud]: 发送给Gemini的图像数据 (Base64, 前100字符): {base64_image[:100]}...")


        try:
            # 如果希望强制JSON输出且模型支持（Gemini通过OpenAI兼容层是否支持需测试）
            # response_format_param = {"type": "json_object"}
            # completion = self.client.chat.completions.create(
            # model=self.model_id,
            # messages=messages_payload,
            # timeout=self.timeout,
            # response_format=response_format_param # 仅当模型和库版本支持时
            # )
            
            # 标准调用
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages_payload,
                timeout=self.timeout,
                # max_tokens=512 # (可选) 限制输出长度
            )

            # print(f"DEBUG [GeminiAnalyzerCloud]: Gemini API原始响应: {completion}")

            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                content_str = completion.choices[0].message.content
                # print(f"DEBUG [GeminiAnalyzerCloud]: Gemini返回的内容字符串: {content_str}")
                
                # 尝试将内容字符串解析为JSON，然后用Pydantic模型验证
                try:
                    # Gemini可能不会完美地只返回JSON，可能包含前后缀文本或markdown代码块
                    # 尝试从中提取JSON部分
                    json_start = content_str.find('{')
                    json_end = content_str.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str_to_parse = content_str[json_start : json_end+1]
                        # print(f"DEBUG [GeminiAnalyzerCloud]: 提取到的JSON字符串: {json_str_to_parse}")
                        parsed_data = GeminiStructuredResponse.model_validate_json(json_str_to_parse)
                        # Pydantic V1: parsed_data = GeminiStructuredResponse.parse_raw(json_str_to_parse)
                        return parsed_data.risk_level, parsed_data.description
                    else:
                        print(f"警告 [GeminiAnalyzerCloud]: Gemini返回的内容似乎不是有效的JSON对象。内容: {content_str}")
                        return "解析失败", f"非JSON响应: {content_str[:200]}"

                except ValidationError as ve:
                    print(f"错误 [GeminiAnalyzerCloud]: Pydantic模型验证失败: {ve}")
                    return "验证失败", f"Pydantic错误: {str(ve)[:200]}"
                except json.JSONDecodeError as je:
                    print(f"错误 [GeminiAnalyzerCloud]: JSON解析失败: {je}")
                    return "解析失败", f"JSON错误: {str(je)[:200]}"
            else:
                print("错误 [GeminiAnalyzerCloud]: Gemini API响应中没有有效内容。")
                return "无响应内容", None

        except Exception as e:
            print(f"错误 [GeminiAnalyzerCloud]: 调用Gemini API失败: {e}")
            import traceback
            traceback.print_exc()
            return "API调用失败", str(e)[:200] # 返回错误信息作为描述


# --- 模块级测试代码 ---
if __name__ == '__main__':
    print("--- GeminiAnalyzerCloud 模块测试 (使用OpenAI库) ---")

    # 确保config_sc171.py可访问，或者GEMINI_API_KEY已通过其他方式设置
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_FALLBACK":
        print("错误：GEMINI_API_KEY未配置。请在.env文件中设置，并通过config_sc171.py加载。测试中止。")
    else:
        analyzer = GeminiAnalyzerCloud()
        if not analyzer.is_initialized:
            print("Gemini分析器初始化失败。测试中止。")
        else:
            # SC171上的测试图片路径 (相对于项目根目录的data/test_images/)
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root_dir = os.path.abspath(os.path.join(current_script_dir, '../../'))
            test_image_filename = "bus.jpg" # 使用你之前测试YOLO的图片
            test_image_path = os.path.join(project_root_dir, "data", "test_images", test_image_filename)

            if not os.path.exists(test_image_path):
                print(f"错误：测试图片 '{test_image_path}' 未找到。")
            else:
                print(f"加载测试图片: {test_image_path}")
                test_frame = cv2.imread(test_image_path)
                if test_frame is None:
                    print(f"错误：无法使用OpenCV加载测试图片。")
                else:
                    # 精心设计Prompt以引导Gemini输出JSON
                    prompt = (
                        "分析以下图片中的场景。\n"
                        "评估当前事件的潜在风险，并给出风险评级。\n"
                        "同时，简要描述图片中的主要事件内容。\n"
                        "请严格以JSON格式返回结果，该JSON对象应包含两个键：\n"
                        "1. 'risk_level': 字符串，值为'低'、'中'或'高'中的一个。\n"
                        "2. 'description': 字符串，为事件的详细文本描述。\n"
                        "不要包含任何JSON之外的解释性文字或markdown标记。"
                    )
                    print(f"\n发送给Gemini的Prompt (用于结构化输出):\n{prompt}")
                    
                    risk, desc = analyzer.analyze_image(test_frame, prompt_text=prompt)

                    print("\n--- Gemini 分析结果 ---")
                    if risk is not None and desc is not None:
                        print(f"  风险评级: {risk}")
                        print(f"  事件描述: {desc}")
                    else:
                        print("  未能从Gemini获取或解析有效的分析结果。")
                        if risk: print(f"  部分风险信息: {risk}")
                        if desc: print(f"  部分描述信息: {desc}")

    print("\n--- GeminiAnalyzerCloud 模块测试完成 ---")