# ~/sc171_monitor_project/src/ai_processing/api_infer_wrapper.py
import sys
import os
import json
import numpy as np # SnpeContext.Execute 方法中使用了 np.float32
import time      # timer 装饰器使用了 time (如果保留)
import functools # timer 装饰器使用了 functools (如果保留)

# --- 1. FIBO_LIB 环境检查与底层api_aisdk_py导入 ---
# 这部分逻辑确保了在SC171上能找到并加载FIBO SDK的核心库
lib_path = os.getenv("FIBO_LIB", "")
if lib_path == "":
    print("关键错误：FIBO_LIB 环境变量未设置。请确保已正确执行FIBO环境配置脚本。")
    # 在模块加载时就失败，可以考虑直接抛出异常，让调用者知道环境问题
    raise EnvironmentError("FIBO_LIB环境变量未设置。脚本无法继续。")

if not os.path.isdir(lib_path):
    print(f"关键错误：FIBO_LIB路径 '{lib_path}' 无效或不是一个目录。")
    raise EnvironmentError(f"FIBO_LIB路径 '{lib_path}' 无效。")

# 将FIBO_LIB添加到sys.path，以便Python解释器可以找到api_aisdk_py模块
if lib_path not in sys.path:
    sys.path.append(lib_path)
    print(f"通知：已将FIBO_LIB路径 '{lib_path}' 添加到sys.path。")

try:
    # 这是FIBO提供的底层Python绑定，通常是一个.so或.pyd文件
    from api_aisdk_py import api_infer_py 
    print("通知：成功从FIBO_LIB导入api_aisdk_py。")
except ImportError as e:
    print(f"关键导入错误：无法从FIBO_LIB ('{lib_path}')导入api_aisdk_py。")
    print("可能原因：")
    print("1. FIBO_LIB环境变量未正确指向包含api_aisdk_py模块的目录。")
    print("2. api_aisdk_py模块文件 (如 .so) 不存在于该路径或其架构与当前Python解释器不兼容。")
    print("3. FIBO AI Stack未正确安装或环境脚本未成功执行。")
    print(f"当前Python sys.path: {sys.path}")
    raise ImportError(f"无法导入api_aisdk_py: {e}") from e


# --- 2. 工具函数与枚举类 (来自你提供的api_infer.py) ---

# (可选) timer 装饰器，如果你想用它来测量SnpeContext中方法的执行时间
# 你可以根据需要决定是否保留它
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"DEBUG_TIMER - {str(func)} : 执行耗时: {elapsed_time * 1000:.2f}ms")
        return value
    return wrapper_timer

def generate_config(user_values):
    """
    根据用户提供的值更新配置模板并生成JSON字符串。
    注意：此配置模板应与FIBO AISDK期望的格式严格一致。
    """
    config_template = {
        "name": "infer_config",
        "version": "1.0.0",
        "logger": {
            "log_level": "error", # 默认日志级别
            "log_path": "fibo_ai_sdk.log",
            "pattern": "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^---%L---%$] [thread %t]:%g %# %v",
            "max_size": 1048576,
            "max_count": 10,
            "enable_console": True,
            "enable_file": False
        },
        "device": { # 这部分信息可能由FIBO SDK自动填充或用于特定目的
            "board_ssid": "board_ssid_placeholder",
            "board_name": "board_name_placeholder",
            "board_manufacturer": "qualcomm",
            "board_type": "board_type_placeholder",
            "board_version": "board_version_placeholder",
            "board_arch": "aarch64", # SC171通常是aarch64
            "board_os": "linux",    # SC171运行Ubuntu (Linux)
            "soc_name": "QCS6490",  # 根据SC171实际SoC填写
            "soc_id": "QCS6490",
            "soc_ip": [] # IP核心信息，通常SDK会自动检测或有默认
        },
        "infer_engine": {
            "name": "default_session",
            "version": "1.0.0",
            "strategy": 0,
            "batch_timeout": 1000,
            "engine_num": 1,
            "priority": 0
        },
        "all_models": [ # 这个列表会被用户提供的值覆盖或填充
            {
                "model_name": "", # 将由SnpeContext的dlc_path填充
                "model_size": "", # 可选
                "version": "1.0.0",
                "model_path": "", # 将由SnpeContext的dlc_path填充
                "model_type": "", # 可选，可能是 "DLC"
                "model_cache": False,
                "model_cache_path": "",
                "run_backend": "CPU", # 将由SnpeContext的runtime填充 (CPU, GPU, DSP)
                "run_framework": "SNPE", # 表明使用SNPE框架
                "model_version": "1.0.0",
                "batch_size": 1,
                # "output_names": [] # 将由SnpeContext的output_tensors填充 (如果SDK支持通过配置指定)
                # "external_params": {} # 将由SnpeContext的profile_level填充
            }
        ],
        "graphs": [ # 这个配置的细节取决于FIBO SDK如何使用它
            {
                "graph_name": "infer_engine", # 默认图名
                "version": "1.0.0",
                "graph_params": "",
                # 以下输入输出信息通常从DLC模型本身获取，
                # 但FIBO SDK的配置可能需要预先定义或用于验证
                "graph_input_names": ["input_placeholder"], # 会被实际模型输入名覆盖
                "graph_input_shapes": [[-1]],
                "graph_input_types": ["float32"],
                "graph_input_layouts": ["NCHW"], # 或模型实际布局
                "graph_output_names": ["output_placeholder"],# 会被实际模型输出名覆盖
                "graph_output_shapes": [[-1]],
                "graph_output_types": ["float32"],
                "graph_output_layouts": ["NCHW"], # 或模型实际布局
                "all_nodes_params": {
                    "nodes": [
                        {
                            "node_name": "infer_node", # 节点名
                            "node_type": "infer",    # 节点类型
                            "version": "1.0.0",
                            "run_backend": "", # 会被模型配置中的run_backend覆盖
                            "run_framework": "SNPE",
                            "model_name": "",  # 会被模型配置中的model_name覆盖
                            "model_type": "all",
                            "net_type": "all",
                            # 以下节点级别的输入输出定义，通常也应与模型本身一致
                            "node_input_names": ["input_placeholder_node"],
                            "node_input_types": ["float32"],
                            "node_input_shapes": [[-1]],
                            "node_input_layouts": ["NCHW"],
                            "node_output_names": ["output_placeholder_node"],
                            "node_output_shapes": [[-1]],
                            "node_output_types": ["float32"],
                            "node_output_layouts": ["NCHW"],
                            "extra_node_params": ""
                        }
                    ]
                }
            }
        ],
        "application": { # 应用级别配置，可选
            "name": "default_snpe_app",
            "version": "1.0.0",
            "description": "SNPE inference application via FIBO AISDK",
            "app_params": ""
            # "input_algorithm_name": [], # 如果有前处理算法插件
            # "output_algorithm_name": [],# 如果有后处理算法插件
            # "all_algorithm_params": {}
        }
    }
    
    # 简单的深层更新字典的辅助函数
    def update_dict_recursively(d, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(d.get(key), dict):
                d[key] = update_dict_recursively(d[key], value)
            elif isinstance(value, list) and isinstance(d.get(key), list) and key == "all_models": # 特殊处理all_models列表
                # 假设我们总是替换或填充all_models的第一个元素
                if d[key] and updates[key]:
                    d[key][0] = update_dict_recursively(d[key][0], updates[key][0])
                elif updates[key]: # 如果模板的d[key]为空或不存在
                    d[key] = updates[key]

            elif isinstance(value, list) and isinstance(d.get(key), list) and key == "graphs": # 特殊处理graphs列表
                 if d[key] and updates[key] and d[key][0].get("all_nodes_params") and updates[key][0].get("all_nodes_params"):
                     if d[key][0]["all_nodes_params"].get("nodes") and updates[key][0]["all_nodes_params"].get("nodes"):
                         # 更新第一个node的参数
                         d[key][0]["all_nodes_params"]["nodes"][0] = update_dict_recursively(
                             d[key][0]["all_nodes_params"]["nodes"][0],
                             updates[key][0]["all_nodes_params"]["nodes"][0]
                         )
            else:
                d[key] = value
        return d
    
    config = update_dict_recursively(config_template, user_values)
    return json.dumps(config, indent=4)


class LogLevel: # 与你提供的示例一致
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warn" # 你的示例是 "warn"，通常是 "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerfProfile: # 与你提供的示例一致
    BALANCED = 0
    HIGH_PERFORMANCE = 1
    POWER_SAVER = 2
    SYSTEM_SETTINGS = 3
    SUSTAINED_HIGH_PERFORMANCE = 4
    BURST = 5
    LOW_POWER_SAVER = 6 # 新增
    HIGH_POWER_SAVER = 7 # 新增
    LOW_BALANCED = 8 # 新增
    EXTREME_POWERSAVER = 9 # 新增


class Runtime: # 与你提供的示例一致
    CPU = "CPU"
    GPU = "GPU"
    DSP = "DSP"


# --- 3. SnpeContext 类 (核心封装) ---
class SnpeContext:
    """
    封装与FIBO AISDK (SNPE) 的交互。
    """
    def __init__(self, dlc_path: str,
                 output_tensors: list = [], # DLC模型的输出张量名称列表
                 runtime: str = Runtime.CPU,
                 profile_level: int = PerfProfile.BALANCED, # 修改默认值为BALANCED
                 log_level: str = LogLevel.INFO):
        """
        Args:
            dlc_path : DLC模型文件的绝对路径。
            output_tensors : 模型输出张量的名称列表。如果FIBO SDK支持，
                             留空可能表示获取所有输出。你需要根据SDK行为调整。
            runtime : 运行目标 ("CPU", "GPU", "DSP")。
            profile_level : SNPE性能模式。
            log_level : 日志级别。
        """
        if not os.path.exists(dlc_path):
            raise FileNotFoundError(f"DLC模型文件未找到: {dlc_path}")

        self.m_dlcpath = dlc_path
        self.m_model_name_from_path = os.path.basename(dlc_path) # 从路径中提取模型名
        self.m_output_tensors = list(output_tensors) # 确保是列表副本
        self.m_runtime = runtime
        self.m_profiling_level = profile_level
        self.m_log_level = log_level
        
        try:
            self.m_context = api_infer_py.InferAPI() # 创建底层SDK实例
        except Exception as e:
            print(f"错误：创建api_infer_py.InferAPI()实例失败。")
            raise RuntimeError("无法创建FIBO InferAPI实例。") from e
        
        self.is_initialized = False

    # @timer # 如果需要计时，取消注释
    def Initialize(self) -> int:
        """
        使用构建的配置初始化FIBO AISDK。
        Returns:
            0 表示成功，其他值表示失败。
        """
        if self.is_initialized:
            print("警告：SnpeContext已初始化。")
            return 0 # 或者根据SDK行为决定是否允许重入

        # 构建传递给 generate_config 的 user_values
        # 这里假设 generate_config 中的模板是基础，我们用具体值覆盖它
        user_values = {
            "logger": {
                "log_level": self.m_log_level,
            },
            "all_models": [ # 注意这里是一个列表，通常只配置一个模型
                {
                    "model_name": self.m_model_name_from_path, # 使用从路径提取的模型名
                    "model_path": self.m_dlcpath,
                    "run_framework": "SNPE", # 明确指定SNPE
                    "run_backend": self.m_runtime,
                    # 如果FIBO SDK通过此配置传递输出张量名给底层SNPE
                    # (这取决于FIBO SDK的设计)
                    # 如果FIBO SDK不需要在这里指定，或者output_tensors是给FetchOutputs用的，
                    # 那么这里的 "output_names" 可能不需要，或者其键名不同
                    "output_names": self.m_output_tensors, 
                    "external_params": { # SNPE特定的性能配置
                        "profile_level": self.m_profiling_level,
                    }
                }
            ],
            "graphs": [ # 通常一个模型对应一个图
                {
                    # 如果需要，可以填充图的输入输出信息，但通常DLC模型自带这些信息
                    # "graph_input_names": ["your_actual_input_name_from_dlc"], 
                    # "graph_output_names": self.m_output_tensors,
                    "all_nodes_params": {
                        "nodes":[ # 通常一个推理图只有一个主要的推理节点
                            {
                                "model_name": self.m_model_name_from_path,
                                "run_framework": "SNPE",
                                "run_backend": self.m_runtime,
                                # "node_input_names": ["your_actual_input_name_from_dlc_for_node"],
                                # "node_output_names": self.m_output_tensors_for_node,
                            }
                        ]
                    }
                }
            ]
        }
        
        config_json_str = generate_config(user_values)
        # print(f"DEBUG: 生成的初始化配置JSON:\n{config_json_str}") # 调试时取消注释

        ret_code = self.m_context.Init(config_json_str)
        if ret_code == 0:
            self.is_initialized = True
            print(f"SnpeContext for model '{self.m_model_name_from_path}' on {self.m_runtime} 初始化成功。")
        else:
            # 尝试获取更详细的错误信息（如果FIBO SDK提供）
            # error_detail = self.m_context.GetLastError() # 假设有这样的方法
            print(f"错误：SnpeContext 初始化失败。返回码: {ret_code}。")
            # print(f"FIBO SDK 错误详情: {error_detail}")
        return ret_code

    # @timer # 如果需要计时，取消注释
    def Execute(self, output_names_to_fetch: list, input_feed: dict) -> dict or None:
        """
        执行模型推理。
        Args:
            output_names_to_fetch: 一个列表，包含希望从模型获取的输出张量的名称。
                                   应与初始化时或DLC模型中定义的输出名称一致。
            input_feed: 一个字典，键是模型的输入张量名称，值是预处理后的输入数据 (NumPy数组)。
                        内部会将其转换为展平的float32列表。
        Returns:
            一个字典，键是输出张量名称，值是展平的原始输出数据列表 (float32)。
            如果执行失败，返回 None。
        """
        if not self.is_initialized:
            print("错误：SnpeContext尚未初始化。请先调用Initialize()。")
            return None

        # 将输入数据从NumPy数组转换为展平的float32列表 (根据你提供的示例)
        # 这里假设input_feed的值已经是预处理好的NumPy数组
        try:
            processed_input_feed = {}
            for k, v_numpy_array in input_feed.items():
                if not isinstance(v_numpy_array, np.ndarray):
                    print(f"错误：输入feed中 '{k}' 的值不是NumPy数组，而是 {type(v_numpy_array)}。")
                    return None
                # 确保是float32，然后展平，再转为list
                processed_input_feed[k] = v_numpy_array.astype(np.float32).flatten().tolist()
        except Exception as e:
            print(f"错误：处理输入feed时发生异常: {e}")
            return None
        
        # print(f"DEBUG: 发送到Execute的input_feed键: {list(processed_input_feed.keys())}")
        # print(f"DEBUG: 发送到Execute的output_names_to_fetch: {output_names_to_fetch}")

        # 调用底层SDK的Execute方法
        # 注意：FIBO SDK的Execute可能不直接接收output_names_to_fetch作为参数，
        # FetchOutputs方法才是根据名称获取。
        # 你提供的示例中，Execute不接收output_names，FetchOutputs接收。
        if self.m_context.Execute(processed_input_feed) == 0:
            # 执行成功后，根据output_names_to_fetch获取输出
            # print(f"DEBUG: 尝试从 {output_names_to_fetch} 获取输出。")
            fetched_outputs = self.m_context.FetchOutputs(output_names_to_fetch)
            # print(f"DEBUG: FetchOutputs 返回: {type(fetched_outputs)}")
            # if isinstance(fetched_outputs, dict):
            #     for k,v in fetched_outputs.items():
            #         print(f"  Output '{k}' type: {type(v)}, len: {len(v) if isinstance(v, list) else 'N/A'}")
            return fetched_outputs
        else:
            print("错误：FIBO AISDK Execute() 方法执行失败。")
            # error_detail = self.m_context.GetLastError()
            # print(f"FIBO SDK 错误详情: {error_detail}")
            return None

    # @timer # 如果需要计时，取消注释
    def Release(self) -> int:
        """
        释放所有已分配的资源。
        Returns:
            0 表示成功，其他值表示失败。
        """
        if not self.is_initialized:
            # print("通知：SnpeContext未初始化或已释放，无需再次释放。")
            return 0 # 或者根据SDK行为返回适当的值

        ret_code = self.m_context.Release()
        if ret_code == 0:
            self.is_initialized = False # 标记为未初始化
            print(f"SnpeContext for model '{self.m_model_name_from_path}' 已成功释放资源。")
        else:
            print(f"错误：SnpeContext 释放资源失败。返回码: {ret_code}。")
        return ret_code

# --- (可选) 模块级测试或示例用法 ---
if __name__ == '__main__':
    print("api_infer_wrapper.py 被直接运行。")
    print("通常这个文件作为模块被其他脚本导入。")
    
    # 你可以在这里添加一个非常简单的测试，例如检查FIBO_LIB是否设置
    print(f"FIBO_LIB 环境变量: {os.getenv('FIBO_LIB', '未设置')}")
    
    # 尝试实例化一个 InferAPI (如果FIBO_LIB正确且api_aisdk_py可导入)
    try:
        print("尝试实例化 api_infer_py.InferAPI()...")
        test_api = api_infer_py.InferAPI()
        print("api_infer_py.InferAPI() 实例化成功。")
        # 注意：这里只是实例化，并没有Init或Release，
        # 真正的测试应该在 yolo_detector_sc171.py 或 test_core_snpe_yolo.py 中进行。
    except Exception as e:
        print(f"实例化 api_infer_py.InferAPI() 失败: {e}")