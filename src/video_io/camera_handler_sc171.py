# ~/sc171_monitor_project/src/video_io/camera_handler_sc171.py
import cv2
import time
import os
import numpy as np # 确保导入numpy
from typing import Tuple, Union, Optional # <--- 导入 typing 中的类型

# ... (尝试导入config_sc171的代码) ...
try:
    from config_sc171 import SC171_CAMERA_SOURCE, DESIRED_FPS
except ImportError:
    print("警告 [CameraHandlerSC171]: 无法从config_sc171导入配置。将使用硬编码默认值。")
    SC171_CAMERA_SOURCE = 2

    DESIRED_FPS = 20.0

class CameraHandlerSC171:
    def __init__(self, camera_source=None, desired_width=None, desired_height=None, desired_fps=None):
        self.camera_source = camera_source if camera_source is not None else SC171_CAMERA_SOURCE
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.desired_fps = desired_fps if desired_fps is not None else DESIRED_FPS
        self.cap = None
        self.is_opened = False
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        print(f"CameraHandlerSC171: 初始化，源: {self.camera_source}")


    def open(self) -> bool:
        # ... (open方法内容) ...
        if self.is_opened:
            print("CameraHandlerSC171: 摄像头已打开。")
            return True
        try:
            if isinstance(self.camera_source, str) and ("gst-launch" in self.camera_source.lower() or "rtspsrc" in self.camera_source.lower()):
                self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_GSTREAMER)
                print(f"CameraHandlerSC171: 尝试使用GStreamer管道打开: {self.camera_source}")
            else:
                try:
                    source_to_open = int(self.camera_source)
                except ValueError:
                    source_to_open = self.camera_source
                self.cap = cv2.VideoCapture(source_to_open)
                print(f"CameraHandlerSC171: 尝试使用OpenCV标准方式打开: {source_to_open}")
            if not self.cap or not self.cap.isOpened():
                print(f"错误 [CameraHandlerSC171]: 无法打开摄像头源: {self.camera_source}")
                self.is_opened = False
                return False
            if self.desired_width is not None: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
            if self.desired_height is not None: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
            if self.desired_fps is not None: self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.frame_width == 0 or self.frame_height == 0:
                print(f"警告 [CameraHandlerSC171]: 打开摄像头后获取到的分辨率为0。源可能无效或配置错误。")
                self.cap.release(); self.is_opened = False; return False
            if self.fps <= 0:
                print(f"警告 [CameraHandlerSC171]: 摄像头报告的FPS为 {self.fps}。将使用期望FPS {self.desired_fps} 或默认值20。")
                self.fps = self.desired_fps if self.desired_fps and self.desired_fps > 0 else 20.0
            self.is_opened = True
            print(f"CameraHandlerSC171: 摄像头 {self.camera_source} 打开成功。")
            print(f"  实际分辨率: {self.frame_width}x{self.frame_height}"); print(f"  实际/设定FPS: {self.fps:.2f}")
            return True
        except Exception as e:
            print(f"错误 [CameraHandlerSC171]: 打开摄像头时发生异常: {e}")
            if self.cap: self.cap.release()
            self.is_opened = False; return False


    # --- 修改这里的返回类型提示 ---
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        从打开的摄像头读取一帧。
        Returns:
            一个元组 (success, frame)。
            success (bool): 如果成功读取帧则为True。
            frame (Optional[np.ndarray]): 读取到的图像帧，如果失败则为None。
        """
        # ... (read_frame方法内部逻辑保持不变) ...
        if not self.is_opened or not self.cap:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    # --- 修改这里的返回类型提示 ---
    def get_resolution(self) -> Optional[Tuple[int, int]]:
        """获取当前视频流的分辨率 (宽度, 高度)。"""
        if self.is_opened:
            return self.frame_width, self.frame_height
        return None # 直接返回None，Optional[Tuple[int,int]] 会处理这种情况

    def get_fps(self) -> float: # float类型提示通常没问题
        """获取当前视频流的FPS。"""
        # ... (get_fps方法内容保持不变) ...
        if self.is_opened:
            return self.fps
        return 0.0

    def close(self):
        # ... (close方法内容保持不变) ...
        if self.cap:
            print(f"CameraHandlerSC171: 正在关闭摄像头 {self.camera_source}...")
            self.cap.release()
            print(f"CameraHandlerSC171: 摄像头 {self.camera_source} 已关闭。")
        self.is_opened = False
        self.cap = None

# --- 模块级测试代码 (保持不变) ---
if __name__ == '__main__':
    # ... (测试代码保持不变) ...
    print("--- CameraHandlerSC171 模块测试 ---")
    test_camera_source = SC171_CAMERA_SOURCE 
    print(f"将尝试打开摄像头源: {test_camera_source}")
    handler = CameraHandlerSC171(camera_source=test_camera_source)
    if handler.open():
        print("\n摄像头打开成功，开始读取帧...")
        frame_count = 0; max_frames_to_test = 100; start_time = time.time()
        while frame_count < max_frames_to_test:
            ret, frame = handler.read_frame()
            if not ret: print(f"读取第 {frame_count + 1} 帧失败。"); break
            frame_count += 1
            if frame_count % 20 == 0: print(f"  已读取 {frame_count} 帧，当前帧形状: {frame.shape if frame is not None else 'None'}")
            if frame_count == 1 or frame_count == 50 or frame_count == 100:
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/cam_test_frames")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"sc171_frame_{frame_count}.jpg")
                try:
                    if frame is not None: cv2.imwrite(save_path, frame); print(f"  已保存帧到: {save_path}")
                except Exception as e: print(f"  保存帧失败: {e}")
        end_time = time.time(); duration = end_time - start_time
        actual_test_fps = frame_count / duration if duration > 0 else 0
        print(f"\n读取测试完成。共读取 {frame_count} 帧。"); print(f"测试耗时: {duration:.2f} 秒。"); print(f"测试期间平均FPS: {actual_test_fps:.2f}")
        res_w, res_h = handler.get_resolution() if handler.get_resolution() is not None else (None, None)
        reported_fps = handler.get_fps()
        print(f"Handler报告的分辨率: {res_w}x{res_h}"); print(f"Handler报告的FPS: {reported_fps:.2f}")
        handler.close()
    else:
        print("\n摄像头打开失败。请检查：")
        print(f"1. 摄像头是否正确连接到SC171并通过 '{test_camera_source}' 可访问。")
        print(f"   (尝试 `ls /dev/video*` 查看可用的视频设备)")
        print("2. 当前用户是否有权限访问摄像头设备。")
        print("3. OpenCV是否能与SC171的摄像头驱动正常工作。")
        print("4. 如果使用GStreamer管道，请确保管道字符串正确且GStreamer已安装。")
    print("\n--- CameraHandlerSC171 模块测试完成 ---")