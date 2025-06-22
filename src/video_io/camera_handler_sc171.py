# ~/sc171_monitor_project/src/video_io/camera_handler.py
# (注意我移除了 sc171 的后缀，让它更通用)

import cv2
import numpy as np
from typing import Tuple, Optional

class CameraHandler:
    """一个封装了OpenCV摄像头操作的类，支持参数设置。"""

    def __init__(self, camera_source: int = 0, width: Optional[int] = None, height: Optional[int] = None, fps: Optional[int] = None):
        """
        初始化并打开摄像头，可选择性地设置分辨率和FPS。

        Args:
            camera_source (int): 摄像头索引号。
            width (int, optional): 期望的画面宽度。
            height (int, optional): 期望的画面高度。
            fps (int, optional): 期望的FPS。
        """
        self.camera_source = camera_source
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.is_opened():
            print(f"错误: 无法打开摄像头 {self.camera_source}。")
            self.cap = None
            return

        # 应用参数设置
        if width is not None and height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 打印最终生效的参数
        self.print_info()

    def print_info(self):
        """打印当前摄像头实际生效的参数信息。"""
        if not self.is_opened():
            return
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print("--- 摄像头当前实际参数 ---")
        print(f"  源: {self.camera_source}")
        print(f"  分辨率: {width}x{height}")
        print(f"  报告的FPS: {fps:.2f}")
        print("--------------------------")

    def is_opened(self) -> bool:
        """检查摄像头是否成功打开。"""
        return self.cap is not None and self.cap.isOpened()

    def get_actual_params(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return width, height, fps

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """从摄像头读取一帧图像。"""
        if not self.is_opened():
            return False, None
        return self.cap.read()

    def get_fps(self) -> float:
        """获取摄像头报告的FPS。"""
        if self.is_opened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    def close(self):
        """释放摄像头资源。"""
        if self.is_opened():
            self.cap.release()
            print(f"信息: 摄像头 {self.camera_source} 已关闭。")
        self.cap = None

if __name__ == '__main__':
    # 测试代码
    cam = CameraHandler(2)
    cam.print_info()
    ret, frame = cam.read_frame()
    print(type(frame))
    cv2.imwrite('test.jpg', frame)
    cam.close()