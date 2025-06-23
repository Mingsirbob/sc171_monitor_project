# ~/sc171_monitor_project/src/video_io/video_saver_sc171.py
import cv2
import numpy as np
from typing import Optional
import os # 用于生成带时间戳的文件名
from datetime import datetime

class VideoSaver:
    def __init__(self, base_filename_template: str, width: int, height: int, fps: float):
        """
        初始化视频写入器。

        Args:
            base_filename_template (str): 基础文件名模板，可以包含格式化占位符，例如 "video_{timestamp}.mp4"。
            width (int): 视频帧的宽度。
            height (int): 视频帧的高度。
            fps (float): 视频的帧率。
        """
        self.base_filename_template = base_filename_template
        self.width = width
        self.height = height
        self.fps = fps
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_filename: Optional[str] = None
        self.is_opened = False
        
        # 第一次使用时会自动调用 open_new_segment
        # print(f"VideoSaver 准备就绪。")

    def _generate_filename(self) -> str:
        """根据模板和当前时间生成唯一的文件名。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 确保目录存在
        directory = os.path.dirname(self.base_filename_template)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # 如果模板中包含 {timestamp}，则替换
        if "{timestamp}" in self.base_filename_template:
            return self.base_filename_template.format(timestamp=timestamp)
        else:
            # 否则，在文件名后追加时间戳，避免覆盖
            name, ext = os.path.splitext(self.base_filename_template)
            return f"{name}_{timestamp}{ext}"

    def open_new_segment(self) -> bool:
        """关闭当前视频段（如果已打开），并开启一个新的视频段文件。"""
        if self.is_opened and self.writer:
            self.writer.release()
            print(f"信息: 上一个视频段 '{self.current_filename}' 已保存。")

        self.current_filename = self._generate_filename()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.current_filename, fourcc, self.fps, (self.width, self.height))
        
        if not self.writer.isOpened():
            print(f"错误: 无法创建或打开新的视频段文件 '{self.current_filename}'")
            self.is_opened = False
            self.writer = None
            return False
        
        self.is_opened = True
        print(f"VideoSaver: 开始录制新的视频段到 '{self.current_filename}'。")
        return True

    def write(self, frame: np.ndarray):
        """向当前打开的视频文件写入一帧。"""
        if self.is_opened and self.writer:
            self.writer.write(frame)
        else:
            # 尝试打开一个新的片段，如果这是第一次写入
            if not self.writer: 
                print("警告: VideoSaver未就绪或已关闭，但收到写入请求。尝试开启新片段...")
                if not self.open_new_segment():
                    print("错误: 尝试开启新片段失败，无法写入帧。")


    def close(self):
        """关闭当前打开的视频文件并释放写入器。"""
        if self.is_opened and self.writer:
            self.writer.release()
            print(f"VideoSaver: 视频段 '{self.current_filename}' 已关闭并保存。")
        self.is_opened = False
        self.writer = None
        # print("VideoSaver 整体已关闭。") # 可以根据需要添加此日志


if __name__ == "__main__":
    from src.video_io.camera_handler_sc171 import CameraHandler
    from config_sc171 import SC171_CAMERA_SOURCE, WIDTH, HEIGHT, DESIRED_FPS
    from src.video_io.video_buffer_sc171 import FrameQueue
    camera = CameraHandler(SC171_CAMERA_SOURCE, WIDTH, HEIGHT, DESIRED_FPS)
    ret, frame = camera.read_frame()
    video_saver = VideoSaver("video_{timestamp}.mp4", WIDTH, HEIGHT, DESIRED_FPS)
    Queue = FrameQueue(max_size=10)
    Queue.put(frame)
    video_saver.open_new_segment()
    video_saver.write(frame)
    video_saver.close()
