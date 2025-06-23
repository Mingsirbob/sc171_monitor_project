# ~/sc171_monitor_project/src/video_io/video_buffer_sc171.py
import cv2
import os
import time
from datetime import datetime, timezone
import numpy as np
from typing import Optional
from collections import deque
import queue
import threading


try:
    from config_sc171 import VIDEO_CACHE_DIR as DEFAULT_CACHE_DIR, DESIRED_FPS as DEFAULT_FPS
except ImportError:
    DEFAULT_CACHE_DIR = "temp_video_buffer_cache_simple"
    DEFAULT_FPS = 30.0

class FrameStack:
    """
    一个线程安全的、固定大小的帧栈 (后进先出)。
    专门为实时处理设计，当栈满时，新来的帧会自动挤掉最老的帧。
    """
    def __init__(self, max_size: int = 5):
        """
        初始化帧栈。

        Args:
            max_size (int): 栈的最大容量。这个值不需要很大，因为它只为了缓冲最新的几帧。
        """
        if max_size <= 0:
            raise ValueError("max_size必须是正整数。")
            
        self._deque = deque(maxlen=max_size)
        # 创建一个锁来保证复合操作的线程安全
        self._lock = threading.Lock()
        self.last_frame = None
        self.count = 0

    def push(self, frame: np.ndarray):
        """
        将一个新帧压入栈顶。这是一个非阻塞操作。

        Args:
            frame (np.ndarray): 要添加的图像帧。
        """
        # deque的append是线程安全的原子操作，所以不需要锁
        self.count = self.count + 1
        print(f"当前入栈的帧的序号是{self.count}")
        self._deque.append(frame)

    def get_latest(self) -> Optional[np.ndarray]:
        """
        获取当前栈顶最新的一帧，但不将其从栈中移除。
        这是一个非阻塞操作。

        Returns:
            Optional[np.ndarray]: 最新的帧，如果栈为空则返回 None。
        """
        print(f"当前栈顶最新的一帧的序号是{self.count}")
        with self._lock:
            if len(self._deque) > 0:
                # 返回deque的最后一个元素，即栈顶
                self.last_frame = self._deque[-1]
                return self.last_frame
            return None

    def get_size(self) -> int:
        """返回当前栈中的帧数。"""
        return len(self._deque)

class FrameQueue:
    """
    一个线程安全的、固定大小的帧队列 (先进先出)。
    专门为需要处理每一帧的场景设计，如视频录制。
    """
    def __init__(self, max_size: int = 128):
        """
        初始化帧队列。

        Args:
            max_size (int): 队列的最大容量。这个值应该足够大，以缓冲生产者和消费者之间的速度差异。
                            例如，如果摄像头30fps，录制器15fps，那么每秒会积压15帧。
        """
        if max_size <= 0:
            raise ValueError("max_size必须是正整数。")
        
        # 直接使用内置的线程安全队列
        self._queue = queue.Queue(maxsize=max_size)

    def put(self, frame: np.ndarray, block: bool = True, timeout: Optional[float] = None):
        """
        将一个新帧放入队尾。

        Args:
            frame (np.ndarray): 要添加的图像帧。
            block (bool): 如果为True，当队列满时会阻塞（等待），直到有空间。
            timeout (float, optional): 等待的超时秒数。
        
        Returns:
            bool: 如果成功放入则返回True，如果因超时或非阻塞模式下队列已满则返回False。
        """
        try:
            self._queue.put(frame, block=block, timeout=timeout)
            return True
        except queue.Full:
            # 仅在非阻塞或超时情况下会发生
            # print("警告: FrameQueue 已满，丢弃一帧。")
            return False

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        从队首获取一帧。

        Args:
            block (bool): 如果为True，当队列空时会阻塞（等待），直到有新帧。
            timeout (float, optional): 等待的超时秒数。

        Returns:
            Optional[np.ndarray]: 获取到的帧，如果因超时或非阻塞模式下队列为空则返回 None。
        """
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            # 仅在非阻塞或超时情况下会发生
            return None

    def get_size(self) -> int:
        """返回当前队列中的帧数。"""
        return self._queue.qsize()

    def is_full(self) -> bool:
        """检查队列是否已满。"""
        return self._queue.full()

    def is_empty(self) -> bool:
        """检查队列是否为空。"""
        return self._queue.empty()


# --- 简化后的模块级测试代码 ---
# if __name__ == '__main__':
#     print("--- VideoBufferSC171 简化测试 ---")
#     print("\n--- VideoBufferSC171 简化测试完成 ---")