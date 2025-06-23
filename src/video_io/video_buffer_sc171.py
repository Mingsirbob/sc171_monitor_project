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
        # print(f"当前入栈的帧的序号是{self.count}")
        self._deque.append(frame)

    def get_latest(self) -> Optional[np.ndarray]:
        """
        获取当前栈顶最新的一帧，但不将其从栈中移除。
        这是一个非阻塞操作。

        Returns:
            Optional[np.ndarray]: 最新的帧，如果栈为空则返回 None。
        """
        # print(f"当前栈顶最新的一帧的序号是{self.count}")
        with self._lock:
            if len(self._deque) > 0:
                # 返回deque的最后一个元素，即栈顶
                self.last_frame = self._deque[-1]
                return self.last_frame
            return None

    def get_size(self) -> int:
        """返回当前栈中的帧数。"""
        return len(self._deque)

class FrameQueue: # 现在更像一个有界、线程安全的deque封装
    """
    一个线程安全的、固定大小的帧队列 (底层使用deque实现滑动窗口)。
    当队列满时，新加入的帧会自动替换掉最早的帧。
    """
    def __init__(self, max_size: int = 128):
        if max_size <= 0:
            raise ValueError("max_size必须是正整数。")
        
        # 使用deque作为底层数据结构，maxlen参数自动处理了固定大小和旧元素出栈的逻辑
        self._deque = deque(maxlen=max_size)
        
        # deque的核心操作是线程安全的，但如果需要复合操作或更强的保证，可以保留锁
        self._lock = threading.Lock() 
        # 对于简单的put和get，deque通常足够，但如果未来有更复杂的操作，锁是有用的

    def put(self, frame: np.ndarray): # put现在是非阻塞的，且自动替换
        """
        将一个新帧放入队列。如果队列已满，最早的帧将被移除。
        这是一个非阻塞操作。

        Args:
            frame (np.ndarray): 要添加的图像帧。
        """
        # deque.append() 是线程安全的，并且会自动处理maxlen
        with self._lock: # 保护deque的修改，虽然append是原子的，但复合操作可能需要
            self._deque.append(frame)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        从队首获取一帧。

        Args:
            block (bool): 如果为True且队列空，则阻塞等待。
            timeout (float, optional): 等待的超时秒数。

        Returns:
            Optional[np.ndarray]: 获取到的帧，或None。
        """
        # 我们需要模拟queue.Queue的阻塞行为，因为deque.popleft()在空时会抛异常
        if block:
            start_time = time.time()
            while True: # 模拟阻塞等待
                with self._lock:
                    if len(self._deque) > 0:
                        return self._deque.popleft() # 从左边（队首）取出
                if timeout is not None and (time.time() - start_time) >= timeout:
                    return None # 超时
                time.sleep(0.001) # 短暂休眠，避免CPU空转，并给其他线程机会
        else: # 非阻塞
            with self._lock:
                if len(self._deque) > 0:
                    return self._deque.popleft()
            return None # 非阻塞且队列为空

    def get_size(self) -> int:
        with self._lock:
            return len(self._deque)

    def is_full(self) -> bool: # deque没有直接的full()，但我们可以判断size是否等于maxlen
        with self._lock:
            return len(self._deque) == self._deque.maxlen

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._deque) == 0


# --- 简化后的模块级测试代码 ---
# if __name__ == '__main__':
#     print("--- VideoBufferSC171 简化测试 ---")
#     print("\n--- VideoBufferSC171 简化测试完成 ---")