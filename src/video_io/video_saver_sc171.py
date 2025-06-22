# ~/sc171_monitor_project/src/video_io/simple_video_saver.py

import cv2
import numpy as np
from typing import Optional
import queue
import time
from .camera_handler_sc171 import CameraHandler
from .video_buffer_sc171 import FrameQueue


class VideoSaver:
    """
    一个简单的、非线程的视频帧写入器。
    它负责将外部传入的帧写入视频文件。
    所有的多线程控制都在此类外部进行。
    """
    def __init__(self, filename: str, width: int, height: int, fps: float):
        """
        初始化视频写入器。

        Args:
            filename (str): 输出视频文件的路径。
            width (int): 视频帧的宽度。
            height (int): 视频帧的高度。
            fps (float): 视频的帧率。
        """
        self.filename = filename
        
        # 1. 定义视频编码器和创建 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise IOError(f"无法创建或打开视频文件 '{filename}'")
            
        self.is_opened = True
        print(f"VideoSaver 初始化完成，准备写入到 '{self.filename}'。")

    def write(self, frame: np.ndarray):
        """
        向视频文件写入一帧。

        Args:
            frame (np.ndarray): 要写入的图像帧。
        """
        if self.is_opened and self.writer is not None:
            self.writer.write(frame)

    def close(self):
        """
        释放写入器，完成视频文件保存。
        """
        if self.is_opened and self.writer is not None:
            self.writer.release()
            self.is_opened = False
            self.writer = None
            print(f"VideoSaver 已关闭，视频 '{self.filename}' 保存完成。")

if __name__ == "__main__":
        # --- 1. 配置参数 ---
    CAM_INDEX = 2
    # 尝试使用摄像头默认参数，如果摄像头支持，也可以尝试设置
    # WIDTH, HEIGHT, FPS = 640, 480, 30
    RECORD_DURATION_SECONDS = 15
    OUTPUT_FILE = "single_thread_test_video.mp4"
    QUEUE_MAX_SIZE = 256 # 队列大小，对于单线程测试，可以大一些以观察效果

    camera = None
    saver = None
    
    print("--- 开始单线程视频缓存与保存测试 ---")

    try:
        # --- 2. 初始化摄像头处理器 ---
        # camera = CameraHandler(CAM_INDEX, WIDTH, HEIGHT, FPS)
        camera = CameraHandler(CAM_INDEX) # 使用摄像头默认参数
        if not camera.is_opened():
            raise RuntimeError("无法打开摄像头，测试终止。")

        # 获取摄像头实际参数用于 VideoSaver
        actual_width, actual_height, actual_fps = camera.get_actual_params()
        if actual_width is None or actual_fps == 0: # 如果FPS为0也认为参数获取失败
            print("警告: 无法获取摄像头实际参数，将使用默认值 640x480 @ 30fps 进行保存。")
            actual_width, actual_height, actual_fps = 640, 480, 30.0
        else:
            print(f"将使用摄像头实际参数进行保存: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")


        # --- 3. 初始化帧队列 ---
        frame_queue = FrameQueue(max_size=QUEUE_MAX_SIZE)
        print(f"帧队列已创建，最大容量: {QUEUE_MAX_SIZE}")

        # --- 4. 初始化视频保存器 ---
        saver = VideoSaver(OUTPUT_FILE, actual_width, actual_height, actual_fps)

        # --- 5. 单线程主循环：读取 -> 入队 -> 出队 -> 保存 ---
        print(f"\n开始进行 {RECORD_DURATION_SECONDS} 秒的视频缓存与保存...")
        start_time = time.time()
        frames_read = 0
        frames_written = 0

        while (time.time() - start_time) < RECORD_DURATION_SECONDS:
            # a. 从摄像头读取帧
            ret, frame = camera.read_frame()
            if not ret:
                print("无法从摄像头读取帧，可能视频流结束。")
                break
            frames_read += 1

            # b. 将帧放入队列 (生产者行为)
            #    在单线程中，如果队列满了，put会阻塞，但我们预期它不会满，或者很快被消耗
            if not frame_queue.is_full():
                frame_queue.put(frame)
            else:
                print("警告: 帧队列已满，可能发生丢帧 (理论上单线程不应如此快速填满)。")

            # c. 从队列中取出帧 (消费者行为)
            if not frame_queue.is_empty():
                frame_to_save = frame_queue.get()
                if frame_to_save is not None:
                    # d. 将帧写入视频文件
                    saver.write(frame_to_save)
                    frames_written += 1
        
        elapsed_time = time.time() - start_time
        print(f"\n{RECORD_DURATION_SECONDS}秒录制时间到。")
        print(f"实际耗时: {elapsed_time:.2f} 秒")
        print(f"总共读取帧数: {frames_read}")
        print(f"总共写入帧数: {frames_written}")
        print(f"队列中剩余帧数: {frame_queue.get_size()}")

        # e. 将队列中剩余的帧全部写入文件 (确保所有缓存的都保存)
        print("正在清空队列并保存剩余帧...")
        while not frame_queue.is_empty():
            remaining_frame = frame_queue.get()
            if remaining_frame is not None:
                saver.write(remaining_frame)
                frames_written += 1
        print(f"清空队列后，总共写入帧数: {frames_written}")


    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    
    finally:
        # --- 6. 关闭所有资源 ---
        print("\n正在关闭所有资源...")
        if camera:
            camera.close()
        if saver:
            saver.close()
        cv2.destroyAllWindows()

    print("\n--- 单线程测试结束 ---")
    print(f"请检查目录下是否生成了视频文件: '{OUTPUT_FILE}'")