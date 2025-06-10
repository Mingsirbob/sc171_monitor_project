# ~/sc171_monitor_project/src/video_io/video_buffer_sc171.py
import cv2
import os
import time
from datetime import datetime, timezone # <--- timezone 用于 _generate_filename
import numpy as np # <--- 新增导入 numpy
from typing import Optional # <--- 新增导入 Optional

# ... (尝试导入config_sc171的代码保持不变) ...
try:
    from config_sc171 import VIDEO_CACHE_DIR, DESIRED_FPS
except ImportError:
    print("警告 [VideoBufferSC171]: 无法从config_sc171导入配置。将使用硬编码后备值。")
    VIDEO_CACHE_DIR = "temp_video_buffer_cache"
    DESIRED_FPS = 20.0


class VideoBufferSC171:
    def __init__(self, 
                 camera_id: str, 
                 cache_duration_seconds: int, 
                 output_directory_root: str, 
                 fps: float, 
                 frame_width: int, 
                 frame_height: int,
                 fourcc_str: str = 'mp4v'):
        # ... (构造函数内容保持不变) ...
        self.camera_id = str(camera_id) 
        self.cache_duration_seconds = int(cache_duration_seconds)
        self.output_directory = os.path.join(output_directory_root, self.camera_id)
        self.fps = float(fps) if fps > 0 else float(DESIRED_FPS) 
        if self.fps <=0 : self.fps = 20.0 
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.fourcc_str = fourcc_str
        self.current_video_writer = None
        self.current_segment_start_time_monotonic = 0.0
        self.current_segment_path = None
        self.is_writer_open = False
        try:
            os.makedirs(self.output_directory, exist_ok=True)
            print(f"VideoBufferSC171 [{self.camera_id}]: 日志输出目录 '{self.output_directory}' 已确保存在。")
        except OSError as e:
            print(f"错误 [VideoBufferSC171]: 创建输出目录 '{self.output_directory}' 失败: {e}")
            return
        self._start_new_segment()

    def _generate_filename(self) -> str: # 返回类型是 str，不是 Optional[str]
        # ... (内容保持不变) ...
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        extension = ".mp4" if self.fourcc_str.lower() in ['mp4v', 'h264', 'avc1'] else ".avi"
        return f"segment_{self.camera_id}_{timestamp_str}{extension}"

    def _start_new_segment(self) -> bool:
        # ... (内容保持不变) ...
        if self.current_video_writer and self.is_writer_open:
            try: self.current_video_writer.release(); print(f"VideoBufferSC171 [{self.camera_id}]: 上一个片段 '{os.path.basename(self.current_segment_path)}' 已关闭。")
            except Exception as e: print(f"错误 [VideoBufferSC171]: 关闭上一个VideoWriter时发生异常: {e}")
        self.is_writer_open = False
        self.current_segment_path = os.path.join(self.output_directory, self._generate_filename())
        try: fourcc_val = cv2.VideoWriter_fourcc(*self.fourcc_str.upper())
        except Exception as e: print(f"错误 [VideoBufferSC171]: 无效的FourCC字符串 '{self.fourcc_str}': {e}. 将尝试默认 'mp4v'。"); fourcc_val = cv2.VideoWriter_fourcc(*'MP4V')
        print(f"VideoBufferSC171 [{self.camera_id}]: 尝试使用FourCC '{self.fourcc_str.upper()}' (值: {fourcc_val})")
        print(f"VideoBufferSC171 [{self.camera_id}]: FPS: {self.fps}, 尺寸: {self.frame_width}x{self.frame_height}")
        try:
            self.current_video_writer = cv2.VideoWriter(self.current_segment_path, fourcc_val, self.fps, (self.frame_width, self.frame_height))
            if not self.current_video_writer.isOpened():
                print(f"错误 [VideoBufferSC171]: 无法打开VideoWriter以保存到 '{self.current_segment_path}'。"); print(f"  请检查编解码器 ('{self.fourcc_str.upper()}') 是否在SC171上受支持，以及路径和权限。")
                self.current_video_writer = None; return False
            self.current_segment_start_time_monotonic = time.monotonic(); self.is_writer_open = True
            print(f"VideoBufferSC171 [{self.camera_id}]: 开始录制新片段到: {self.current_segment_path}")
            return True
        except Exception as e: print(f"错误 [VideoBufferSC171]: 初始化VideoWriter时发生异常: {e}"); self.current_video_writer = None; return False

    # --- 修改这里的返回类型提示 ---
    def add_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        向当前视频片段添加一帧。
        Args:
            frame: 要添加的 OpenCV 图像帧 (BGR, HWC)。
        Returns:
            如果一个片段已录制满时长并已成功保存，则返回该片段的完整路径。
            否则返回 None。
        """
        # ... (add_frame方法内部逻辑保持不变) ...
        if not self.is_writer_open or not self.current_video_writer: return None
        try:
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                frame_to_write = cv2.resize(frame, (self.frame_width, self.frame_height))
            else: frame_to_write = frame
            self.current_video_writer.write(frame_to_write)
        except Exception as e:
            print(f"错误 [VideoBufferSC171]: 写入帧到VideoWriter时发生异常: {e}"); self.close(); self._start_new_segment(); return None
        current_duration = time.monotonic() - self.current_segment_start_time_monotonic
        if current_duration >= self.cache_duration_seconds:
            path_of_completed_segment = self.current_segment_path
            print(f"VideoBufferSC171 [{self.camera_id}]: 片段 '{os.path.basename(path_of_completed_segment)}' "
                  f"已达到 {current_duration:.2f}s (目标: {self.cache_duration_seconds}s)。")
            if self._start_new_segment(): return path_of_completed_segment
            else: print(f"错误 [VideoBufferSC171]: 开始新片段失败，无法最终确定旧片段 '{path_of_completed_segment}'。"); return None
        return None

    def close(self):
        # ... (close方法内容保持不变) ...
        if self.current_video_writer and self.is_writer_open:
            try: self.current_video_writer.release(); self.is_writer_open = False; print(f"VideoBufferSC171 [{self.camera_id}]: VideoBuffer已关闭，最后一个片段 '{os.path.basename(self.current_segment_path)}' 已保存。")
            except Exception as e: print(f"错误 [VideoBufferSC171]: 关闭VideoWriter时发生异常: {e}")
        self.current_video_writer = None

# --- 模块级测试代码 (保持不变) ---
if __name__ == '__main__':
    # ... (测试代码保持不变) ...
    print("--- VideoBufferSC171 模块测试 ---")
    test_cam_id = "cam_buffer_test_01"; test_duration_sec = 5; test_output_root = VIDEO_CACHE_DIR
    mock_fps = 25.0; mock_frame_w, mock_frame_h = 640, 480
    print(f"测试参数: CamID='{test_cam_id}', Duration={test_duration_sec}s, FPS={mock_fps}, Size={mock_frame_w}x{mock_frame_h}")
    print(f"视频片段将保存到 '{os.path.join(test_output_root, test_cam_id)}' 子目录。")
    buffer = VideoBufferSC171( camera_id=test_cam_id, cache_duration_seconds=test_duration_sec, output_directory_root=test_output_root, fps=mock_fps, frame_width=mock_frame_w, frame_height=mock_frame_h, fourcc_str='mp4v' )
    if not buffer.is_writer_open: print("VideoWriter未能成功初始化。测试中止。")
    else:
        print("\n开始模拟视频流并添加到缓冲区...")
        dummy_frame = np.zeros((mock_frame_h, mock_frame_w, 3), dtype=np.uint8)
        total_frames_to_simulate = int(mock_fps * test_duration_sec * 2.5)
        frames_written_this_segment = 0; segments_completed = 0
        try:
            for i in range(total_frames_to_simulate):
                cv2.putText(dummy_frame, f"Frame: {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                completed_segment = buffer.add_frame(dummy_frame)
                if completed_segment: segments_completed += 1; print(f"  测试：第 {segments_completed} 个片段已完成并返回路径: {os.path.basename(completed_segment)}"); frames_written_this_segment = 0
                else: frames_written_this_segment += 1
                time.sleep(1.0 / mock_fps) 
                if i % int(mock_fps) == 0 : print(f"  模拟中... 当前已处理 {i+1} 帧。当前片段已写入 {frames_written_this_segment} 帧。")
        except KeyboardInterrupt: print("\n测试被用户中断。")
        finally: buffer.close()
        print(f"\n--- 模拟结束 ---"); print(f"共模拟了 {total_frames_to_simulate} 帧。"); print(f"期望完成约 {total_frames_to_simulate / (mock_fps * test_duration_sec):.1f} 个片段。"); print(f"实际完成了 {segments_completed} 个片段。")
        print(f"请检查目录 '{os.path.join(test_output_root, test_cam_id)}' 中的视频文件。"); print("尝试用视频播放器打开它们，确认内容和时长。")
    print("\n--- VideoBufferSC171 模块测试完成 ---")