import cv2
import numpy as np
from typing import Type, TypeVar, Optional

from config import Config

from .base import FpsCapture
from .utils import InPersonRegions, SpectatorRegions, crop2bbox


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class VideoFileCapture(FpsCapture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(VideoFileCapture, self).__init__(config, region_model)
        
        self.cap = self.__load_video(config)
        self.fps = self.__get_fps(self.cap)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
    
    def __load_video(self, config: Config) -> cv2.VideoCapture:
        if config.capture.file is None:
            raise ValueError("Invalid Config! Video file is not provided! Please add videofile path to capture.file")

        video_path = config.capture.file
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Video Capture Error: Unable to open video file {video_path}")
        
        return cap

    def __get_fps(self, cap: cv2.VideoCapture) -> float:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Video Capture Error: Invalid FPS value. Cannot proceed.")

        return fps

    def __next_frame(self, frame_interval: int) -> Optional[np.ndarray]:
        new_frame = self.current_frame + frame_interval
        if new_frame >= self.total_frames:
            return None

        for _ in range(frame_interval):
            ret, frame = self.cap.read()
            if not ret:
                return None

        self.current_frame = new_frame 
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def next(self, dt: float) -> Optional[T]:
        frame_interval = max(int(round(self.fps * dt)), 1)
        frame = self.__next_frame(frame_interval)
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        return self._set_regions(frame)
    
    def stop(self) -> None:
        self.cap.release()
