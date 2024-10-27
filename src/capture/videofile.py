import numpy as np
from decord import VideoReader
from typing import Type, TypeVar, Optional

from config import Config
from utils import Timestamp

from .base import FpsCapture
from .utils import InPersonRegions, SpectatorRegions, crop2bbox


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class VideoFileCapture(FpsCapture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(VideoFileCapture, self).__init__(config, region_model)
        
        self.vr = self.__load_video(config)
        self.fps = self.vr.get_avg_fps()

        self.total_frames = len(self.vr)
        self.frame_idx = self.__offset(config.capture.offset)
    
    def __offset(self, offset: Optional[Timestamp]) -> int:
        if offset is None:
            return 0

        frame_idx = int(round(self.fps * offset.to_int()))
        if frame_idx >= self.total_frames:
            raise ValueError("Capture offset is >= total frames")
        
        return frame_idx
    
    def __load_video(self, config: Config) -> VideoReader:
        if config.capture.file is None:
            raise ValueError("Invalid Config! Video file is not provided! Please add videofile path to capture.file")

        video_path = config.capture.file
        try:
            return VideoReader(str(video_path), num_threads=1)
        except Exception as e:
            raise ValueError(f"Video Capture Error: Unable to open video file {video_path}\n{e}")

    def __next_frame(self, frame_interval: int) -> Optional[np.ndarray]:
        self.frame_idx = self.frame_idx + frame_interval
        if self.frame_idx >= self.total_frames:
            return None
        
        return self.vr[self.frame_idx].asnumpy()

    def next(self, dt: float, *_, **__) -> Optional[T]:
        frame_interval = max(int(round(self.fps * dt)), 1)
        frame = self.__next_frame(frame_interval)
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        return self._set_regions(frame)
    
    def stop(self) -> None:
        ...
