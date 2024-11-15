import numpy as np

import tesserocr  # bugfix, tesserocr will not load and crash program if imported after decord
from decord import VideoReader
from typing import Optional

from args import AnalyserArgs
from config import Config
from utils.cv import crop2bbox

from .base import FpsCapture
from .regions import Regions


class VideoFileCapture(FpsCapture):
    def __init__(self, args: AnalyserArgs, config: Config) -> None:
        super(VideoFileCapture, self).__init__(args, config)
        
        self.vr = self.__load_video(config)
        self.__fps = self.vr.get_avg_fps()

        self.total_frames = len(self.vr)
        self.__frame_idx = self._get_start()
        if self.__frame_idx >= self.total_frames:
            raise ValueError("Start time > video duration")

    @property
    def frame_idx(self) -> int:
        return self.__frame_idx

    @property
    def fps(self) -> float:
        return self.__fps
    
    def __load_video(self, config: Config) -> VideoReader:
        if config.capture.file is None:
            raise ValueError("Invalid Config! Video file is not provided! Please add videofile path to capture.file")

        video_path = config.capture.file
        try:
            return VideoReader(str(video_path), num_threads=1)
        except Exception as e:
            raise ValueError(f"Video Capture Error: Unable to open video file {video_path}\n{e}")

    def __next_frame(self, frame_interval: int) -> Optional[np.ndarray]:
        self.__frame_idx = self.__frame_idx + frame_interval
        if self.__frame_idx >= self.total_frames:
            return None
        
        return self.vr[self.__frame_idx].asnumpy()

    def next(self, dt: float, *_, **__) -> Optional[Regions]:
        frame_interval = max(int(round(self.fps * dt)), 1)
        frame = self.__next_frame(frame_interval)
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        return self._set_regions(frame)
    
    def stop(self) -> None:
        ...
