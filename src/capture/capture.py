import cv2
import numpy as np
from abc import ABC, abstractmethod
from time import time
from pyautogui import screenshot
from typing import Optional, Generic, TypeVar, Type, Generator

from config import Config
from utils.enums import CaptureMode
from .utils import RegionBBoxes, SpectatorRegions, InPersonRegions, crop_bboxes


__all__ = ["Capture"]


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class Capture(Generic[T], ABC):
    def __init__(self, config: Config, region_model: Type[T]):
        self.__region_model = region_model

        region_dump = config.capture.regions.model_dump(exclude_none=True)
        self._region_bboxes = RegionBBoxes.model_validate(region_dump)

        self.__images: Optional[T] = None

    @abstractmethod
    def next(self) -> Optional[T]:
        ...

    @abstractmethod
    def get_time(self) -> float:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    def get(self) -> Optional[T]:
        return self.__images

    def get_region(self, name: str) -> Optional[np.ndarray]:
        if self.__images is not None and hasattr(self.__images, name):
            return getattr(self.__images, name)
        return None
    
    def _set_regions(self, image: np.ndarray, bboxes: RegionBBoxes) -> T:
        cropped_regions = crop_bboxes(image, bboxes)
        self.__images = self.__region_model.model_validate(cropped_regions)
        return self.__images


class ScreenshotCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(ScreenshotCapture, self).__init__(config, region_model)

    def next(self, bboxes: RegionBBoxes) -> T:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        sshot_img = screenshot(allScreens=True, region=bboxes.max_bounds)
        image = np.asarray(sshot_img)
        return self._set_regions(image, bboxes)
    
    def stop(self) -> None:
        ...
    
    def get_time(self) -> float:
        return time()


class VideoFileCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(VideoFileCapture, self).__init__(config, region_model)
        
        assert config.capture.file is not None, "Invalid Config! Video file is not provided! Please add videofile path to capture.file"
        self.video_path = config.capture.file
        self.frame_period = config.capture.period
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Video Capture Error: Unable to open video file {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise ValueError("Video Capture Error: Invalid FPS value. Cannot proceed.")

        self.frame_interval = max(int(round(self.fps * self.frame_period)), 1)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def __next_frame(self) -> np.ndarray | None:
        if self.current_frame >= self.total_frames:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()

        if not ret:
            return None

        self.current_frame += self.frame_interval
        return frame

    def next(self, bboxes: RegionBBoxes) -> Optional[T]:
        frame = self.__next_frame()
        if frame is None:
            return None

        return self._set_regions(frame, bboxes)

    def get_time(self) -> float:
        return self.current_frame / self.fps
    
    def stop(self) -> None:
        self.cap.release()


class StreamCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(StreamCapture, self).__init__(config, region_model)
        
        raise NotImplementedError("Steam Capturing is not implemented yet!")
    
    def next(self, bboxes: RegionBBoxes) -> T:
        ...
    
    def get_time(self) -> float:
        ...
    
    def stop(self) -> None:
        ...


class YoutubeCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(YoutubeCapture, self).__init__(config, region_model)
        
        raise NotImplementedError("Youtube Capturing is not implemented yet!")
    
    def next(self, bboxes: RegionBBoxes) -> T:
        ...
    
    def get_time(self) -> float:
        ...
    
    def stop(self) -> None:
        ...


def create_capture(config: Config, region_model: Type[T]) -> Capture[T]:
    match config.capture.mode:
        case CaptureMode.SCREENSHOT:
            return ScreenshotCapture[T](config, region_model)
        case CaptureMode.VIDEOFILE:
            return VideoFileCapture[T](config, region_model)
        case CaptureMode.STREAM:
            return StreamCapture[T](config, region_model)
        case CaptureMode.YOUTUBE:
            return YoutubeCapture[T](config, region_model)
