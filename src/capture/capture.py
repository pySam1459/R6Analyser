import numpy as np
from abc import ABC, abstractmethod
from time import time
from pyautogui import screenshot
from typing import Optional, Generic, TypeVar, Type

from config import Config
from utils.enums import CaptureMode
from .utils import RegionBBoxes, SpectatorRegions, InPersonRegions, crop_bboxes


__all__ = ["Capture"]


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class Capture(Generic[T], ABC):
    def __init__(self, config: Config, region_model: Type[T]):
        self._region_model = region_model

        region_dump = config.capture.regions.model_dump(exclude_none=True)
        self._region_bboxes = RegionBBoxes.model_validate(region_dump)

        self._images: Optional[T] = None

    @abstractmethod
    def next(self) -> T:
        ...

    @abstractmethod
    def get_time(self) -> float:
        ...

    def get(self) -> Optional[T]:
        return self._images

    def get_region(self, name: str) -> Optional[np.ndarray]:
        if self._images is not None and hasattr(self._images, name):
            return getattr(self._images, name)
        return None


class ScreenshotCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(ScreenshotCapture, self).__init__(config, region_model)

    def next(self, bboxes: RegionBBoxes) -> T:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        sshot_img = screenshot(allScreens=True, region=bboxes.max_bounds)
        image = np.asarray(sshot_img)

        cropped_regions = crop_bboxes(image, bboxes)
        self._images = self._region_model.model_validate(cropped_regions)
        return self._images
    
    def get_time(self) -> float:
        return time()


class VideoFileCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(VideoFileCapture, self).__init__(config, region_model)
        
        raise NotImplementedError("Video File Capturing is not implemented yet!")
    
    def next(self, bboxes: RegionBBoxes) -> T:
        ...
    
    def get_time(self) -> float:
        ...


class StreamCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(StreamCapture, self).__init__(config, region_model)
        
        raise NotImplementedError("Steam Capturing is not implemented yet!")
    
    def next(self, bboxes: RegionBBoxes) -> T:
        ...
    
    def get_time(self) -> float:
        ...


class YoutubeCapture(Capture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(YoutubeCapture, self).__init__(config, region_model)
        
        raise NotImplementedError("Youtube Capturing is not implemented yet!")
    
    def next(self, bboxes: RegionBBoxes) -> T:
        ...
    
    def get_time(self) -> float:
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
