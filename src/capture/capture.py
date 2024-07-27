import numpy as np
from pyautogui import screenshot
from PIL.Image import Image
from abc import ABC, abstractmethod
from typing import TypeAlias

from config import Config
from .utils import CaptureMode


__all__ = [
    "Regions_t",
    "CaptureMode",
    "Capture"
]


Regions_t: TypeAlias = dict[str, np.ndarray]
class Capture(ABC):
    def __init__(self, config: Config):
        self._config = config
        self._regions = {}

    @staticmethod
    def new(config: Config) -> 'Capture':
        match config.capture.mode:
            case CaptureMode.SCREENSHOT:
                return ScreenshotCapture(config)
            case CaptureMode.VIDEOFILE:
                return VideoFileCapture(config)

        raise Exception("Invalid config capture mode")

    @abstractmethod
    def next(self, keys: list[str]) -> Regions_t:
        ...

    def get(self) -> Regions_t:
        return self._regions

    def get_region(self, key: str) -> np.ndarray:
        assert key in self._regions, f"key: {key} not a valid region"
        return self._regions.get(key, None)


class ScreenshotCapture(Capture):
    def __init__(self, config: Config) -> None:
        super(ScreenshotCapture, self).__init__(config)

    def next(self, keys: list[str]) -> Regions_t:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        sshot_img = screenshot(allScreens=True)
        self._regions = {
            region_key: self.__select_region(sshot_img, region_key)
            for region_key in keys
        }
        return self._regions

    @staticmethod
    def convert_region(region: list[int]) -> tuple[int,int,int,int]:
        """Converts (X,Y,W,H) -> (Left,Top,Right,Bottom)"""
        x, y, w, h = region
        return (x, y, x + w, y + h)

    def __select_region(self, sshot: Image, region_key: str) -> np.ndarray:
        region_box = self._config.capture.regions[region_key]
        img = sshot.crop(ScreenshotCapture.convert_region(region_box))
        return np.array(img)


class VideoFileCapture(Capture):
    def __init__(self, config: Config) -> None:
        super(VideoFileCapture, self).__init__(config)
        
        raise NotImplementedError("Video File Capturing is not implemented yet!")
    
    def next(self, keys: list[str]) -> Regions_t:
        ...
