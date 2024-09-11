import numpy as np
from abc import ABC, abstractmethod
from pyautogui import screenshot
from typing import Optional

from config import Config
from utils.enums import CaptureMode
from .utils import RegionBBoxes, RegionImages, SpectatorRegions, InPersonRegions, ImageRegions_t


__all__ = ["Capture", "create_capture"]


class Capture(ABC):
    def __init__(self, config: Config):
        self._config = config
        self._region_images: Optional[ImageRegions_t] = None

    @abstractmethod
    def next(self, region_bboxes: RegionBBoxes) -> ImageRegions_t:
        ...

    def get(self) -> Optional[ImageRegions_t]:
        return self._region_images

    def get_region(self, name: str) -> Optional[np.ndarray]:
        if self._region_images is not None and hasattr(self._region_images, name):
            return getattr(self._region_images, name)
        return None


class ScreenshotCapture(Capture):
    def __init__(self, config: Config) -> None:
        super(ScreenshotCapture, self).__init__(config)

    def next(self, region_bboxes: RegionBBoxes) -> ImageRegions_t:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        sshot_img = screenshot(allScreens=True, region=region_bboxes.max_bounds)
        numpy_img = np.asarray(sshot_img)

        _images = RegionImages(image=numpy_img, region_bboxes=region_bboxes)
        if self._config.spectator:
            self._region_images = SpectatorRegions.model_validate(_images.model_dump(exclude_none=True))
        else:
            self._region_images = InPersonRegions.model_validate(_images.model_dump(exclude_none=True))
        
        return self._region_images


class VideoFileCapture(Capture):
    def __init__(self, config: Config) -> None:
        super(VideoFileCapture, self).__init__(config)
        
        raise NotImplementedError("Video File Capturing is not implemented yet!")
    
    def next(self, region_bboxes: RegionBBoxes) -> RegionImages:
        ...


def create_capture(config: Config) -> Capture:
    match config.capture.mode:
        case CaptureMode.SCREENSHOT:
            return ScreenshotCapture(config)
        case CaptureMode.VIDEOFILE:
            return VideoFileCapture(config)
