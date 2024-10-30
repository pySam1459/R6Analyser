import numpy as np
from abc import ABC, abstractmethod
from time import time
from typing import Optional

from config import Config
from utils.enums import CaptureTimeType

from .regions import RegionBBoxes, Regions


__all__ = [
    "Capture",
    "TimeCapture",
    "FpsCapture"
]


class Capture(ABC):
    def __init__(self, config: Config):
        region_dump = config.capture.regions.model_dump(exclude_none=True)
        self._bboxes = RegionBBoxes.model_validate(region_dump)

        self.__images: Optional[Regions] = None
    
    @property
    @abstractmethod
    def time_type(self) -> CaptureTimeType:
        ...

    @abstractmethod
    def get_time(self) -> float:
        ...

    @abstractmethod
    def next(self, dt: Optional[float] = None, jump = False) -> Optional[Regions]:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


    def get(self) -> Optional[Regions]:
        return self.__images

    def get_region(self, name: str) -> Optional[np.ndarray]:
        if self.__images is not None and hasattr(self.__images, name):
            return getattr(self.__images, name)
        return None
    
    def _set_regions(self, image: np.ndarray) -> Regions:
        cropped_regions = self._bboxes.crop(image)
        self.__images = Regions.model_validate(cropped_regions)
        return self.__images


class TimeCapture(Capture, ABC):
    @property
    def time_type(self) -> CaptureTimeType:
        return CaptureTimeType.TIME
    
    @abstractmethod
    def next(self) -> Optional[Regions]:
        ...

    def get_time(self) -> float:
        return time()


class FpsCapture(Capture, ABC):
    frame_idx: int
    fps: float

    @property
    def time_type(self) -> CaptureTimeType:
        return CaptureTimeType.FPS

    @abstractmethod
    def next(self, dt: Optional[float] = None, jump = False) -> Optional[Regions]:
        ...
    
    def get_time(self) -> float:
        return self.frame_idx / self.fps
