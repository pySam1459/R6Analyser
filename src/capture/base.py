import numpy as np
from abc import ABC, abstractmethod
from time import time
from typing import Optional, Generic, TypeVar, Type

from config import Config
from utils.enums import CaptureTimeType

from .utils import RegionBBoxes, SpectatorRegions, InPersonRegions, crop_bboxes


__all__ = [
    "Capture",
    "TimeCapture",
    "FpsCapture"
]


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class Capture(Generic[T], ABC):
    def __init__(self, config: Config, region_model: Type[T]):
        region_dump = config.capture.regions.model_dump(exclude_none=True)
        self._bboxes = RegionBBoxes.model_validate(region_dump)

        self.__region_model = region_model
        self.__images: Optional[T] = None
    
    @property
    @abstractmethod
    def time_type(self) -> CaptureTimeType:
        ...

    @abstractmethod
    def get_time(self) -> float:
        ...

    @abstractmethod
    def next(self, dt: Optional[float] = None, jump = False) -> Optional[T]:
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
    
    def _set_regions(self, image: np.ndarray) -> T:
        cropped_regions = crop_bboxes(image, self._bboxes)
        self.__images = self.__region_model.model_validate(cropped_regions)
        return self.__images


class TimeCapture(Capture[T], ABC):
    @property
    def time_type(self) -> CaptureTimeType:
        return CaptureTimeType.TIME
    
    @abstractmethod
    def next(self) -> Optional[T]:
        ...

    def get_time(self) -> float:
        return time()


class FpsCapture(Capture[T], ABC):
    frame_idx: int
    fps: float

    @property
    def time_type(self) -> CaptureTimeType:
        return CaptureTimeType.FPS

    @abstractmethod
    def next(self, dt: Optional[float] = None, jump = False) -> Optional[T]:
        ...
    
    def get_time(self) -> float:
        return self.frame_idx / self.fps
