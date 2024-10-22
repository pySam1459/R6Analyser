from numpy import asarray
from pyautogui import screenshot
from typing import Type, TypeVar

from config import Config

from .base import TimeCapture
from .utils import InPersonRegions, SpectatorRegions


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class ScreenshotCapture(TimeCapture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(ScreenshotCapture, self).__init__(config, region_model)

    def next(self) -> T:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        sshot_img = screenshot(allScreens=True, region=self._bboxes.max_bounds)
        image = asarray(sshot_img)
        return self._set_regions(image)
    
    def stop(self) -> None:
        ...
