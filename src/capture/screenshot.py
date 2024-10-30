from numpy import asarray
from pyautogui import screenshot

from config import Config

from .base import TimeCapture
from .regions import Regions


class ScreenshotCapture(TimeCapture):
    def __init__(self, config: Config) -> None:
        super(ScreenshotCapture, self).__init__(config)

    def next(self) -> Regions:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        sshot_img = screenshot(allScreens=True, region=self._bboxes.max_bounds)
        image = asarray(sshot_img)
        return self._set_regions(image)
    
    def stop(self) -> None:
        ...
