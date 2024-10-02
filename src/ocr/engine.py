import cv2
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, OEM
from typing import Sequence, Optional, Callable, overload

from settings import Settings
from utils import BBox_t


__all__ = [
    "OCREngine",
    "OCReadMode",
    "OCRLineResult"
]

MatLike = cv2.typing.MatLike

## TODO : remember headshot
@dataclass
class OCRLineResult:
    left:     Optional[str]
    right:    str
    headshot: bool


lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])


@dataclass
class Segment:
    image: np.ndarray
    rect:  list[int]


class OCReadMode(IntEnum):
    LINE = PSM.SINGLE_LINE
    WORD = PSM.SINGLE_WORD


class OCREngine:
    def __init__(self, settings: Settings) -> None:
        self.api = PyTessBaseAPI(path=settings.tessdata, # type: ignore
                                 oem=OEM.LSTM_ONLY)      # type: ignore
    
    def read_kfline(self, kfline_img: np.ndarray) -> OCRLineResult:
        hsv_image = cv2.cvtColor(kfline_img, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = red_mask1 | red_mask2 # type: ignore
        contours_blue = self._find_contours(blue_mask)
        contours_red  = self._find_contours(red_mask)

    def _find_contours(self, mask: MatLike) -> Sequence[MatLike]:
        return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    def _filter_contours(self, ) -> Sequence[MatLike]:
        ...

    def _crop_segments(self, image, contours) -> list[Segment]:
        return [self._get_segment(image, contour) for contour in contours]
    
    def _get_segment(self, image: MatLike, contour: MatLike) -> Segment:
        x, y, w, h = cv2.boundingRect(contour)
        return Segment(image[y:y+h, x:x+w], [x,y,w,h])

    def __readtext(self, image: Image.Image | np.ndarray) -> str:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.api.SetImage(image)
        return self.api.GetUTF8Text()[:-1] #, self.api.MeanTextConf()

    @overload
    def readtext(self, image: np.ndarray, read_mode: OCReadMode, charlist: str) -> str: ...
    @overload
    def readtext(self, image: Sequence[np.ndarray], read_mode: OCReadMode, charlist: str) -> list[str]: ...

    def readtext(self, image: np.ndarray | Sequence[np.ndarray], read_mode: OCReadMode, charlist: str) -> str | list[str]:
        self.api.SetVariable("tessedit_pageseg_mode", str(read_mode))
        self.api.SetVariable("tessedit_char_whitelist", charlist)

        if isinstance(image, Sequence):
            return [self.__readtext(_img) for _img in image]
        elif isinstance(image, np.ndarray):
            return self.__readtext(image)


def is_headshot(image_line: np.ndarray, hs_asset: np.ndarray,
                threshold: float = 0.65,
                _debug_print: Optional[Callable] = None) -> Optional[BBox_t]:
    """Uses template matching to determine if a kf line has a headshot"""
    result = cv2.matchTemplate(image_line, hs_asset, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, best_pt = cv2.minMaxLoc(result)

    if _debug_print is not None:
        _debug_print("headshot_perc", max_val)

    if max_val < threshold:
        return None

    h,w = hs_asset.shape
    return (best_pt[0], best_pt[1], w, h)