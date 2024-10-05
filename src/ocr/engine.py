import cv2
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, OEM
from typing import Self, Sequence, Optional, Callable, overload

from assets import Assets
from settings import Settings
from utils import BBox_t
from utils.constants import IGN_CHARLIST


__all__ = [
    "OCREngine",
    "OCReadMode",
    "OCRLineResult"
]

@dataclass
class OCRLineResult:
    left:     Optional[str]
    right:    str
    headshot: bool


@dataclass
class Segment:
    image: np.ndarray
    rect:  list[int]


@dataclass
class KfLineSegments:
    left:   Optional[Segment]
    middle: Segment
    right:  Segment


MIN_SEGMENT_SIZE = 0.1

LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 50])


def segment(image: np.ndarray) -> Optional[KfLineSegments]:
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    black_mask = cv2.inRange(hsv_img, LOWER_BLACK, UPPER_BLACK)
    rects = get_rects(black_mask)
    if len(rects) == 0:
        return None

    rect = rects[-1]
    x,y,w,h = rect

    return KfLineSegments(left=get_segment(image, [0, y, x, h]),
                          middle=get_segment(image, rect),
                          right=get_segment(image, [x+w, y, image.shape[1]-x-w, h]))


def filter_condition(rect: cv2.typing.Rect, mask: cv2.typing.MatLike) -> bool:
    if rect[2] * rect[3] < mask.size * 0.025:
        return False
    return True

def join_rects(rects: list[list[int]], width: int) -> list[list[int]]:
    out = []
    r = rects[0]
    for i in range(1, len(rects)):
        r2 = rects[i]
        xdiff = r[0]+r[2] - r2[0]
        if xdiff < width * 0.95:
            r[2] = r2[0]+r2[2] - r[0]
        else:
            out.append(r)
            r = r2
    out.append(r)
    return out

def get_rects(mask: cv2.typing.MatLike) -> list[list[int]]:
    contours_black, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours_black]
    rects = [r for r in rects if filter_condition(r, mask)]

    rects = [[r[0], 0, r[2], mask.shape[0]] for r in rects]
    if len(rects) < 2:
        return rects

    rects = list(sorted(rects, key=lambda r: r[0]))
    return join_rects(rects, mask.shape[1])

def get_segment(image: np.ndarray, rect: list[int]) -> Segment:
    x,y,w,h = rect
    return Segment(image[y:y+h,x:x+w], rect)


class OCReadMode(IntEnum):
    LINE = PSM.SINGLE_LINE
    WORD = PSM.SINGLE_WORD


class OCREngine:
    def __init__(self, settings: Settings, assets: Assets) -> None:
        self.__assets = assets

        self.api = PyTessBaseAPI(path=str(settings.tessdata), oem=OEM.LSTM_ONLY) # type: ignore


    def read_kfline(self, kfline_img: np.ndarray, charlist = IGN_CHARLIST) -> Optional[OCRLineResult]:
        """Reads a single killfeed line and returns an OCRLineResult instance"""
        segment_output = segment(kfline_img)
        if segment_output is None:
            return None

        left_text  = (segment_output.left and
                        self.readtext(segment_output.left.image, OCReadMode.WORD, charlist))

        right_text = self.readtext(segment_output.right.image, OCReadMode.WORD, charlist)

        middle_image = cv2.cvtColor(segment_output.middle.image, cv2.COLOR_RGB2GRAY)
        headshot = is_headshot(middle_image, self.__assets.headshot) is not None

        return OCRLineResult(left_text, right_text, headshot)


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

    def __readtext(self, image: Image.Image | np.ndarray) -> str:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self.api.SetImage(image)
        return self.api.GetUTF8Text()[:-1] #, self.api.MeanTextConf()


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