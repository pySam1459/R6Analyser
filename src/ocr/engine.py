import cv2
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from re import match
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, OEM
from typing import Sequence, Optional, Callable, overload, cast

from assets import Assets
from settings import Settings
from utils import Timestamp, filter_none
from utils.constants import BOMB_COUNTDOWN_RT, IGN_CHARLIST, OCR_TIMER_THRESHOLD, TIMER_CHARLIST

from .utils import get_timer_redperc


__all__ = [
    "OCREngine",
    "OCReadMode",
    "OCRLineResult"
]


@dataclass
class Segment:
    image: np.ndarray
    rect:  list[int]


@dataclass
class KfLineSegments:
    left:   Optional[Segment]
    middle: Segment
    right:  Segment


SEGMENT_PARAMS = {
    "black_threshold": 64,
    "template_width": 0.15,
    "template_buffer": 4,
    "norm_threshold": 96/255,
    "cumsum_count_threshold": 5,
    "min_seg_perc": 0.025
}


def segment(image: np.ndarray) -> Optional[KfLineSegments]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask: np.ndarray = (gray_image < SEGMENT_PARAMS["black_threshold"]).astype(np.uint8) * 255
    black_mask = trim_mask(black_mask)

    rects = get_rects(black_mask)
    if len(rects) == 0:
        return None

    rect = sorted(rects, key=lambda r: r[0]+r[2], reverse=True)[0]
    x,y,w,h = rect

    left_idx = get_left_xcoord(image, [0, y, x, h])

    return KfLineSegments(left=get_segment(image, [left_idx, y, x-left_idx, h]),
                          middle=get_segment(black_mask, rect),
                          right=get_segment(image, [x+w, y, image.shape[1]-x-w, h]))

def trim_mask(black_mask: np.ndarray) -> np.ndarray:
    h,w = black_mask.shape

    avg = np.mean(black_mask, axis=0)
    ### TODO FINISH


def filter_condition(rect: cv2.typing.Rect, mask: cv2.typing.MatLike) -> bool:
    if rect[2] * rect[3] < mask.size * SEGMENT_PARAMS["min_seg_perc"]:
        return False
    return True

def get_rects(mask: cv2.typing.MatLike) -> list[list[int]]:
    contours_black, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours_black]
    rects = [r for r in rects if filter_condition(r, mask)]

    rects = [[r[0], 0, r[2], mask.shape[0]] for r in rects]
    return rects

def get_left_xcoord(image: np.ndarray, rect: list[int]) -> int:
    _,y,w,h = rect
    tw = int(w * SEGMENT_PARAMS["template_width"])
    buf = SEGMENT_PARAMS["template_buffer"]
    x, xw = w-tw-buf, w-buf
    template = image[y:y+h,x:xw]

    cmp_result = cv2.matchTemplate(image[y:y+h,:w], template, cv2.TM_CCOEFF_NORMED)
    cmp_result = cast(np.ndarray, cmp_result)

    min_, max_ = np.min(cmp_result), np.max(cmp_result)
    norm_data: np.ndarray = (cmp_result - min_) / (max_ - min_)

    inc_data = np.cumsum(norm_data > SEGMENT_PARAMS["norm_threshold"])
    return np.where(inc_data == SEGMENT_PARAMS["cumsum_count_threshold"])[0][0]

def get_segment(image: np.ndarray, rect: list[int]) -> Segment:
    x,y,w,h = rect
    return Segment(image[y:y+h,x:x+w], rect)


def is_headshot(black_segment: np.ndarray, hs_asset: np.ndarray,
                threshold: float = 0.65,
                _debug_print: Optional[Callable] = None) -> bool:
    """Uses template matching to determine if a kf line has a headshot"""
    result = cv2.matchTemplate(black_segment, hs_asset, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    if _debug_print is not None:
        _debug_print("headshot_perc", max_val)

    return max_val >= threshold


class OCReadMode(IntEnum):
    LINE = PSM.SINGLE_LINE
    WORD = PSM.SINGLE_WORD


@dataclass
class OCRLineResult:
    left:     Optional[str]
    right:    str
    headshot: bool

    left_image:   Optional[np.ndarray] = None
    middle_image: Optional[np.ndarray] = None
    right_image:  Optional[np.ndarray] = None


class OCREngine:
    def __init__(self, settings: Settings, assets: Assets, _debug_print: Optional[Callable] = None) -> None:
        self.__assets = assets
        self._debug_print = _debug_print

        self.api = PyTessBaseAPI(path=str(settings.tessdata), oem=OEM.LSTM_ONLY) # type: ignore

    def read_kfline(self, kfline_img: np.ndarray, charlist = IGN_CHARLIST) -> Optional[OCRLineResult]:
        """Reads a single killfeed line and returns an OCRLineResult instance"""
        segment_output = segment(kfline_img)
        if segment_output is None:
            return None

        left_image = segment_output.left and segment_output.left.image
        left_text = segment_output.left and self.readtext(~segment_output.left.image, OCReadMode.WORD, charlist)

        right_image = segment_output.right.image
        right_text = self.readtext(~right_image, OCReadMode.WORD, charlist)

        middle_mask = segment_output.middle.image
        self.__assets.resize_height("headshot_mask", middle_mask.shape[0])
        is_headshot_ = is_headshot(middle_mask, self.__assets["headshot_mask"], _debug_print=self._debug_print)

        return OCRLineResult(left_text, right_text, is_headshot_,
                             left_image, middle_mask, right_image)


    def read_timer(self, timer_img: np.ndarray) -> tuple[Optional[Timestamp], bool]:
        """Returns (timer: Optional[Timestamp], is_bomb_countdown: bool)"""
        if self.get_is_bomb_countdown(timer_img):
            return None, True

        not_timer_img = ~cv2.cvtColor(timer_img, cv2.COLOR_RGB2GRAY) # type: ignore
        denoised_image = cv2.medianBlur(not_timer_img, 3)
        th_image = denoised_image > OCR_TIMER_THRESHOLD # type: ignore

        results = self.readtext([denoised_image, th_image], OCReadMode.LINE, TIMER_CHARLIST)
        return self.__pick_timer_result(results), False

    def get_is_bomb_countdown(self, timer_img: np.ndarray) -> bool:
        red_perc = get_timer_redperc(timer_img)
        if self._debug_print is not None:
            self._debug_print("red_percentage", f"{red_perc=}")
        return red_perc > BOMB_COUNTDOWN_RT

    def __pick_timer_result(self, results: list[str]) -> Optional[Timestamp]:
        converted_results = [self.__convert_raw_to_ts(res) for res in results]
        filtered_results = filter_none(converted_results)
        if len(filtered_results) == 0:
            return None

        ## TODO: if multiple, select one with abs time diff closest to last timer result
        return filtered_results[0]

    def __convert_raw_to_ts(self, raw_result: str) -> Optional[Timestamp]:
        timer_match = match(r"(\d?\d)([:\.])(\d\d)", raw_result)
        if timer_match is None:
            return None

        if timer_match.group(2) == ":":
            return Timestamp(
                minutes=int(timer_match.group(1)),
                seconds=int(timer_match.group(3))
            )
        elif timer_match.group(2) == ".":
            return Timestamp(
                minutes=0,
                seconds=int(timer_match.group(1))
            )
        ## TODO: could improve this? maybe assume '220' == '2m 20s', infer from previous timer?

        return None

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

