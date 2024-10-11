import cv2
import numpy as np
from dataclasses import dataclass
from re import match
from typing import Optional, Callable, cast

from assets import Assets
from settings import Settings
from utils import Timestamp, filter_none
from utils.constants import BOMB_COUNTDOWN_RT, IGN_CHARLIST, OCR_TIMER_THRESHOLD, TIMER_CHARLIST

from .base import BaseOCREngine
from .utils import OCReadMode, get_timer_redperc


__all__ = [
    "OCREngine",
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


OCR_PARAMS = {
    "black_threshold": 64,
    "dist_threshold": 0.75,
    "dist_vertical_threshold": 0.5,
    "left_clip": 0.4,
    "right_clip": 0.1,

    "min_seg_perc": 0.025
}


def segment(image: np.ndarray) -> Optional[KfLineSegments]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = get_black_section(gray_image)
    if rect is None or rect[2]*rect[3] < gray_image.size * OCR_PARAMS["min_seg_perc"]:
        return None

    x,y,w,h = rect
    left_idx = get_left_xcoord(image[y:y+h,:x], 10)#, [0, y, x, h])
    left_idx = int(left_idx)

    width = gray_image.shape[1]
    return KfLineSegments(left=get_segment(image, [left_idx, y, x-left_idx, h]),
                          middle=Segment(gray_image[y:y+h,x:x+w], [x,y,w,h]),
                          right=get_segment(image, [x+w, y, width-x-w, h]))


def get_black_section(gray_image: np.ndarray) -> Optional[list[int]]:
    height = gray_image.shape[0]
    black_th = OCR_PARAMS["black_threshold"]

    horz_dist = get_distribution(gray_image, black_th, OCR_PARAMS["dist_threshold"])
    lr = get_leftright(horz_dist, (OCR_PARAMS["left_clip"], OCR_PARAMS["right_clip"]))
    if lr is None:
        return None
    
    x, w = lr[0], lr[1]-lr[0]
    
    ## flip image to clip top/bottom after left/right clip
    vertical_gimg = gray_image[:,lr[0]:lr[1]].T
    vert_dist = get_distribution(vertical_gimg, black_th, OCR_PARAMS["dist_vertical_threshold"])
    tb = get_leftright(vert_dist)
    if tb is None:
        y, h = 0, height
    else:
        y, h = tb[0], min(height, tb[1]-tb[0]+1)
    
    return [x, y, w, h]

def get_distribution(gray_image: np.ndarray, black_th: int, dist_th: float) -> np.ndarray:
    """Computes a vector containing the average black value for each column in the input image"""
    th_img = gray_image < black_th
    return np.mean(th_img, axis=0) > dist_th  ## (width,)

def get_leftright(out_dist: np.ndarray, clip: Optional[tuple[float, float]] = None) -> Optional[tuple[int,int]]:
    """Computes the left and right bounds of the black section of the image (post clipping)"""
    if clip is not None:
        w = out_dist.size
        x1, x2 = int(w * clip[0]), int(w * clip[1])
        out_dist = out_dist[x1:-x2]
    else:
        x1 = 0

    indicies = np.where(out_dist)[0] + x1
    if len(indicies) < 2:
        return None
    
    return indicies[0], indicies[-1]

def sliding_window_diffmean(data, window):
    diff = np.abs(np.diff(data, prepend=data[0]))
    kernel = np.ones(window)
    return np.convolve(diff, kernel, mode="same") / window

def max_pool1d(data, pool_size, stride):
    # Calculate the number of output elements
    output_length = ((len(data) - pool_size) // stride) + 1

    # Use stride tricks to create a 2D array where each row is a window
    strided_shape = (output_length, pool_size)
    strides = (stride * data.strides[0], data.strides[0])
    windows = np.lib.stride_tricks.as_strided(data, shape=strided_shape, strides=strides)

    # Perform max pooling across each row (axis 1)
    pooled = np.max(windows, axis=1)
    
    return pooled

def get_left_xcoord(image: np.ndarray, window: int) -> int:
    w = image.shape[1]
    newgray_image = np.mean(image, axis=(0, 2))
    messy_mask = sliding_window_diffmean(newgray_image, window) > 0.85
    maxpool_mask = max_pool1d(messy_mask, 3, 3)
    return np.where(maxpool_mask[:int(w*0.9)])[0][-1] * 1.5

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


@dataclass
class OCRLineResult:
    left:     Optional[str]
    right:    str
    headshot: bool

    left_image:   Optional[np.ndarray] = None
    middle_image: Optional[np.ndarray] = None
    right_image:  Optional[np.ndarray] = None


class OCREngine(BaseOCREngine):
    def __init__(self, settings: Settings, assets: Assets, _debug_print: Optional[Callable] = None) -> None:
        super(OCREngine, self).__init__(settings)
        self.__assets = assets
        self._debug_print = _debug_print

    def read_kfline(self, kfline_img: np.ndarray, charlist = IGN_CHARLIST) -> Optional[OCRLineResult]:
        """Reads a single killfeed line and returns an OCRLineResult instance"""
        segment_output = segment(kfline_img)
        if segment_output is None:
            return None

        if segment_output.left is None:
            left_image = None
            left_text = None
        else:
            left_image = segment_output.left.image
            left_text = self.readtext(~left_image, OCReadMode.WORD, charlist)

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
