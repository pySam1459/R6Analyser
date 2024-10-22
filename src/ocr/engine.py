import cv2
import numpy as np
from dataclasses import dataclass as odataclass
from re import match
from tesserocr import PSM
from typing import Optional, Callable, cast

from assets import Assets
from settings import Settings
from utils import Timestamp, resize_height, filter_none, squeeze_image, clip_around
from utils.constants import *

from .base import BaseOCREngine
from .utils import OCReadMode, OCRParams, get_timer_redperc


__all__ = [
    "OCREngine",
    "OCRLineResult"
]


@odataclass
class Segment:
    image: np.ndarray
    rect:  list[int]


@odataclass
class KfLineSegments:
    left:   Optional[Segment]
    right:  Segment
    middle: Segment


def segment(image: np.ndarray, params: OCRParams) -> Optional[KfLineSegments]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = get_black_section(gray_image, params)
    if rect is None or rect[2]*rect[3] < gray_image.size * params.seg_min_area:
        return None

    x,y,w,h = rect
    left_idx = get_left_xcoord(image[y:y+h,:x], params)
    left_idx = int(left_idx)
    if left_idx < 0:
        return None

    width = gray_image.shape[1]
    return KfLineSegments(left=get_segment(image, [left_idx, y, x-left_idx, h]),
                          right=get_segment(image, [x+w, y, width-x-w, h]),
                          middle=Segment(gray_image[y:y+h,x:x+w], [x,y,w,h]))


def get_black_section(gray_image: np.ndarray, params: OCRParams) -> Optional[list[int]]:
    height = gray_image.shape[0]
    black_th = int(params.seg_black_th)

    horz_dist = get_distribution(gray_image, black_th, params.seg_dist_th)
    lr = get_leftright(horz_dist, (params.bs_left_clip, params.bs_right_clip))
    if lr is None:
        return None
    
    x, w = lr[0], lr[1]-lr[0]
    
    ## flip image to clip top/bottom after left/right clip
    vertical_gimg = gray_image[:,lr[0]:lr[1]].T
    vert_dist = get_distribution(vertical_gimg, black_th, params.seg_dist_vert_th)
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

def max_pool1d(data: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    # Calculate the number of output elements
    output_length = ((len(data) - pool_size) // stride) + 1

    # Use stride tricks to create a 2D array where each row is a window
    strided_shape = (output_length, pool_size)
    strides = (stride * data.strides[0], data.strides[0])
    windows = np.lib.stride_tricks.as_strided(data, shape=strided_shape, strides=strides)

    # Perform max pooling across each row (axis 1)
    return np.max(windows, axis=1)

def get_left_xcoord(image: np.ndarray, params: OCRParams) -> int:
    w = image.shape[1]
    newgray_image = np.mean(image, axis=(0, 2))
    messy_mask = sliding_window_diffmean(newgray_image, params.xc_window) > params.xc_messy_th
    maxpool_mask = max_pool1d(messy_mask, 3, 3)

    rclip = int(w * (1-params.xc_right_clip))
    indices = np.where(maxpool_mask[:rclip])[0]
    if len(indices) == 0:
        return -1

    return indices[-1] * 1.5 ## 1.5 to rescale after sliding window

def get_segment(image: np.ndarray, rect: list[int]) -> Segment:
    x,y,w,h = rect
    return Segment(image[y:y+h,x:x+w], rect)


def get_headshot_match(middle_segment: np.ndarray,
                       hs_asset: np.ndarray,
                       params: OCRParams):
    """Uses template matching to determine if a kf line has a headshot"""

    hs_asset = resize_height(hs_asset, middle_segment.shape[0])
    result = cv2.matchTemplate(middle_segment, hs_asset, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    hs_asset_wide = cv2.resize(hs_asset, None, fx=params.hs_wide_sf, fy=1.0, interpolation=cv2.INTER_LINEAR)
    result_wide = cv2.matchTemplate(middle_segment, hs_asset_wide, cv2.TM_CCOEFF_NORMED)
    _, max_val_wide, _, _ = cv2.minMaxLoc(result_wide)

    max_val = max(max_val, max_val_wide)
    return max_val


@odataclass
class OCRLineResult:
    left:     Optional[str]
    right:    str
    headshot: bool

    left_image:   Optional[np.ndarray] = None
    right_image:  Optional[np.ndarray] = None
    middle_image: Optional[np.ndarray] = None


class OCREngine(BaseOCREngine):
    def __init__(self, params: OCRParams, settings: Settings, assets: Assets, _debug_print: Optional[Callable] = None) -> None:
        super(OCREngine, self).__init__(settings, _debug_print)
        self.__assets = assets
        self.params = params

    def read_score(self, score: np.ndarray) -> Optional[str]:
        score = clip_around(score, self.params.sl_clip_around)
        score = cv2.resize(score,
                           None,
                           fx=self.params.sl_scalex,
                           fy=self.params.sl_scaley,
                           interpolation=cv2.INTER_CUBIC)
        text = self.readtext(score, PSM.SINGLE_BLOCK, DIGITS)
        if match(SCORELINE_PATTERN, text):
            return text
        
        texts = self.readtext(np.hstack([score]*6), PSM.SINGLE_LINE, DIGITS)
        if len(texts) > 0 and texts.count(texts[0]) >= 2:
            return texts[0]

        return None

    def read_kfline(self, kfline_img: np.ndarray, charlist = IGN_CHARLIST) -> Optional[OCRLineResult]:
        """Reads a single killfeed line and returns an OCRLineResult instance"""
        segment_output = segment(kfline_img, self.params)
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

        middle_image = segment_output.middle.image
        headshot_match = get_headshot_match(middle_image, self.__assets["headshot_mask"], self.params)

        is_headshot = headshot_match >= self.params.hs_th
        self.debug_print("headshot_match", headshot_match)

        if (left_text is not None and len(left_text) <= 2) or len(right_text) <= 2:
            return None

        return OCRLineResult(left_text, right_text, is_headshot,
                             left_image, right_image, middle_image)


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
        self.debug_print("red_percentage", f"{red_perc=}")

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
