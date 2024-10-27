import cv2
import numpy as np
from enum import IntEnum
from dataclasses import dataclass as odataclass
from pydantic.dataclasses import dataclass
from tesserocr import PSM
from typing import Optional, cast


@dataclass
class OCRParams:
    sl_scalex:        float = 0.1
    sl_scaley:        float = 0.15
    sl_clip_around:   float = 0.05

    hue_offset:       int   = 38
    hue_std:          float = 0.0025
    sat_std:          float = 0.02
    col_zscore:       float = 4

    seg_min_area:     float = 0.025
    seg_mask_th:      float = 0.25
    seg_min_width:    float = 0.1
    seg_black_clip:   int   = 4
    seg_black_th:     int   = 64
    seg_dist_th:      float = 0.75
    seg_dist_vert_th: float = 0.5

    hs_wide_sf:       float = 1.35
    hs_th:            float = 0.5


class OCReadMode(IntEnum):
    LINE = PSM.SINGLE_LINE
    WORD = PSM.SINGLE_WORD


@odataclass
class HSVColourRange:
    low:  np.ndarray
    high: np.ndarray


## Defines the range for red color in HSV space
RED_RGB_SPACE = np.array([ 
    [230, 0, 0],
    [255, 40, 40]
])

def get_timer_redperc(image: np.ndarray) -> float:
    """Returns the % of the image which is red"""
    mask = cv2.inRange(image, RED_RGB_SPACE[0], RED_RGB_SPACE[1])
    return cast(float, np.sum(mask > 0) / mask.size) # type: ignore


def cvt_rgb2hsv(image: np.ndarray, offset: int) -> np.ndarray:
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    hsv_img[:,:,0] += offset  # type: ignore
    return hsv_img

def _round(x: int | float, ndigits: Optional[int] = None) -> int:
    return int(round(x, ndigits))

def get_hsv_range(hs: tuple[int,int], std: tuple[float, float], zscore: float = 3.0) -> HSVColourRange:
    """Generates a upper/lower bound for team colour using tested standard deviations"""
    lowh  = _round(max(0,   hs[0] - std[0] * zscore))
    lows  = _round(max(0,   hs[1] - std[1] * zscore))
    highh = _round(min(255, hs[0] + std[0] * zscore))
    highs = _round(min(255, hs[1] + std[1] * zscore))

    low  = np.array([lowh,  lows,  100], dtype=np.uint8)
    high = np.array([highh, highs, 255], dtype=np.uint8)
    return HSVColourRange(low=low, high=high)
