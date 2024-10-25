import cv2
import numpy as np
from colorsys import rgb_to_hsv
from enum import IntEnum
from dataclasses import dataclass as odataclass
from pydantic.dataclasses import dataclass
from tesserocr import PSM
from typing import Sequence, cast


@dataclass
class OCRParams:
    sl_scalex:        float = 0.4
    sl_scaley:        float = 0.5
    sl_clip_around:   float = 0.05

    hue_std:          float = 0.0025
    sat_std:          float = 0.02
    col_zscore:       float = 4

    seg_min_area:     float = 0.025
    seg_mask_th:      float = 0.25
    seg_min_width:    float = 0.1
    seg_black_th:     int   = 64
    seg_dist_th:      float = 0.75
    seg_dist_vert_th: float = 0.5

    hs_wide_sf:       float = 1.35
    hs_th:            float = 0.65


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


def get_colour_range(rgb: tuple[float,float,float], std: tuple[float, float], zscore: float = 3.0) -> HSVColourRange:
    """Generates a upper/lower bound for team colour using tested standard deviations"""
    hsv = rgb_to_hsv(*rgb)
    return get_hsv_dist(hsv, std, zscore)
    

def get_hsv_dist(hsv: Sequence[float], std: Sequence[float], zscore: float = 3.0) -> HSVColourRange:
    h,s = hsv[:2]
    lowh  = h - std[0] * zscore
    lows  = s - std[1] * zscore
    highh = h + std[0] * zscore
    highs = s + std[1] * zscore
    low  = np.array([int(lowh*180), int(lows*255), 100], dtype=np.int64)
    high = np.array([int(highh*180), int(highs*255), 255], dtype=np.int64)

    return HSVColourRange(low=low, high=high)