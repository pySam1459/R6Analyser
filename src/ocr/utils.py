import cv2
import numpy as np
from dataclasses import dataclass as odataclass
from typing import Optional, cast


@odataclass
class HSVColourRange:
    low:  np.ndarray
    high: np.ndarray


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
