import cv2
import numpy as np
from enum import IntEnum
from pydantic.dataclasses import dataclass
from tesserocr import PSM
from typing import cast


@dataclass
class OCRParams:
    sl_scalex:        float = 0.4
    sl_scaley:        float = 0.5
    sl_clip_around:   float = 0.05

    seg_min_area:     float = 0.025
    seg_black_th:     int   = 64
    seg_dist_th:      float = 0.75
    seg_dist_vert_th: float = 0.5

    bs_left_clip:     float = 0.4
    bs_right_clip:    float = 0.1

    xc_window:        int   = 10
    xc_messy_th:      float = 0.85
    xc_right_clip:    float = 0.1

    hs_wide_sf:       float = 1.35
    hs_th:            float = 0.65


class OCReadMode(IntEnum):
    LINE = PSM.SINGLE_LINE
    WORD = PSM.SINGLE_WORD


## Defines the range for red color in HSV space
RED_RGB_SPACE = np.array([ 
    [230, 0, 0],
    [255, 40, 40]
])

def get_timer_redperc(image: np.ndarray) -> float:
    """Returns the % of the image which is red"""
    mask = cv2.inRange(image, RED_RGB_SPACE[0], RED_RGB_SPACE[1])
    return cast(float, np.sum(mask > 0) / mask.size) # type: ignore
