import cv2
import numpy as np
from typing import Optional, Callable, cast

from utils import BBox_t


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


## Defines the range for red color in HSV space
RED_RGB_SPACE = np.array([ 
    [230, 0, 0],
    [255, 40, 40]
])

def get_timer_redperc(image: np.ndarray) -> float:
    """Returns the % of the image which is red"""
    mask = cv2.inRange(image, RED_RGB_SPACE[0], RED_RGB_SPACE[1])
    return cast(float, np.sum(mask > 0) / mask.size)
