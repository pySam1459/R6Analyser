import cv2
import numpy as np
from dataclasses import dataclass
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, OEM
from typing import Sequence

from settings import Settings


__all__ = ["OCREngine", "OCRLineResult"]

MatLike = cv2.typing.MatLike

## TODO : remember headshot
@dataclass
class OCRLineResult:
    left:  str
    right: str


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


class OCREngine:
    def __init__(self, settings: Settings) -> None:
        self.api = PyTessBaseAPI(path=settings.tessdata, # type: ignore
                                 psm=PSM.SINGLE_WORD,    # type: ignore
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
        

    def _crop_segments(self, image, contours) -> list[Segment]:
        return [self._get_segment(image, contour) for contour in contours]
    
    def _get_segment(self, image: MatLike, contour: MatLike) -> Segment:
        x, y, w, h = cv2.boundingRect(contour)
        return Segment(image[y:y+h, x:x+w], [x,y,w,h])

    def read_text(self, image: np.ndarray, charlist: str) -> tuple[str, int]:
        self.api.SetVariable("tessedit_char_whitelist", charlist)
        return self._read_text(Image.fromarray(image))
    
    def _read_text(self, pil_image: Image.Image) -> tuple[str, int]:
        self.api.SetImage(pil_image)
        return self.api.GetUTF8Text()[:-1], self.api.MeanTextConf()