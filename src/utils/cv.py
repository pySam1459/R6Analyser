import cv2
import numpy as np
from typing import cast

from . import BBox_t


def offset_bbox(bbox: BBox_t, offset: tuple[int,int]) -> BBox_t:
    return (bbox[0]-offset[0], bbox[1]-offset[1], bbox[2], bbox[3])

def crop2bbox(image: np.ndarray, bbox: BBox_t) -> np.ndarray:
    w,h = bbox[2:]
    x,y = bbox[0], bbox[1]
    return image[y:y+h,x:x+w]


def resize_height(image: np.ndarray, height: int, inter = cv2.INTER_LINEAR) -> np.ndarray:
    h,w = image.shape
    dim = (int(w*height/h), height)
    return cast(np.ndarray, cv2.resize(image, dim, interpolation=inter))

def squeeze_image(image: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(image, None, fx=factor, fy=1.0, interpolation=cv2.INTER_CUBIC)

def clip_around(image: np.ndarray, factor: float) -> np.ndarray:
    assert factor < 0.5
    h,w,*_ = image.shape
    wbuf = int(w*factor)
    hbuf = int(h*factor)
    return image[hbuf:-hbuf,wbuf:-wbuf]
