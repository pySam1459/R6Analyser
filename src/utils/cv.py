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


def gen_gaussian2d(size: int, spread = 3.0) -> np.ndarray:
    center = size // 2

    x = np.arange(0, size, 1)
    y = np.arange(0, size, 1)
    x, y = np.meshgrid(x, y)

    sigma = size / spread
    return np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))

def guassian_threshold(image: np.ndarray, gaussian: np.ndarray) -> np.ndarray:
    h,w = image.shape
    gaussian_resize = cv2.resize(gaussian, (w, h), interpolation=cv2.INTER_CUBIC)
    gaussian_image = cast(np.ndarray, gaussian_resize * image)

    min_, max_ = np.min(gaussian_image), np.max(gaussian_image)
    if (max_-min_) == 0:
        return np.zeros_like(image)

    image_norm = (gaussian_image - min_) / (max_ - min_)
    return (255 * image_norm).astype(np.uint8)
