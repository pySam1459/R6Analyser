import cv2
import numpy as np
from pathlib import Path
from typing import Self

from utils.constants import ASSETS_PATH


class Assets:
    atkside_icon: np.ndarray
    headshot: np.ndarray

    _ASSET_MAP = {
        "atkside_icon": "atkside_icon.jpg",
        "headshot": "headshot.jpg"
    }

    def __init__(self) -> None:
        for name, file in Assets._ASSET_MAP.items():
            path = ASSETS_PATH / file
            setattr(self, name, Assets._load_asset(path))
    
    @staticmethod
    def _load_asset(path: Path) -> np.ndarray:
        if not path.exists():
            raise ValueError(f"Asset path {path} does not exist!")
        
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def resize(self, name: str, dim: tuple[int,int]) -> Self:
        img = getattr(self, name)
        img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        setattr(self, name, img_resize)
        return self
    
    def resize_height(self, name: str, height: int) -> Self:
        img = getattr(self, name)
        h,w = img.shape
        dim = (int(w*height/h), height)
        img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        setattr(self, name, img_resize)
        return self
