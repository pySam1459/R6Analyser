import numpy as np
from enum import IntEnum
from PIL import Image
from tesserocr import PyTessBaseAPI, OEM, PSM
from typing import Sequence, Optional, Callable, overload

from settings import Settings


__all__ = [
    "BaseOCREngine",
    "OCRMode"
]


class OCRMode(IntEnum):
    CHAR = PSM.SINGLE_CHAR
    WORD = PSM.SINGLE_WORD
    LINE = PSM.SINGLE_LINE


class BaseOCREngine:
    def __init__(self, settings: Settings, _debug_print: Optional[Callable] = None) -> None:
        self._api = PyTessBaseAPI(path=str(settings.tessdata), oem=OEM.LSTM_ONLY) # type: ignore

        self._debug_print = _debug_print

    @overload
    def readtext(self, image: np.ndarray, read_mode: OCRMode, charlist: str) -> str: ...
    @overload
    def readtext(self, image: Sequence[np.ndarray], read_mode: OCRMode, charlist: str) -> list[str]: ...

    def readtext(self, image: np.ndarray | Sequence[np.ndarray], read_mode: OCRMode, charlist: str) -> str | list[str]:
        self._api.SetVariable("tessedit_pageseg_mode", str(read_mode))
        self._api.SetVariable("tessedit_char_whitelist", charlist)

        if isinstance(image, Sequence):
            return list(map(self.__readtext, image))
        elif isinstance(image, np.ndarray):
            return self.__readtext(image)

    def __readtext(self, image: Image.Image | np.ndarray) -> str:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        self._api.SetImage(image)
        return self._api.GetUTF8Text()[:-1]
    
    def _read_threshold(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        self._api.SetImage(image)
        return np.asarray(self._api.GetThresholdedImage())
    
    def stop(self) -> None:
        self._api.End()
    

    def debug_print(self, *args) -> None:
        if self._debug_print is not None:
            self._debug_print(*args)
