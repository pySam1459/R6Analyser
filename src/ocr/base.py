import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, OEM
from typing import overload, Sequence

from settings import Settings

from .utils import OCReadMode


class BaseOCREngine:
    def __init__(self, settings: Settings) -> None:
        self._api = PyTessBaseAPI(path=str(settings.tessdata), oem=OEM.LSTM_ONLY) # type: ignore

    @overload
    def readtext(self, image: np.ndarray, read_mode: OCReadMode, charlist: str) -> str: ...
    @overload
    def readtext(self, image: Sequence[np.ndarray], read_mode: OCReadMode, charlist: str) -> list[str]: ...

    def readtext(self, image: np.ndarray | Sequence[np.ndarray], read_mode: OCReadMode, charlist: str) -> str | list[str]:
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
    
    def stop(self) -> None:
        self._api.End()
    