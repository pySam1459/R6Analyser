import numpy as np
from pathlib import Path
from PIL import Image
from tesserocr import PyTessBaseAPI, OEM, PSM
from typing import overload, Sequence, Optional, Callable

from settings import Settings

from .utils import OCReadMode


TESSERACT_VARS = {"tessedit_fix_hyphens": "false"}

class BaseOCREngine:
    def __init__(self, settings: Settings, _debug_print: Optional[Callable] = None) -> None:
        self._api = PyTessBaseAPI(path=str(settings.tessdata), # type: ignore
                                  oem=OEM.LSTM_ONLY,           # type: ignore
                                  variables=TESSERACT_VARS)    # type: ignore

        self._debug_print = _debug_print

    @overload
    def readtext(self, image: np.ndarray, read_mode: OCReadMode, charlist: str) -> str: ...
    @overload
    def readtext(self, image: Sequence[np.ndarray], read_mode: OCReadMode, charlist: str) -> list[str]: ...

    def readtext(self, image: np.ndarray | Sequence[np.ndarray], read_mode: OCReadMode | PSM, charlist: str) -> str | list[str]:
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
    

    def debug_print(self, *args) -> None:
        if self._debug_print is not None:
            self._debug_print(*args)
    

    def _debug_threshold(self, image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        self._api.SetImage(image)
        return self._api.GetThresholdedImage()
