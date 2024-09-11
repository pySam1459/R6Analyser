import importlib
import easyocr
import numpy as np
from abc import ABC, abstractmethod
from pydantic import BaseModel, model_validator
from typing import Any, Optional

from ignmatrix import IGNMatrix
from utils import bbox_to_rect
from utils.enums import OCREngineType


__all__ = ["OCREngine", "OCResult", "OCRLine"]


class OCResult(BaseModel):
    """
    A dataclass containing the data of a single easyOCR reading, the data stored includes
        rect - rectangle bounding the text, text, prob - probability assigned by the easyOCR engine, eval_score - IGNMatrix.evaluate score
    """
    bbox: list[list[int]]
    rect: list[int]
    text: str
    prob: float
    eval_score: Optional[float] = None

    @model_validator(mode='before')
    @classmethod
    def compute_rect(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict) and "rect" not in data:
            bbox = data.get("bbox")
            if bbox is not None:
                data["rect"] = bbox_to_rect(bbox)
        return data

    def eval(self, ign_matrix: IGNMatrix) -> float:
        """Sets the eval_score of self.text given an IGNMatrix"""
        self.eval_score = ign_matrix.evaluate(self.text)
        return self.eval_score
    
    def join(self, other: 'OCResult') -> 'OCResult':
        """
        Combines 2 OCResult objects into 1,
          a solution to a issue with the EasyOCR's detection engine where it wouldn't properly detect a name with an underscore in it
        """
        x, y = min(self.rect[0], other.rect[0]), min(self.rect[1], other.rect[1])
        x2, y2 = max(self.bbox[1][0], other.bbox[1][0]), max(self.bbox[2][1], other.bbox[2][1])

        return OCResult.model_validate({
            "bbox": [[x, y], [x2, y], [x2, y2], [x, y2]],
            "text": f"{self.text}_{other.text}",
            "prob": min(self.prob, other.prob)
        })
    
    def __str__(self) -> str:
        if self.eval_score is not None: return f"{self.text}|eval={self.eval_score*100:.2f}"
        else: return f"{self.text}|prob={self.prob*100:.2f}"

    __repr__ = __str__


class OCRLine(BaseModel):
    """A helper model containing the OCResults and other info from a single killfeed line"""
    results: list[OCResult]
    headshot: bool = False


class OCREngine(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def load_msg(self) -> str:
        ...
    
    @abstractmethod 
    def read(self, image: np.ndarray, /, **kwargs) -> list[OCResult]:
        ...
    
    @abstractmethod
    def read_batch(self, images: list[np.ndarray], /, **kwargs) -> tuple[list[OCResult]]:
        ...
    
    @staticmethod
    def new(reader_type: OCREngineType, language = "en", *args, **kwargs) -> 'OCREngine':
        match reader_type:
            case OCREngineType.EASYOCR:
                return EasyOCREngine(language, *args, **kwargs)
            case _:
                raise ValueError(f"Invalid OCReader type {reader_type}")


class EasyOCREngine(OCREngine):
    def __init__(self, language: str) -> None:
        self._reader = easyocr.Reader([language], gpu=True)

    def load_msg(self) -> str:
        return "EasyOCR Engine loaded!"
    
    def read(self, image: np.ndarray, **kwargs) -> list[OCResult]:
        return self._reader.readtext(image, **kwargs)
    
    def read_batch(self, images: list[np.ndarray], **kwargs) -> tuple[list[OCResult]]:
        return self._reader.readtext_batched(images, **kwargs) # type: ignore
