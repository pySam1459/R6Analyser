from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from config import Config
from history import History
from ignmatrix import IGNMatrix, Player
from utils.enums import SaveFileType
from .utils import make_copyfile


__all__ = ["Writer"]


class Writer(ABC):
    def __init__(self, _type: SaveFileType, save_path: Path, config: Optional[Config]) -> None:
        self._type = _type
        self._config = config

        if save_path.exists():
            self._save_path = make_copyfile(save_path)
        else:
            self._save_path = save_path

    @property
    def type(self) -> SaveFileType:
        return self._type
    
    @property
    def save_path(self) -> Path:
        return self._save_path

    @abstractmethod
    def write(self, history: History, ignmat: IGNMatrix) -> None:
        ...
