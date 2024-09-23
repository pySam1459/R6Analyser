import json
from pathlib import Path

from history import History
from ignmatrix import IGNMatrix
from utils.enums import SaveFileType
from .base import Writer


__all__ = ["JsonWriter"]


class JsonWriter(Writer):
    def __init__(self, save_path: Path) -> None:
        super(JsonWriter, self).__init__(SaveFileType.JSON, save_path, None)

    def write(self, history: History, *_) -> None:
        """Writes the history to a json file"""

        with open(self._save_path, "w") as f_out:
            json.dump(history.model_dump(), f_out, indent=4)
