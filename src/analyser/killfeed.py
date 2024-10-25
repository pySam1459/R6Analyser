from enum import Enum, auto
from typing import Callable

from config import Config
from history import History, KFRecord
from utils.enums import IGNMatrixMode


__all__ = [
    "KFRecord_t",
    "SmartKillfeed"
]


class KFRecord_t(Enum):
    NORMAL = auto()
    SUICIDE = auto()


NORMAL_OCCR_THRESHOLDS = {
    IGNMatrixMode.FIXED: 0,
    IGNMatrixMode.INFER: 1
}

SUICIDE_OCCR_THRESHOLDS = {
    IGNMatrixMode.FIXED: 1,
    IGNMatrixMode.INFER: 1
}


class SmartKillfeed:
    def __init__(self, config: Config, history: History, callback: Callable[[KFRecord],None]) -> None:
        self.__config = config
        self.__history = history
        self.__callback = callback ## will be called when ready to add to history

        self.__purgatory: dict[KFRecord, int] = {}
    
    def add(self, record: KFRecord, _type: KFRecord_t) -> None:
        if record in self.__history.cround.killfeed:
            return

        match _type:
            case KFRecord_t.NORMAL:
                occr_th = NORMAL_OCCR_THRESHOLDS[self.__config.ign_mode]
                self.__add(record, occr_th)
            case KFRecord_t.SUICIDE:
                occr_th = SUICIDE_OCCR_THRESHOLDS[self.__config.ign_mode]
                self.__add(record, occr_th)
    
    def __add(self, record: KFRecord, occr_th: int) -> None:
        if self.__purgatory.get(record, 0) >= occr_th:
            self.__callback(record)

        elif record not in self.__purgatory:
            self.__purgatory[record] = 1
        else:
            self.__purgatory[record] += 1

    def reset(self) -> None:
        self.__purgatory.clear()
