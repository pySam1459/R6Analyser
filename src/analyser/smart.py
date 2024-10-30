import numpy as np
from enum import Enum, auto
from dataclasses import dataclass as odataclass
from typing import Callable, Optional

from capture.regions import Regions
from config import Config
from history import History, KFRecord
from ignmatrix import Player
from ocr import OCREngine
from utils import Scoreline, mode_count
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
        if self.is_dead(record.target):
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
    
    def is_dead(self, pl: Player) -> bool:
        return any([record.target == pl
                    for record in self.__history.cround.killfeed])

    def reset(self) -> None:
        self.__purgatory.clear()


@odataclass
class _ScorelineVote:
    sl: Optional[Scoreline]
    team0_image: np.ndarray
    team1_image: np.ndarray


class SmartScoreline:
    def __init__(self, ocr_engine: OCREngine, majority_th: int):
        self.__ocr_engine = ocr_engine
        self.__majority_th = majority_th
        self.__ballet_box: list[_ScorelineVote] = []
    
    def get_scoreline(self, regions: Regions) -> Optional[Scoreline]:
        left_text  = self.__ocr_engine.read_score(regions.team0_score, regions.team0_side)
        right_text = self.__ocr_engine.read_score(regions.team1_score, regions.team1_side)
        if left_text is None or right_text is None:
            return None
        else:
            return Scoreline(left=int(left_text), right=int(right_text))

    def read(self, regions: Regions) -> Optional[Scoreline]:
        sl = self.get_scoreline(regions)

        vote = _ScorelineVote(sl, regions.team0_score, regions.team1_score)
        self.__ballet_box.append(vote)

        if len(self.__ballet_box) > self.__majority_th:
            self.__ballet_box.pop(0)

        if any([vote.sl is None for vote in self.__ballet_box]):
            return None

        vote = self.__deciding_vote()
        if not self.__ocr_engine.has_colours:
            self.__ocr_engine.set_colours(vote.team0_image, vote.team1_image)

        return vote.sl

    def __deciding_vote(self) -> _ScorelineVote:
        left_image_map = {
            vote.sl.left: vote.team0_image
            for vote in self.__ballet_box
            if vote.sl is not None
        }
        left_scores = [vote.sl.left for vote in self.__ballet_box if vote.sl is not None]
        left_score = mode_count(left_scores)

        right_image_map = {
            vote.sl.right: vote.team1_image
            for vote in self.__ballet_box
            if vote.sl is not None
        }
        right_scores = [vote.sl.right for vote in self.__ballet_box if vote.sl is not None]
        right_score = mode_count(right_scores)

        sl = Scoreline(left=left_score, right=right_score)
        return _ScorelineVote(sl, left_image_map[sl.left], right_image_map[sl.right])

    def clear(self) -> None:
        self.__ballet_box.clear()
