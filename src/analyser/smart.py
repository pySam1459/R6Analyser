import numpy as np
from dataclasses import dataclass as odataclass
from typing import Optional

from capture.regions import Regions
from params import ScorelineParams
from ocr import OCREngine
from utils import Scoreline, mode_count
from utils.enums import Team


__all__ = [
    "SmartScoreline",
]


@odataclass
class _ScorelineVote:
    sl: Optional[Scoreline]
    team0_image: np.ndarray
    team1_image: np.ndarray

    def __repr__(self) -> str:
        if self.sl is None:
            return "None"
        return f"Scoreline[left={self.sl.left},right={self.sl.right}]"


class SmartScoreline:
    def __init__(self, ocr_engine: OCREngine, params: ScorelineParams):
        self.__ocr_engine = ocr_engine
        self.__params = params

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

        if len(self.__ballet_box) > self.__params.ballet_box_size:
            self.__ballet_box.pop(0)

        vote = self.__deciding_vote()
        if vote is None:
            return None

        if not self.__ocr_engine.has_colours:
            self.__ocr_engine.set_colours(vote.team0_image, vote.team1_image)

        return vote.sl

    def __deciding_vote(self) -> Optional[_ScorelineVote]:
        left  = self.__deciding_vote_side(Team.TEAM0)
        right = self.__deciding_vote_side(Team.TEAM1)
        if left is None or right is None:
            return None

        sl = Scoreline(left=left[0], right=right[0])
        return _ScorelineVote(sl, left[1], right[1])

    def __deciding_vote_side(self, team: Team) -> Optional[tuple[int, np.ndarray]]:
        sl_attr, image_attr = {0: ("left", "team0_image"), 1: ("right", "team1_image")}[team.value]

        image_map = {
            getattr(vote.sl, sl_attr): getattr(vote, image_attr)
            for vote in self.__ballet_box
            if vote.sl is not None
        }
        scores = [getattr(vote.sl, sl_attr)
                  for vote in self.__ballet_box
                  if vote.sl is not None]
        if len(scores) < self.__params.majority_threshold:
            return None
        
        mode_score, count = mode_count(scores)
        if count < self.__params.majority_threshold:
            return None
        
        return mode_score, image_map[mode_score]


    def clear(self) -> None:
        self.__ballet_box.clear()
