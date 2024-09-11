from functools import partial
from Levenshtein import ratio as leven_ratio
from typing import Optional

from utils.enums import IGNMatrixMode, Team
from .base import IGNMatrix
from .utils import Player, LEVEN_THRESHOLD


__all__ = ["IGNMatrixFixed"]


class IGNMatrixFixed(IGNMatrix):
    """
    This subclass of IGNMatrix only handles the case where all IGNs are known beforehand
    Separating the different IGN modes aims to improve efficiency, readability and modularity
    """

    def __init__(self, team0: list[str], team1: list[str]) -> None:
        super(IGNMatrixFixed, self).__init__(IGNMatrixMode.FIXED, team0, team1)

        self.__teams = ([Player(ign, Team.TEAM0) for ign in team0],
                        [Player(ign, Team.TEAM1) for ign in team1])

    def get(self, pign: str) -> Optional[Player]:
        """Returns the most-likely Player object from a pseudoIGN or None if the pseudoIGN does not meet the threshold"""
        return self._ttable.check(pign, LEVEN_THRESHOLD)

    def get_teams(self) -> tuple[list[Player], list[Player]]:
        """Returns a tuple of the teams currently playing"""
        return self.__teams

    def evaluate(self, pign: str) -> float:
        """Evaluates a given pseudoIGN and returns as score between 0-1"""
        if pign in self._ttable:
            return 1.0

        ratio_func = partial(leven_ratio, pign)
        return max(map(ratio_func, self._ttable.keys()))
