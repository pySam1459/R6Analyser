from functools import partial
from Levenshtein import ratio
from typing import Optional

from utils.constants import IM_LEVEN_THRESHOLD
from utils.enums import IGNMatrixMode, Team

from .base import IGNMatrix
from .player import Player, FixedPlayer
from .utils import Teams


__all__ = ["IGNMatrixFixed"]


class IGNMatrixFixed(IGNMatrix):
    """
    This subclass of IGNMatrix only handles the case where all IGNs are known beforehand
    Separating the different IGN modes aims to improve efficiency, readability and modularity
    """

    def __init__(self, team0: list[str], team1: list[str]) -> None:
        super(IGNMatrixFixed, self).__init__(IGNMatrixMode.FIXED, team0, team1)

        self.__teams = Teams([FixedPlayer(ign, Team.TEAM0) for ign in team0],
                             [FixedPlayer(ign, Team.TEAM1) for ign in team1])

    def get(self, pign: str) -> Optional[Player]:
        """Returns the most-likely Player object from a pseudoIGN or None if the pseudoIGN does not meet the threshold"""
        return self._ttable.check(pign, IM_LEVEN_THRESHOLD)

    def get_teams(self) -> Teams:
        """Returns a tuple of the teams currently playing"""
        return self.__teams

    def evaluate(self, pign: str) -> float:
        """Evaluates a given pseudoIGN and returns as score between 0-1"""
        if pign in self._ttable:
            return 1.0

        ratio_func = partial(ratio, pign)
        return max(map(ratio_func, self._ttable.igns()))
