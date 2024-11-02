from Levenshtein import ratio
from typing import Optional

from utils.enums import Team

from .base import IGNMatrix
from .player import Player
from .utils import Teams


__all__ = ["IGNMatrixFixed"]


class IGNMatrixFixed(IGNMatrix):
    """
    This subclass of IGNMatrix only handles the case where all IGNs are known beforehand
    Separating the different IGN modes aims to improve efficiency, readability and modularity
    """

    def __init__(self, team0: list[str], team1: list[str], leven_th: float) -> None:
        super(IGNMatrixFixed, self).__init__(team0, team1, leven_th)

        self.__teams = Teams([self._ttable[ign] for ign in team0],
                             [self._ttable[ign] for ign in team1])

    def get(self, pign: str, pteam: Team) -> Optional[Player]:
        """Returns the most-likely Player object from a pseudoIGN or None if the pseudoIGN does not meet the threshold"""
        pl = self._ttable.check(pign, self._ign_threshold)
        if pl is not None and pl.team == pteam:
            return pl
        return None

    def get_teams(self) -> Teams:
        """Returns a tuple of the teams currently playing"""
        return self.__teams

    def evaluate(self, pign: str) -> float:
        """Evaluates a given pseudoIGN and returns as score between 0-1"""
        if pign in self._ttable:
            return 1.0

        return max([ratio(pign, ign) for ign in self._ttable.igns()])
