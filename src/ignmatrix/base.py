from abc import ABC, abstractmethod
from typing import Optional

from utils.enums import IGNMatrixMode, Team
from .utils import Player, TeamTable


__all__ = [ "IGNMatrix" ]


class IGNMatrix(ABC):
    """
    IGN Matrix infers the true IGN from the OCR engine reading of the IGN (pseudoIGN/pign) from the killfeed.
    The matrix can be initialised with a prior list of 'fixed' IGNs (IGNs known before starting the game)
    The matrix uses Levenshtein distance to determine whether two IGNs are the same (could be improved?)
    """

    def __init__(self, mode: IGNMatrixMode, team0: list[str], team1: list[str]) -> None:
        self.__mode = mode
        self._team0 = team0
        self._team1 = team1

        self._ttable = TeamTable({ign: Team.TEAM0 for ign in team0} | {ign: Team.TEAM1 for ign in team1})

    @property
    def team0(self) -> list[str]:
        return self._team0

    @property
    def team1(self) -> list[str]:
        return self._team1

    @property
    def mode(self) -> IGNMatrixMode:
        return self.__mode

    @abstractmethod
    def get(self, pign: str) -> Optional[Player]:
        ...
    
    @abstractmethod
    def get_teams(self) -> tuple[list[Player], list[Player]]:
        """Returns a a pair of lists containing the valid players"""
        ...
    
    @abstractmethod
    def evaluate(self, pign: str) -> float:
        ...
    
    def update_mats(self, pign: str, tign: str) -> None:
        """Records pign killing t(arget)ign, used by IGNMatrixInfer"""
        ...
