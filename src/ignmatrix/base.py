from abc import ABC, abstractmethod
from typing import Optional

from utils.enums import Team

from .player import Player
from .utils import TeamTable, Teams, get_chars


__all__ = ["IGNMatrix"]


class IGNMatrix(ABC):
    """
    IGN Matrix infers the true IGN from the OCR engine reading of the IGN (pseudoIGN/pign) from the killfeed.
    The matrix can be initialised with a prior list of 'fixed' IGNs (IGNs known before starting the game)
    The matrix uses Levenshtein distance to determine whether two IGNs are the same (could be improved?)
    """

    def __init__(self, team0: list[str], team1: list[str], leven_th: float) -> None:
        self._team0 = team0
        self._team1 = team1
        self._ign_threshold = leven_th

        self._charlist = get_chars(team0 + team1)

        self._ttable = TeamTable({ign: Team.TEAM0 for ign in team0} | {ign: Team.TEAM1 for ign in team1})

    @property
    def team0(self) -> list[str]:
        return self._team0

    @property
    def team1(self) -> list[str]:
        return self._team1
    
    @property
    def charlist(self) -> str:
        return self._charlist


    @abstractmethod
    def get(self, pign: str, pteam: Team) -> Optional[Player]:
        """Returns the most-likely Player instance for a given pign"""
    
    @abstractmethod
    def get_teams(self) -> Teams:
        """Returns a Teams object containing 2 lists of Player instances"""
    
    @abstractmethod
    def evaluate(self, pign: str) -> float:
        """Returns a eval score for a given pign, eval'd against the ttable and matrix"""
    
    def update_mat(self, pign: str, pteam: Team) -> None:
        """Records pign killing t(arget)ign, used by IGNMatrixInfer"""
