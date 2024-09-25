from Levenshtein import ratio as leven_ratio
from typing import Optional

from utils.enums import Team

from .player import Player, FixedPlayer


class TeamTable:
    """A fixed table of (ign, team) pairs"""
    def __init__(self, player_data: dict[str, Team]) -> None:
        self.__table = player_data
        self.__keys = list(player_data.keys())

    def check(self, pign: str, threshold: float) -> Optional[Player]:
        if pign in self.__table:
            return FixedPlayer(pign, self.__table[pign])

        score, ign = max([(leven_ratio(ign, pign), ign) for ign in self.__table.keys()],
                         key=lambda t: t[0])
        
        if score >= threshold:
            return FixedPlayer(ign, self.__table[ign])

        return None
    
    def eval(self, pign: str) -> float:
        if pign in self.__table:
            return 1.0
        
        return max([leven_ratio(ign, pign) for ign in self.__table.keys()])
    
    def get_players(self) -> list[Player]:
        return [FixedPlayer(ign, team) for ign, team in self.__table.items()]

    def igns(self) -> list[str]:
        return self.__keys.copy()

    def __len__(self) -> int:
        return len(self.__table)
    
    def __contains__(self, el: str) -> bool:
        return el in self.__table
