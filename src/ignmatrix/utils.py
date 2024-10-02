 from dataclasses import dataclass
from Levenshtein import ratio
from typing import Optional

from utils.enums import Team

from .player import Player, FixedPlayer


@dataclass
class Teams:
    team0: list[Player]
    team1: list[Player]

    def order(self, first: Team) -> tuple[list[Player], list[Player]]:
        if first == Team.TEAM1:
            return (self.team1, self.team0)
        else: ## team0 and unknown
            return (self.team0, self.team1)

    def combine(self, first: Team) -> list[Player]:
        if first == Team.TEAM1:
            return self.team1 + self.team0
        else: ## team0 and unknown
            return self.team0 + self.team1


class TeamTable:
    """A fixed table of (ign, team) pairs"""
    def __init__(self, player_data: dict[str, Team]) -> None:
        self.__table = player_data
        self.__keys = list(player_data.keys())

    def check(self, pign: str, threshold: float) -> Optional[Player]:
        if pign in self.__table:
            return FixedPlayer(pign, self.__table[pign])

        score, ign = max([(ratio(ign, pign), ign) for ign in self.__table.keys()],
                         key=lambda t: t[0])
        
        if score >= threshold:
            return FixedPlayer(ign, self.__table[ign])

        return None
    
    def eval(self, pign: str) -> float:
        if pign in self.__table:
            return 1.0
        
        return max([ratio(ign, pign) for ign in self.__table.keys()])
    
    def get_players(self) -> list[Player]:
        return [FixedPlayer(ign, team) for ign, team in self.__table.items()]

    def igns(self) -> list[str]:
        return self.__keys.copy()

    def __len__(self) -> int:
        return len(self.__table)
    
    def __contains__(self, el: str) -> bool:
        return el in self.__table


def get_chars(names: list[str]) -> str:
    concat = "".join(names)
    return "".join(set(concat))
