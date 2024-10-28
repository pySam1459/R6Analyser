from dataclasses import dataclass as odataclass
from Levenshtein import ratio
from typing import Optional

from utils.enums import Team

from .player import Player, FixedPlayer


@odataclass
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
        self.__table = {pign: FixedPlayer(pign, pteam)
                        for pign, pteam in player_data.items()}
        self.__keys = list(player_data)

    def check(self, pign: str, threshold: float) -> Optional[Player]:
        if pign in self.__keys:
            return self.__table[pign]

        score, ign = max([(ratio(ign, pign), ign) for ign in self.__keys],
                         default=(0, None))
        
        if score >= threshold and ign is not None:
            return self.__table[ign]

        return None
    
    def eval(self, pign: str) -> float:
        return (float(pign in self.__keys) or
                max([ratio(ign, pign) for ign in self.__table], default=0))
    
    def get_players(self) -> list[Player]:
        return list(self.__table.values())

    def igns(self) -> list[str]:
        return self.__keys.copy()

    def __len__(self) -> int:
        return len(self.__keys)
    
    def __contains__(self, pign: str) -> bool:
        return pign in self.__keys
    
    def __getitem__(self, key: str) -> FixedPlayer:
        return self.__table[key]


def get_chars(names: list[str]) -> str:
    concat = "".join(names)
    return "".join(set(concat))
