from dataclasses import dataclass
from enum import Enum, auto
from Levenshtein import ratio as leven_ratio
from typing import Optional

from utils.enums import Team


LEVEN_THRESHOLD = 0.65 ## threshold for Levenshtein distance to determine equality
TEAM_DET_THRESHOLD = 3


class Player_t(Enum):
    FIXED = auto()
    INFER = auto()


@dataclass
class Player:
    _type: Player_t
    ign:   str
    team:  Team

    def __init__(self, ign: str, team: Team) -> None:
        self._type = Player_t.FIXED
        self.ign = ign
        self.team = team

        self.__id = hash(ign)
    
    @property
    def uid(self) -> int:
        return self.__id
    
    @property
    def type(self) -> Player_t:
        return self._type
    
    def __eq__(self, other: 'Player') -> bool:
        return self.uid == other.uid


class TeamTable(dict[str, Team]):
    def check(self, pign: str, threshold: float) -> Optional[Player]:
        if pign in self:
            return Player(pign, self[pign])

        score, ign = max([(leven_ratio(ign, pign), ign) for ign in self.keys()],
                         key=lambda t: t[0])
        
        if score >= threshold:
            return Player(ign, self[ign])

        return None
    
    def eval(self, pign: str) -> float:
        if pign in self:
            return 1.0
        
        return max([leven_ratio(ign, pign) for ign in self.keys()])
    
    def get_players(self) -> list[Player]:
        return [Player(ign, team) for ign, team in self.items()]
