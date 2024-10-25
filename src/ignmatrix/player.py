from abc import ABC, abstractmethod
from enum import Enum, auto
from Levenshtein import ratio
from typing import Optional, Callable, Self

from utils.constants import IM_TEAM_DET_THRESHOLD
from utils.enums import Team


class Player_t(Enum):
    FIXED = auto()
    INFER = auto()


class Player(ABC):
    @property
    @abstractmethod
    def uid(self) -> int: ...

    @property
    @abstractmethod
    def type(self) -> Player_t: ...

    @property
    @abstractmethod
    def ign(self) -> str: ...

    @property
    @abstractmethod
    def team(self) -> Team: ...

    def has_team(self) -> bool:
        return self.team != Team.UNKNOWN

    def __eq__(self, other: 'Player') -> bool:
        return self.uid == other.uid


class FixedPlayer(Player):
    def __init__(self, ign: str, team: Team) -> None:
        self.__ign = ign
        self.__team = team

        self.__uid = hash(ign)
    
    @property
    def uid(self) -> int:
        return self.__uid
    
    @property
    def type(self) -> Player_t:
        return Player_t.FIXED
    
    @property
    def ign(self) -> str:
        return self.__ign
    
    @property
    def team(self) -> Team:
        return self.__team


class AdaptivePlayer(Player):
    """A smart player object which keeps track of the pign's observed and 
       uses the name which has occured the most"""
    def __init__(self, pign: str, team: Optional[Team] = None):
        self.__uid = hash(pign + "<|SALT|>")
        self.__names  = [pign]
        self.__counts = [1]

        self.__noccr = 1
        self.__best_idx = 0 ## idx of most-likely pign so far in __names

        self.__team = team
        self.__team_counter = [0, 0]
        self.__opps: dict[int, int] = {}
    
    @property
    def uid(self) -> int:
        return self.__uid
    
    @property
    def type(self) -> Player_t:
        return Player_t.INFER

    @property
    def ign(self) -> str:
        return self.__names[self.__best_idx]

    @property
    def team(self) -> Team:
        if self.__team is not None:
            return self.__team

        t0, t1 = self.__team_counter
        if t0 < t1: ## encounters with team1 => self is team0
            return Team.TEAM0
        elif t0 > t1: ## vice versa
            return Team.TEAM1
        else:
            return Team.UNKNOWN

    @property
    def noccr(self) -> int:
        return self.__noccr

    ## --- Add and Update ---
    def _inc(self, idx: int) -> None:
        self.__counts[idx] += 1
        self.__noccr += 1

        if self.__counts[idx] > self.__counts[self.__best_idx]:
            self.__best_idx = idx

    def _new(self, pign: str) -> None:
        self.__names.append(pign)
        self.__counts.append(1)
        self.__noccr += 1

    def fuzzy_contains(self, pign: str, threshold: float) -> bool:
        return any(map(lambda var: ratio(pign, var) >= threshold, self.__names))

    def add(self, pign: str, threshold: float) -> bool:
        if pign in self.__names:
            pidx = self.__names.index(pign)
            self._inc(pidx)
            return True
        elif self.fuzzy_contains(pign, threshold):
            self._new(pign)
            return True

        return False

    def contains(self, pign: str, threshold: float) -> bool:
        return pign in self.__names or self.fuzzy_contains(pign, threshold)
    
    def evaluate(self, pign: str) -> float:
        return float(pign in self.__names) or max([ratio(pign, var) for var in self.__names])
    
    ## --- Opps ---
    def inc_oppteam(self, oppteam: Team, td_callback: Callable[[Self], None], count = 1) -> None:
        if oppteam == Team.UNKNOWN:
            return

        opp_idx = oppteam.value
        self.__team_counter[opp_idx] += count

        if self.__team_counter[opp_idx] >= IM_TEAM_DET_THRESHOLD:
            self.__team = Team(1-opp_idx)
            self.__opps.clear()
            td_callback(self)

    def add_opp(self, opp_uid: int) -> None:
        if opp_uid not in self.__opps:
            self.__opps[opp_uid] = 1
        else:
            self.__opps[opp_uid] += 1

    def rem_opp(self, opp_uid: int) -> int:
        return self.__opps.pop(opp_uid, 0)
