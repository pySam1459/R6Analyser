from abc import ABC, abstractmethod
from enum import Enum
from Levenshtein import ratio

from utils.enums import Team


class Player(ABC):
    @property
    @abstractmethod
    def uid(self) -> int: ...

    @property
    @abstractmethod
    def ign(self) -> str: ...

    @property
    @abstractmethod
    def team(self) -> Team: ...

    def __eq__(self, other: "Player") -> bool:
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
    def ign(self) -> str:
        return self.__ign
    
    @property
    def team(self) -> Team:
        return self.__team

    def __str__(self) -> str:
        return f"FixedPlayer[ign={self.ign},team={self.team}]"
    __repr__ = __str__


class AdaptivePlayer(Player):
    """A smart player object which keeps track of the pign's observed and 
       uses the name which has occured the most"""
    def __init__(self, pign: str, team: Team):
        self.__uid = hash(pign + "<|SALT|>") ## maybe add team in hash as well?
        self.__team = team

        self.__names  = [pign]
        self.__counts = [1]

        self.__noccr = 1
        self.__best_idx = 0 ## idx of most-likely pign so far in __names

    @property
    def uid(self) -> int:
        return self.__uid

    @property
    def ign(self) -> str:
        return self.__names[self.__best_idx]

    @property
    def team(self) -> Team:
        return self.__team
    
    def __str__(self) -> str:
        return f"AdaptivePlayer[ign={self.ign},team={self.team}]"
    __repr__ = __str__

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
