from Levenshtein import ratio as leven_ratio
from typing import Optional, Callable, Self

from utils.enums import IGNMatrixMode, Team

from .base import IGNMatrix
from .utils import Player, LEVEN_THRESHOLD, TEAM_DET_THRESHOLD


__all__ = ["IGNMatrixInfer"]


class AdaptivePlayer(Player):
    """A smart player object which keeps track of the pign's observed and 
       uses the name which has occured the most"""
    def __init__(self, pign: str, team: Optional[Team] = None):
        self.__id = hash(pign)
        self.__names  = [pign]
        self.__counts = [1]

        self.__noccr = 1
        self.__best_idx = 0 ## idx of most-likely pign so far in __names

        self.__team = team
        self.__team_counter = [0, 0]
        self.__opps: dict[int, int] = {}

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
    def id(self) -> int:
        return self.__id

    @property
    def noccr(self) -> int:
        return self.__noccr

    def has_team(self) -> bool:
        return self.team != Team.UNKNOWN

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
        return any(map(lambda var: leven_ratio(pign, var) >= threshold, self.__names))

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
        return float(pign in self.__names) or max([leven_ratio(pign, var) for var in self.__names])
    
    ## --- Opps ---
    def inc_oppteam(self, oppteam: Team, td_callback: Callable[[Self], None], count = 1) -> None:
        if oppteam == Team.UNKNOWN:
            return

        self.__team_counter[oppteam] += count

        if self.__team_counter[oppteam] >= TEAM_DET_THRESHOLD:
            self.__team = Team(1-oppteam)
            self.__opps.clear()
            td_callback(self)

    def add_opp(self, _id: int) -> None:
        if _id not in self.__opps:
            self.__opps[_id] = 1
        else:
            self.__opps[_id] += 1

    def rem_opp(self, _id: int) -> int:
        return self.__opps.pop(_id, 0)


class IGNMatrixInfer(IGNMatrix):
    def __init__(self, team0: list[str], team1: list[str]) -> None:
        super(IGNMatrixInfer, self).__init__(IGNMatrixMode.INFER, team0, team1)

        self.__mat: list[AdaptivePlayer] = []

    def get(self, pign: str) -> Optional[Player]:
        if (pl := self._ttable.check(pign, LEVEN_THRESHOLD)) is not None:
            return pl

        return self._mat_get(pign)

    def _mat_get(self, pign: str) -> Optional[AdaptivePlayer]:
        return next((ap for ap in self.__mat if ap.contains(pign, LEVEN_THRESHOLD)), None)
    
    def _mat_get_topK(self, k: int) -> list[AdaptivePlayer]:
        """sorted is more appropriate over heapq.nlargest as k ~= len(__heap)"""
        return sorted(filter(AdaptivePlayer.has_team, self.__mat), key=lambda x: x.noccr)[:k]

    def get_teams(self) -> tuple[list[Player], list[Player]]:
        players = self._ttable.get_players()
        fxd_length = len(players)

        if fxd_length < 10:
            players.extend(self._mat_get_topK(10-fxd_length))

        return ([pl for pl in players if pl.team == 0],
                [pl for pl in players if pl.team == 1])

    def evaluate(self, pign: str) -> float:
        max_score = max([ap.evaluate(pign) for ap in self.__mat])
        return max(max_score, self._ttable.eval(pign))


    def update_mat(self, pign: str, tign: str) -> None:
        """Records an interaction between pign and tign"""
        pl = self._ttable.check(pign, LEVEN_THRESHOLD)
        if pl is not None: ## already known
            return

        if (pap := self._mat_get(pign)) is None:
            ## pign has not been seen, add to heap
            ## assign to team0 if first ign ever seen
            self.__mat_add(pign)
            return

        if (tap := self.get(tign)) is None:
            tap = self.__mat_add(tign)

        pap.add(pign, LEVEN_THRESHOLD)
        if pap.has_team():
            return
        
        if tap.team != Team.UNKNOWN:
            pap.inc_oppteam(tap.team, self.team_det_callback)

        elif isinstance(tap, AdaptivePlayer):
            tap.add_opp(pap.id)
            
    
    def __mat_add(self, pign: str) -> AdaptivePlayer:
        team = Team.TEAM0 if self.is_first_ign() else None
        ap = AdaptivePlayer(pign, team)
        self.__mat.append(ap)
        return ap

    def is_first_ign(self) -> bool:
        return len(self._ttable) == 0 and len(self.__mat) == 0
    
    def team_det_callback(self, pap: AdaptivePlayer) -> None:
        for tap in self.__mat:
            if not tap.has_team() and (count := tap.rem_opp(pap.id)) > 0:
                tap.inc_oppteam(pap.team, self.team_det_callback, count=count)
