from Levenshtein import ratio as leven_ratio
from typing import Optional, Sequence

from utils.constants import IM_LEVEN_THRESHOLD, IM_TEAM_DET_THRESHOLD
from utils.enums import IGNMatrixMode, Team

from .base import IGNMatrix
from .player import Player, AdaptivePlayer
from .utils import Player


__all__ = ["IGNMatrixInfer"]


class IGNMatrixInfer(IGNMatrix):
    def __init__(self, team0: list[str], team1: list[str]) -> None:
        super(IGNMatrixInfer, self).__init__(IGNMatrixMode.INFER, team0, team1)

        self.__mat: list[AdaptivePlayer] = []

    def get(self, pign: str) -> Optional[Player]:
        if (pl := self._ttable.check(pign, IM_LEVEN_THRESHOLD)) is not None:
            return pl

        return self._mat_get(pign)

    def _mat_get(self, pign: str) -> Optional[AdaptivePlayer]:
        return next((ap for ap in self.__mat if ap.contains(pign, IM_LEVEN_THRESHOLD)), None)
    
    def _mat_get_topK(self, k: int) -> list[AdaptivePlayer]:
        """sorted is more appropriate over heapq.nlargest as k ~= len(__heap)"""
        return sorted(filter(Player.has_team, self.__mat), key=lambda x: x.noccr)[:k]

    def get_teams(self) -> tuple[Sequence[Player], Sequence[Player]]:
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
        pl = self._ttable.check(pign, IM_LEVEN_THRESHOLD)
        if pl is not None: ## already known
            return

        if (pap := self._mat_get(pign)) is None:
            ## pign has not been seen, add to heap
            ## assign to team0 if first ign ever seen
            self.__mat_add(pign)
            return

        if (tap := self.get(tign)) is None:
            tap = self.__mat_add(tign)

        pap.add(pign, IM_LEVEN_THRESHOLD)
        if pap.has_team():
            return
        
        if tap.team != Team.UNKNOWN:
            pap.inc_oppteam(tap.team, self.team_det_callback)

        elif isinstance(tap, AdaptivePlayer):
            tap.add_opp(pap.uid)
            
    
    def __mat_add(self, pign: str) -> AdaptivePlayer:
        team = Team.TEAM0 if self.is_first_ign() else None
        ap = AdaptivePlayer(pign, team)
        self.__mat.append(ap)
        return ap

    def is_first_ign(self) -> bool:
        return len(self._ttable) == 0 and len(self.__mat) == 0
    
    def team_det_callback(self, pap: AdaptivePlayer) -> None:
        for tap in self.__mat:
            if not tap.has_team() and (count := tap.rem_opp(pap.uid)) > 0:
                tap.inc_oppteam(pap.team, self.team_det_callback, count=count)
