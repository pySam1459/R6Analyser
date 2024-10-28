from typing import Optional

from utils.constants import IGN_CHARLIST
from utils.enums import IGNMatrixMode, Team

from .base import IGNMatrix
from .player import Player, AdaptivePlayer
from .utils import Teams


__all__ = ["IGNMatrixInfer"]


class IGNMatrixInfer(IGNMatrix):
    def __init__(self, team0: list[str], team1: list[str], leven_th: float) -> None:
        super(IGNMatrixInfer, self).__init__(IGNMatrixMode.INFER, team0, team1, leven_th)

        self.__mat: list[AdaptivePlayer] = []
    
    @property
    def charlist(self) -> str:
        return IGN_CHARLIST

    def get(self, pign: str, pteam: Team) -> Optional[Player]:
        if ((pl := self._ttable.check(pign, self._ign_threshold)) is not None and
             pl.team == pteam):
            return pl

        return self._mat_get(pign)

    def _mat_get(self, pign: str) -> Optional[AdaptivePlayer]:
        return next((ap for ap in self.__mat
                     if ap.contains(pign, self._ign_threshold)), None)
    
    def _mat_get_topK(self, k: int) -> list[AdaptivePlayer]:
        """sorted is more appropriate over heapq.nlargest as k ~= len(__heap)"""
        return sorted(self.__mat, key=lambda x: x.noccr, reverse=True)[:k]

    def get_teams(self) -> Teams:
        players = self._ttable.get_players()
        fxd_length = len(players)

        if fxd_length < 10:
            players.extend(self._mat_get_topK(10-fxd_length))

        return Teams([pl for pl in players if pl.team == Team.TEAM0],
                     [pl for pl in players if pl.team == Team.TEAM1])

    def evaluate(self, pign: str) -> float:
        table_eval = self._ttable.eval(pign)
        mat_eval = max([ap.evaluate(pign) for ap in self.__mat])
        return max(table_eval, mat_eval)


    def update_mat(self, pign: str, pteam: Team) -> None:
        """Records an interaction between pign and tign"""
        pl = self._ttable.check(pign, self._ign_threshold)
        if pl is not None: ## already known
            return

        pap = self._mat_get(pign)
        if pap is not None:
            pap.add(pign, self._ign_threshold)

        else: ## pign has not been seen, add to the matrix
            self.__mat.append(AdaptivePlayer(pign, pteam))
