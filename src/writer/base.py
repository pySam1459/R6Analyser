from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from config import Config
from history import History
from ignmatrix import IGNMatrix, Player
from utils.enums import SaveFileType


class Writer(ABC):
    _team0: list[Player]
    _team1: list[Player]

    def __init__(self, _type: SaveFileType, save_path: Path, config: Optional[Config]) -> None:
        self._type = _type
        self._save_path = save_path
        self._config = config

    @property
    def type(self) -> SaveFileType:
        return self._type

    @abstractmethod
    def write(self, history: History, ignmat: IGNMatrix) -> None:
        ...

    def _pre_write(self, history: History, ignmat: IGNMatrix) -> None:
        """Helper method which preps the data which the writer will save"""
        self._players = ignmat.get_players()
        history.update(ignmat)
        self._clean_history(history)

    def _clean_history(self, history: History) -> None:
        """Creates separate clean_killfeed and clean_death attributes to each HistoryRound with all players who aren't in `__players` removed"""
        indices = [pl.idx for pl in self._players]
        for round in history.get_rounds():
            round.clean_killfeed = [record for record in round.killfeed if record.player.idx in indices and record.target.idx in indices]
            round.clean_deaths   = [didx for didx in round.deaths if didx in indices]
