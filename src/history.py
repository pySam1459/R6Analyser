from dataclasses import dataclass, field
from typing import Optional

from ignmatrix import Player, IGNMatrix, IGNMatrixMode
from utils import Timestamp, StrEnum


__all__ = [
    "WinCondition",
    "History",
    "HistoryRound",
    "KFRecord"
]


class WinCondition(StrEnum):
    KILLED_OPPONENTS = "KilledOpponents"
    TIME             = "Time"
    DEFUSED_BOMB     = "DefusedBomb"
    DISABLED_DEFUSER = "DisabledDefuser"
    UNKNOWN          = "Unknown" 


@dataclass
class KFRecord:
    """
    Dataclass to record an player interaction, who killed who, at what time, and if it was a headshot.
      player: killer, target: dead
    """
    player: Player
    target: Player
    time: Timestamp
    headshot: bool = False
    
    def __eq__(self, other: 'KFRecord') -> bool:
        """Equality check only requires index, not time/headshot (A can only kill B once)"""
        return self.player.idx == other.player.idx and self.target.idx == other.target.idx
    
    def __hash__(self) -> int:
        return self.player.idx * 101 + self.target.idx
    
    def update(self, ign_mat: IGNMatrix) -> 'KFRecord':
        """Updates the details for the player/target with the current IGNMatrix data"""
        if ign_mat.mode == IGNMatrixMode.INFER:
            self.player = ign_mat.get_from_idx(self.player.idx) or self.player
            self.target = ign_mat.get_from_idx(self.target.idx) or self.player
        return self

    def to_json(self) -> dict:
        """Converts KFRecord object to json-handlable dictionary"""
        return {
            "time": str(self.time),
            "player": self.player.ign,
            "target": self.target.ign,
            "headshot": self.headshot
        }

    def to_str(self) -> str:
        headshot_str = "(X) " if self.headshot else ""
        return f"{self.time}| {self.player.ign} -> {headshot_str}{self.target.ign}"

    __repr__ = to_str


@dataclass
class HistoryRound:
    """Dataclass storing all of the data gathered from 1 round"""
    scoreline:           Optional[list[int]] = None
    atk_side:            Optional[int]       = None
    bomb_planted_at:     Optional[Timestamp] = None
    disabled_defuser_at: Optional[Timestamp] = None
    round_end_at:        Optional[Timestamp] = None
    win_condition:       WinCondition        = WinCondition.UNKNOWN
    winner:              Optional[int]       = None
    killfeed:            list[KFRecord]      = field(default_factory=list)
    deaths:              list[int]           = field(default_factory=list)

    clean_killfeed:      list[KFRecord]      = field(default_factory=list)
    clean_deaths:        list[int]           = field(default_factory=list)

    def to_json(self) -> dict:
        """Converts a HistoryRound object to a json-handlable dictionary"""
        return {
            "scoreline":            self.scoreline,
            "atk_side":             self.atk_side,
            "bomb_planted_at":      str(self.bomb_planted_at),
            "disabled_defuser_at":  str(self.disabled_defuser_at),
            "round_end_at":         str(self.round_end_at),
            "win_condition":        self.win_condition.value,
            "winner":               self.winner,
            "killfeed":             [kfr.to_json() for kfr in (self.clean_killfeed if self.clean_killfeed else self.killfeed)],
        }


class History:
    """
    This History class maintains a record of all game data recorded in a dictionary of game round: HistoryRound
    The program's round number counter is coupled to this class and only modified through the `new_round` method
    This class will only record round data after `new_round` is called for the first time
    """
    def __init__(self) -> None:
        self.__roundn = -1
        self.__round_data: dict[int, HistoryRound] = {}
        self.__phantom_round = HistoryRound()

    @property
    def is_ready(self) -> bool:
        return self.__roundn > 0
    
    @property
    def roundn(self) -> int:
        return self.__roundn
    
    def get_round(self, roundn: int) -> Optional[HistoryRound]:
        return self.__round_data.get(roundn, None)
    
    def get_rounds(self) -> list[HistoryRound]:
        return list(self.__round_data.values())
    def get_round_nums(self) -> list[int]:
        return list(self.__round_data.keys())
    def __contains__(self, key: int) -> bool:
        return key in self.__round_data

    @property
    def cround(self) -> HistoryRound:
        """
        Property to access the current HistoryRound
        Returns a Phantom round in-case history is not ready (__roundn <= 0)
        """
        return self.__round_data.get(self.__roundn, self.__phantom_round)

    def new_round(self, round_number: int) -> None:
        """This method should be called at the start of a new round"""
        self.__roundn = round_number
        self.__round_data[round_number] = HistoryRound()
    
    def fix_round(self) -> None:
        """Should be called by _fix_state, in-case program incorrectly thinks round ended"""
        self.cround.round_end_at = None
        
    def to_json(self) -> dict:
        """Converts all game data recorded to json-handlable"""
        return {ridx: round.to_json() for ridx, round in self.__round_data.items()}
    
    def update(self, ignmat: IGNMatrix) -> None:
        """Updates all of the KFRecords with the most up-to-date player information from the IGNMatrix"""
        for round in self.__round_data.values():
            for record in round.killfeed:
                record.update(ignmat)


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
