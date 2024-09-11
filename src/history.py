from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer
from typing import Optional

from ignmatrix import Player
from utils import Timestamp, Scoreline
from utils.enums import WinCondition


__all__ = [
    "History",
    "HistoryRound",
    "KFRecord"
]


class KFRecord(BaseModel):
    """
    Dataclass to record an player interaction, who killed who, at what time, and if it was a headshot.
      player: killer, target: dead
    """
    player:   Player
    target:   Player
    time:     Timestamp
    headshot: bool = False

    model_config = ConfigDict(extra="ignore")

    def __eq__(self, other: 'KFRecord') -> bool:
        """Equality check only requires index, not time/headshot (A can only kill B once)"""
        return self.player.ign == other.player.ign and self.target.ign == other.target.ign

    def __hash__(self) -> int:
        return hash(f"{self.player.ign}>{self.target.ign}")

    def to_str(self, show_time=True) -> str:
        if self.headshot:
            kill_msg = f"{self.player.ign} -> (X) {self.target.ign}"
        else:
            kill_msg = f"{self.player.ign} -> {self.target.ign}"

        if show_time:
            kill_msg = f"{self.time}| {kill_msg}"

        return kill_msg

    __str__ = to_str
    __repr__ = to_str

    @field_serializer("time")
    def serialize_timestamps(self, ts: Timestamp) -> str:
        return str(ts)


class HistoryRound(BaseModel):
    """Model containing all of the data gathered from 1 round"""

    scoreline:           Optional[Scoreline] = None
    atk_side:            Optional[int]       = None
    bomb_planted_at:     Optional[Timestamp] = None
    disabled_defuser_at: Optional[Timestamp] = None
    round_end_at:        Optional[Timestamp] = None
    win_condition:       WinCondition        = WinCondition.UNKNOWN
    winner:              Optional[int]       = None
    killfeed:            list[KFRecord]      = Field(default_factory=list, exclude=True)
    deaths:              list[str]           = Field(default_factory=list, exclude=True)

    clean_killfeed:      list[KFRecord]      = Field(default_factory=list, exclude=True)
    clean_deaths:        list[str]           = Field(default_factory=list, exclude=True)

    model_config = ConfigDict(extra="ignore")

    @computed_field
    @property
    def out_feed(self) -> list[KFRecord]:
        return self.clean_killfeed or self.killfeed
    
    @field_serializer("bomb_planted_at", "disabled_defuser_at", "round_end_at")
    def serialize_timestamps(self, ts: Optional[Timestamp]) -> Optional[str]:
        if ts is None:
            return None
        return str(ts)


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
        self.__round_data[round_number] = HistoryRound() # type: ignore

    def fix_round(self) -> None:
        """Should be called by _fix_state, in-case program incorrectly thinks round ended"""
        self.cround.round_end_at = None

    def model_dump(self) -> dict:
        """Converts all game data recorded to json-handlable"""
        return {ridx: round.model_dump() for ridx, round in self.__round_data.items()}


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
