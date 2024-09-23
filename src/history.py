from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator, model_serializer
from typing import Any, Optional, Generator

from ignmatrix import Player
from utils import Timestamp, Scoreline
from utils.enums import Team, WinCondition


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

    @property
    def is_valid(self) -> bool:
        """If team for both player and target is known"""
        return self.player.team != Team.UNKNOWN and self.target.team != Team.UNKNOWN

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


class _KillFeed(list[KFRecord]):
    def __iter__(self) -> Generator[KFRecord, None, None]:
        for record in super().__iter__():
            if record.is_valid:
                yield record


@dataclass
class Disconnect:
    player: Player
    time: Timestamp


class HistoryRound(BaseModel):
    """Model containing all of the data gathered from 1 round"""
    round_number:        int

    scoreline:           Optional[Scoreline] = None
    atk_side:            Optional[Team]      = None
    bomb_planted_at:     Optional[Timestamp] = None
    disabled_defuser_at: Optional[Timestamp] = None
    round_end_at:        Optional[Timestamp] = None
    win_condition:       WinCondition        = WinCondition.UNKNOWN
    winner:              Team                = Team.UNKNOWN
    __killfeed:          _KillFeed           = PrivateAttr(default_factory=_KillFeed)

    disconnects:         list[Disconnect]    = Field(default_factory=list)
    deaths:              list[Player]        = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @property
    def killfeed(self) -> _KillFeed:
        return self.__killfeed

    @model_serializer(mode='wrap')
    def serialize_model(self, serializer):
        data = serializer(self)
        data['killfeed'] = [record.model_dump() for record in self.killfeed]
        return data

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
        self.__phantom_round = HistoryRound(round_number=-1)

    @property
    def is_ready(self) -> bool:
        return self.__roundn > 0

    @property
    def roundn(self) -> int:
        return self.__roundn

    @property
    def cround(self) -> HistoryRound:
        """
        Property to access the current HistoryRound
        Returns a Phantom round in-case history is not ready (__roundn <= 0)
        """
        return self.__round_data.get(self.__roundn, self.__phantom_round)
    
    def get_rounds(self) -> list[HistoryRound]:
        return list(self.__round_data.values())
    
    def get_first_round(self) -> HistoryRound:
        assert self.is_ready

        first_roundn = min(self.__round_data.keys())
        return self.__round_data[first_roundn]

    def new_round(self, round_number: int) -> None:
        """This method should be called at the start of a new round"""
        self.__round_data[round_number] = HistoryRound(round_number=round_number)
        self.__roundn = round_number
    
    def __contains__(self, other: int) -> bool:
        return other in self.__round_data

    def fix_round(self) -> None:
        """Should be called by _fix_state, in-case program incorrectly thinks round ended"""
        self.cround.round_end_at = None

    def model_dump(self) -> dict:
        """Converts all game data recorded to json-handlable"""
        return {ridx: hround.model_dump() for ridx, hround in self.__round_data.items()}
