from dataclasses import dataclass as odataclass
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing import Any, Optional

from ignmatrix import IGNMatrix, Player
from utils import Timestamp, Scoreline
from utils.enums import Team, WinCondition


__all__ = [
    "History",
    "HistoryRound",
    "KFRecord"
]


@odataclass
class KFRecord:
    """
    Dataclass to record a player interaction
    who killed who, at what time, and if it was a headshot.
      player: killer, target: dead
    """
    player:   Player
    target:   Player
    time:     Timestamp
    headshot: bool

    def to_str(self, show_time=True) -> str:
        if self.headshot:
            kill_msg = f"{self.player.ign} -> (X) {self.target.ign}"
        else:
            kill_msg = f"{self.player.ign} -> {self.target.ign}"

        if show_time:
            kill_msg = f"{self.time}| {kill_msg}"

        return kill_msg

    def __eq__(self, other: 'KFRecord') -> bool:
        """Equality check only requires index, not time/headshot (A can only kill B once)"""
        return self.player.uid == other.player.uid and self.target.uid == other.target.uid

    def __hash__(self) -> int:
        return self.player.uid * 101 + self.target.uid

    __str__ = to_str
    __repr__ = to_str

    def model_dump(self) -> dict[str, Any]:
        return {
            "player":   self.player.ign,
            "target":   self.target.ign,
            "time":     str(self.time),
            "headshot": self.headshot,
        }


@odataclass
class TradedRecord(KFRecord):
    traded:    bool = False

    def __init__(self, record: KFRecord) -> None:
        super(TradedRecord, self).__init__(**vars(record))


@odataclass
class Disconnect:
    player: Player
    time: Timestamp


class HistoryRound(BaseModel):
    """Model containing all of the data gathered from 1 round"""
    round_number:        int

    scoreline:           Optional[Scoreline] = None
    atk_side:            Team                = Team.UNKNOWN
    bomb_planted_at:     Optional[Timestamp] = None
    disabled_defuser_at: Optional[Timestamp] = None
    round_end_at:        Optional[Timestamp] = None
    win_condition:       WinCondition        = WinCondition.UNKNOWN
    winner:              Team                = Team.UNKNOWN
    killfeed:            list[KFRecord]      = Field(default_factory=list)

    disconnects:         list[Disconnect]    = Field(default_factory=list, exclude=True)

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    @field_validator("killfeed", mode="before")
    @classmethod
    def validate_killfeed(cls, v: Any) -> list[KFRecord]:
        if not isinstance(v, list):
            raise ValueError("Killfeed is not a list")
        elif len(v) == 0:
            return list()
        elif all([isinstance(record, KFRecord) for record in v]):
            return v

        raise ValueError("Invalid Killfeed, some elements are non-KFRecord instances")

    @field_serializer("killfeed")
    def serialize_killfeed(self, kf: list[KFRecord]) -> list[dict[str,Any]]:
        return [record.model_dump() for record in kf]

    @field_serializer("bomb_planted_at", "disabled_defuser_at", "round_end_at")
    def serialize_timestamps(self, ts: Optional[Timestamp]) -> Optional[str]:
        if ts is None:
            return None
        return str(ts)
    
    def get_wincon(self, ignmat: IGNMatrix) -> WinCondition:
        """Returns the win condition from the round history"""
        if self.winner == Team.UNKNOWN or self.atk_side == Team.UNKNOWN:
            return WinCondition.UNKNOWN

        if self.bomb_planted_at is not None:
            if self.winner == self.atk_side:
                return WinCondition.DEFUSED_BOMB
            elif self.winner == self.atk_side.opp:
                return WinCondition.DISABLED_DEFUSER
        
        elif self.__is_killopp_wincon(ignmat):
            return WinCondition.KILLED_OPPONENTS
        
        elif (self.round_end_at is not None and
              0 <= self.round_end_at.to_int() <= 2):
            return WinCondition.TIME
        
        return WinCondition.UNKNOWN
    
    def __is_killopp_wincon(self, ignmat: IGNMatrix) -> bool:
        atk_team, def_team = ignmat.get_teams().order(self.atk_side)

        atk_deaths = len([rec
                          for rec in self.killfeed
                          if rec.target.team == self.atk_side and rec.target in atk_team])
        def_deaths = len([rec
                          for rec in self.killfeed
                          if rec.target.team == self.atk_side.opp and rec.target in def_team])

        return atk_deaths == len(atk_team) or def_deaths == len(def_team)
    
    def is_dead(self, pl: Player) -> bool:
        return any([record.target == pl for record in self.killfeed])


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
        ## TODO: optimize? replace with self.__cround and update on new_round()...
        return self.__round_data.get(self.__roundn, self.__phantom_round)
    
    def get_rounds(self) -> list[HistoryRound]:
        return list(self.__round_data.values())
    
    def get_first_round(self) -> HistoryRound:
        assert self.is_ready

        first_roundn = min(self.__round_data)
        return self.__round_data[first_roundn]

    def new_round(self, round_number: int) -> None:
        """This method should be called at the start of a new round"""
        self.__round_data[round_number] = HistoryRound(round_number=round_number)
        self.__roundn = round_number
    
    def __contains__(self, other: int | Scoreline) -> bool:
        if isinstance(other, int):
            return other in self.__round_data
        elif isinstance(other, Scoreline):
            return other.total+1 in self.__round_data
        return False
    
    def __len__(self) -> int:
        return len(self.__round_data)

    def fix_round(self) -> None:
        """Should be called by _fix_state, in-case program incorrectly thinks round ended"""
        self.cround.round_end_at = None

    def model_dump(self) -> dict:
        """Converts all game data recorded to json-handlable"""
        return {ridx: hround.model_dump() for ridx, hround in self.__round_data.items()}
