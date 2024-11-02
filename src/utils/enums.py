from enum import Enum, IntEnum, auto


class CaptureMode(str, Enum):
    SCREENSHOT = "screenshot"
    VIDEOFILE  = "videofile"
    YOUTUBE    = "youtube"
    TWITCH     = "twitch"


class CaptureTimeType(Enum):
    TIME = auto()
    FPS  = auto()


class WinCondition(str, Enum):
    KILLED_OPPONENTS = "KilledOpponents"
    TIME             = "Time"
    DEFUSED_BOMB     = "DefusedBomb"
    DISABLED_DEFUSER = "DisabledDefuser"
    UNKNOWN          = "Unknown" 


class Team(IntEnum):
    UNKNOWN = -1
    TEAM0   =  0
    TEAM1   =  1

    @property
    def opp(self) -> 'Team':
        assert self != Team.UNKNOWN
        return Team(1-self.value)


class GameType(str, Enum):
    COMP       = "comp"
    SCRIM      = "scrim"
    RANKED     = "ranked"
    STANDARD   = "standard"
    CUSTOM     = "custom"


class SaveFileType(str, Enum):
    XLSX = "xlsx"
    JSON = "json"
