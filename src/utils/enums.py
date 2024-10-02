from enum import Enum, IntEnum


class IGNMatrixMode(str, Enum):
    """
    Mode for the IGN Matrix
        fixed - all 10 IGNs are present before program starts and remains unchanged throughout
        infer - <10 IGNs are present, the remaining IGNs will be inferred through the killfeed
    """
    FIXED = "fixed"
    INFER = "infer"


class CaptureMode(str, Enum):
    SCREENSHOT = "screenshot"
    VIDEOFILE  = "videofile"
    STREAM     = "stream"
    YOUTUBE    = "youtube"


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
