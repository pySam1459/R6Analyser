from enum import Enum


__all__ = [
    "IGNMatrixMode",
    "CaptureMode",
    "WinCondition"
]


class StrEnum(Enum):
    """Helper class to implement `from_string` for Enum's with String values"""
    @classmethod
    def from_string(cls, value: str):
        """Class method to convert a string to an Enum value, with validity checks."""
        for enum_member in cls:
            if enum_member.value == value:
                return enum_member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")

    def __str__(self):
        return self.value


class IGNMatrixMode(StrEnum):
    """
    Mode for the IGN Matrix
        fixed - all 10 IGNs are present before program starts and remains unchanged throughout
        infer - <10 IGNs are present, the remaining IGNs will be inferred through the killfeed
    """
    FIXED = "fixed"
    INFER = "infer"


class CaptureMode(StrEnum):
    SCREENSHOT = "SCREENSHOT"
    VIDEOFILE  = "VIDEOFILE"


class WinCondition(StrEnum):
    KILLED_OPPONENTS = "KilledOpponents"
    TIME             = "Time"
    DEFUSED_BOMB     = "DefusedBomb"
    DISABLED_DEFUSER = "DisabledDefuser"
    UNKNOWN          = "Unknown" 
