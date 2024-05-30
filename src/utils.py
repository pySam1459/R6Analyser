from enum import Enum
from dataclasses import dataclass
from typing import Any


class StrEnum(Enum):
    """Helper class to implement `from_string` for Enum's with String values"""
    @classmethod
    def from_string(cls, value: str):
        """
        Class method to convert a string to an Enum value, with validity checks.
        """
        for enum_member in cls:
            if enum_member.value == value:
                return enum_member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


@dataclass
class Timestamp:
    """Helper dataclass to deal with the timer"""
    minutes: int
    seconds: int

    def __sub__(self, other: 'Timestamp') -> int:
        return self.to_int() - other.to_int()

    def to_int(self) -> int:
        return self.minutes * 60 + self.seconds

    @staticmethod
    def from_int(num: int) -> 'Timestamp':
        m = int(num / 60)
        s = num - 60*m
        return Timestamp(m, s)
    
    def __str__(self) -> str:
        if self.seconds < 0: return f"{self.minutes}:{self.seconds:03}"
        else: return f"{self.minutes}:{self.seconds:02}"

    __repr__ = __str__


@dataclass
class SaveFile:
    """Helper dataclass to handle SaveFile paths"""
    filename: str
    ext: str

    def __str__(self) -> str:
        return f"{self.filename}.{self.ext}"
    __repr__ = __str__

    def copy(self) -> 'SaveFile':
        return SaveFile(self.filename, self.ext)


# ----- HELPER FUNCTIONS -----
def ndefault(value, default):
    if value is None: return default
    return value

def transpose(matrix: list[list[Any]]) -> list[list[Any]]:
    return [list(row) for row in zip(*matrix)]


def rect_collision(r1: list[int], r2: list[int]) -> bool:
    return r1[0] <= r2[0]+r2[2] and r1[1] <= r2[1]+r2[3] \
            and r2[0] <= r1[0]+r1[2] and r2[1] <= r1[1]+r1[3]


def bbox_to_rect(bbox: list[list[int]]) -> list[int]:
    tl, tr, _, bl = bbox
    return [int(tl[0]), int(tl[1]), int(tr[0]-tl[0]), int(bl[1]-tl[1])]


def box_collision(b1: list[list[int]], b2: list[list[int]]) -> bool:
    return rect_collision(bbox_to_rect(b1), bbox_to_rect(b2))


def rect_proximity(r1: list[int], r2: list[int]) -> float:
    if not (r1[1] <= r2[1]+r2[3]/2 <= r1[1]+r1[3]): return float("inf")
    return r2[0] - (r1[0]+r1[2])


def point_in_rect(point: list[int], rect: list[int]) -> bool:
    return rect[0] <= point[0] <= rect[0]+rect[2] and rect[1] <= point[1] <= rect[1]+rect[3]
