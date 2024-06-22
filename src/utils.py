from enum import Enum
from dataclasses import dataclass
from typing import Any


class StrEnum(Enum):
    """Helper class to implement `from_string` for Enum's with String values"""
    @classmethod
    def from_string(cls, value: str):
        """Class method to convert a string to an Enum value, with validity checks."""
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
def ndefault(value: Any, default: Any) -> Any:
    """return value is None ? default : value"""
    if value is None: return default
    return value

def transpose(matrix: list[list[Any]]) -> list[list[Any]]:
    """Transposes a 2d matrix"""
    return [list(row) for row in zip(*matrix)]


def rect_collision(r1: list[int], r2: list[int]) -> bool:
    """Checks if 2 [x,y,w,h] rectangles are colliding"""
    return r1[0] <= r2[0]+r2[2] and r1[1] <= r2[1]+r2[3] \
            and r2[0] <= r1[0]+r1[2] and r2[1] <= r1[1]+r1[3]


def bbox_to_rect(bbox: list[list[int]]) -> list[int]:
    """Converts from a EasyOCR bounding box [TL, TR, BR, BL] to [x,y,w,h] rectangle"""
    tl, tr, _, bl = bbox
    return [int(tl[0]), int(tl[1]), int(tr[0]-tl[0]), int(bl[1]-tl[1])]


def box_collision(b1: list[list[int]], b2: list[list[int]]) -> bool:
    """Checks whether 2 EasyOCR formatted bounding-boxes are colliding"""
    return rect_collision(bbox_to_rect(b1), bbox_to_rect(b2))


def rect_proximity(r1: list[int], r2: list[int]) -> float:
    """Computes the distance between the right side of r1 and left side of r2"""
    if not (r1[1] <= r2[1]+r2[3]/2 <= r1[1]+r1[3]): return float("inf")
    return r2[0] - (r1[0]+r1[2])


def point_in_rect(point: list[int], rect: list[int]) -> bool:
    """Checks if a 2d point is inside a rectangle"""
    return rect[0] <= point[0] <= rect[0]+rect[2] and rect[1] <= point[1] <= rect[1]+rect[3]


def compute_iou(rect1: list[int], rect2: list[int]) -> float:
    """Computes the Intersection over Union of 2 rectangles"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    rect1_area = w1 * h1
    rect2_area = w2 * h2

    union_area = rect1_area + rect2_area - intersection_area
    iou = intersection_area / union_area

    return iou


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
