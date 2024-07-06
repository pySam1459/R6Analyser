import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


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


class Config:
    DONT_SAVE = ["cfg_file_path", "name"]

    def __init__(self, _inital: dict, *, name: Optional[str] = None) -> None:
        for key, value in _inital.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, Config.__fixkey(key), value)

        if name is not None:
            self.name = name

    def __getitem__(self, key: str) -> Any:
        return getattr(self, Config.__fixkey(key))

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, Config.__fixkey(key), value)

    def __contains__(self, key: str) -> bool:
        return Config.__fixkey(key) in self.__dict__

    def get(self, key: str, _default: Any = None) -> Any:
        if key in self:
            return self.__dict__[key]
        return _default

    @staticmethod
    def __fixkey(key: str) -> str:
        return key.lower().replace(" ", "_")

    def __repr__(self) -> str:
        return self._repr(0)

    def _repr(self, indent: int) -> str:
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                items.append(f'{" " * indent}{key}:\n{value._repr(indent + 2)}')
            else:
                items.append(f'{" " * indent}{key}: {value}')
        return '\n'.join(items)
    
    def enumerate(self) -> list[str]:
        key_paths = []
        
        for key, value in self.__dict__.items():
            if type(value) == Config:
                key_paths += [f"{key}/{val_enum}" for val_enum in value.enumerate()]
            else:
                key_paths.append(key)

        return key_paths

    def to_dict(self) -> dict:
        func = lambda v: v.to_dict() if type(v) == Config else Config._to_dict(v)
        return {k: func(v) for k,v in self.__dict__.items()}

    @staticmethod
    def _to_dict(value: Any) -> Any:
        if isinstance(value, Enum):
            return str(value)
        else:
            return value

    def save(self, file_path: str) -> None:
        for key in Config.DONT_SAVE:
            self.__dict__.pop(key)
        with open(file_path, "w") as f_out:
            json.dump(self.to_dict(), f_out, indent=4)
    


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


class PhantomTqdm:
    total: Any
    n: Any

    def set_description_str(self, *args, **kwargs) -> None:
        ...
    def set_postfix_str(self, *args, **kwargs) -> None:
        ...
    def refresh(self) -> None:
        ...
    def close(self) -> None:
        ...


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


def compute_rinr(r1: list[int], r2: list[int]) -> float:
    # r1 and r2 are tuples of the form (xmin, ymin, xmax, ymax)
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    # Calculate intersection coordinates
    inter_x_min = max(x1, x2)
    inter_y_min = max(y1, y2)
    inter_x_max = min(x1 + w1, x2 + w2)
    inter_y_max = min(y1 + h1, y2 + h2)

    # Calculate width and height of intersection
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)

    # Calculate intersection area
    inter_area = inter_width * inter_height
    r1_area = w1 * h1

    return inter_area / r1_area if r1_area > 0 else 0


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
