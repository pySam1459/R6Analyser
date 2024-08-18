import sys
from tqdm import tqdm
from dataclasses import dataclass
from io import StringIO
from os.path import exists
from json import load as __json_load
from typing import Any, Optional


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


class ProgressBar:
    def __init__(self, verbose: int, disable: bool = False) -> None:
        bar_format = "{desc}|{bar}"
        if verbose == 3:
            bar_format += "|{postfix}"

        if not disable:
            self.__tqdmbar = tqdm(total=180, bar_format=bar_format)
        else:
            self.__tqdmbar = PhantomTqdm()

        self.__header = ""
        self.__time = ""
        self.__value = "-"

    def set_total(self, value: int) -> None:
        self.__tqdmbar.total = value

    def set_desc(self, value: str) -> None:
        self.__value = value
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {value} ", refresh=False)

    def set_time(self, value: Timestamp | int) -> None:
        assert type(value) in [Timestamp, int], f"Invalid value type: {type(value)}"
        if isinstance(value, Timestamp):
            self.__time = str(value)
            value = value.to_int()
        elif isinstance(value, int):
            self.__time = str(Timestamp.from_int(value))

        self.__tqdmbar.n = value
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {self.__value} ", refresh=False)

    def set_header(self, nround: int, s1: int, s2: int) -> None:
        self.__header = f"{nround}/{s1}:{s2}"
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {self.__value} ", refresh=False)
    
    def set_postfix(self, value: str) -> None:
        self.__tqdmbar.set_postfix_str(value, refresh=False)

    def refresh(self) -> None:
        self.__tqdmbar.refresh()

    def close(self) -> None:
        self.__tqdmbar.close()

    def reset(self) -> None:
        self.__tqdmbar.n = 180
        self.__tqdmbar.total = 180
        self.__value = "-"
        self.__time = "3:00"
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {self.__value} ")

    def bomb(self) -> None:
        self.set_time(45)
        self.set_total(45)
        self.refresh()

    @staticmethod
    def print(*prompt: object, sep: Optional[str] = " ", end: Optional[str] = "\n", flush: bool = False) -> None:
        """Replaces the builtin `print` function with one which works with the tqdm progress bar"""
        temp_out = StringIO()
        sys.stdout = temp_out
        print(*prompt, sep=sep, end=end, flush=flush)
        sys.stdout = sys.__stdout__
        tqdm.write(temp_out.getvalue(), end='')


# ----- HELPER FUNCTIONS -----
def load_json(file_path: str) -> dict:
    """Loads json from `file_path` and handles any errors"""
    if not exists(file_path):
        raise FileNotFoundError(f"'{file_path}' does not exist!")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f_in:
            return __json_load(f_in)

    except Exception as e:
        exit(f"JSON LOAD ERROR: Could not open {file_path}!\n{str(e)}")


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
