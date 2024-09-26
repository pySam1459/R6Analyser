import json
import random
from colorsys import hsv_to_rgb
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, Callable


# ----- HELPER FUNCTIONS -----
# Resource loaders
def load_file(file_path: Path) -> str:
    """Loads text from `file_path` and handles any errors"""
    if not file_path.exists():
        raise FileNotFoundError(f"'{file_path}' does not exist!")

    try:
        with open(file_path, "r", encoding="utf-8") as f_in:
            return f_in.read()

    except Exception as e:
        exit(f"ERROR: Exception occurred when trying to open {file_path}!\n{str(e)}")

def load_json(file_path: Path):
    """Loads json from `file_path` and handles any errors"""
    if not file_path.exists():
        raise FileNotFoundError(f"'{file_path}' does not exist!")

    try:
        with open(file_path, "r", encoding="utf-8") as f_in:
            return json.load(f_in)

    except Exception as e:
        exit(f"JSON ERROR: Exception occurred when trying to open {file_path}!\n{str(e)}")


# non-builtin functions
def ndefault(value: Any, default: Any) -> Any:
    """return value is None ? default : value"""
    if value is None: return default
    return value

T = TypeVar('T')
def transpose(matrix: list[list[T]]) -> list[list[T]]:
    """Transposes a 2d matrix"""
    return [list(row) for row in zip(*matrix)]

def recursive_union(src: dict[str,Any], add: dict[str,Any]):
    result = src.copy()
    for key, value in add.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = recursive_union(result[key], value)
        else:
            result[key] = value
    return result

def gen_random_colour() -> tuple[int,int,int]:
    h = random.random()
    s = random.uniform(0.5, 1.0)
    v = random.uniform(0.7, 1.0)
    
    r,g,b = hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

def argmax(arr: list[Any]) -> int:
    """https://github.com/cjohnson318/til/blob/main/python/argmax-without-numpy.md"""
    return max(range(len(arr)), key=lambda i: arr[i])

def flatten(arr: list[list[T]]) -> list[T]:
    return [el for inner in arr for el in inner]

def listsub(src: list[T], sub: list[T]) -> list[T]:
    """Removes the elements in sub from src"""
    return [v for v in src if v not in sub]

def gen_default_name() -> str:
    return datetime.now().strftime("r6analyser-%d_%m_%Y-%H:%M:%S")

def str2f(v: float) -> str:
    """Converts v to a 2f rounded string"""
    return f"{v:.2f}"

def perc_s(n: int|float, d: int|float) -> float:
    """Performs n / d, returning 0.0 on d == 0"""
    if d == 0:
        return 0.0
    return n / d * 100.0


# bounding box functions
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


def compute_rect_ratio(rect1: list[int], rect2: list[int]) -> float:
    """Computes Area(rect1) / Area(rect2)"""
    return (rect1[2] * rect1[3]) / (rect2[2] * rect2[3])


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
