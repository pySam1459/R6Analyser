import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from sys import exit
from typing import Any, TypeVar, Optional, Callable


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
T = TypeVar('T')
def ndefault(value: Any, default: Any) -> Any:
    """return value is None ? default : value"""
    if value is None: return default
    return value

def filter_none(arr: list[T | None]) -> list[T]:
    return [el for el in arr if el is not None]

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


def argmax(arr: list[Any], key: Optional[Callable] = None) -> int:
    if key is None:
        return max(range(len(arr)), key=lambda i: arr[i])
    else:
        return max(range(len(arr)), key=lambda i: key(arr[i]))

def argmin(arr: list[Any], key: Optional[Callable] = None) -> int:
    if key is None:
        return min(range(len(arr)), key=lambda i: arr[i])
    else:
        return min(range(len(arr)), key=lambda i: key(arr[i]))

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

def fmt_s(*args):
    return str2f(perc_s(*args))

def mode_count(arr: list[T]) -> tuple[T, int]:
    args = [(x, arr.count(x)) for x in set(arr)]
    return max(args)


def check_duplicates_werr(values: list[Any], err_msg: str) -> list[Any]:
    dups = [v for v,c in Counter(values).items() if c > 1]
    if len(dups) > 0:
        raise ValueError(err_msg.format(", ".join(dups)))
    return dups