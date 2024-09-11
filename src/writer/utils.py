from datetime import datetime
from pathlib import Path


def make_copyfile(save_path: Path) -> Path:
    """Creates a copy file based on the save_path"""
    parent = save_path.parent
    stem = save_path.stem
    suffix = save_path.suffix
    
    counter = 1
    while True:
        new_path = parent / f"{stem} ({counter}){suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def default_save_file(suffix="xlsx") -> str:
    now_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{now_time}.{suffix}"
