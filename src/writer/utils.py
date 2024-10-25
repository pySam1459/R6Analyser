from datetime import datetime
from functools import wraps
from pathlib import Path

from ignmatrix import IGNMatrix, Player
from history import HistoryRound, KFRecord
from utils.constants import RED, WHITE
from utils.enums import Team


def handle_write_errors(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except PermissionError as e:
            print(f"{RED}SAVE PERMISSION ERROR{WHITE} - An error occurred when attempting to save to file {self._save_path}\n{str(e)}")
        except OSError as e:
            print(f"{RED}SAVE OS ERROR{WHITE} - An error occurred when attempting to save to file {self._save_path}\n{str(e)}")
        except Exception as e:
            print(f"{RED}SAVE ERROR{WHITE} - An error occurred when attempting to save to file {self._save_path}\n{str(e)}")
    return wrapper


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


def get_players(hround: HistoryRound, ignmat: IGNMatrix) -> list[Player]:
    return ignmat.get_teams().combine(hround.atk_side)

def is_valid_kill(record: KFRecord) -> bool:
    return record.player.team != record.target.team
