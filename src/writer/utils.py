from datetime import datetime
from pathlib import Path

from ignmatrix import IGNMatrix, Player
from history import HistoryRound, KFRecord
from utils.enums import Team


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
    team0, team1 = ignmat.get_teams()
    if hround.atk_side == Team.TEAM1:
        return team1 + team0
    else: ## if atk_side == Team0 or Unknown
        return team0 + team1

def is_valid_kill(record: KFRecord) -> bool:
    return record.player.team != record.target.team
