from config import Config
from utils.enums import IGNMatrixMode

from .base import IGNMatrix
from .fixed import IGNMatrixFixed
from .infer import IGNMatrixInfer
from .player import Player_t, Player


__all__ = [
    "IGNMatrix",
    "Player",
    "Player_t",
    "create_ignmatrix"
]


def create_ignmatrix(cfg: Config) -> IGNMatrix:
    """Creates a new IGNMatrix object from a list of fixed IGNs and the IGNMatrix Mode"""
    if len(cfg.team0) + len(cfg.team1) == 10:
        cfg.ign_mode = IGNMatrixMode.FIXED

    match cfg.ign_mode:
        case IGNMatrixMode.FIXED:
            return IGNMatrixFixed(cfg.team0, cfg.team1)
        case IGNMatrixMode.INFER:
            return IGNMatrixInfer(cfg.team0, cfg.team1)
