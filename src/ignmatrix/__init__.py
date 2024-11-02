from config import Config

from .base import IGNMatrix
from .fixed import IGNMatrixFixed
from .infer import IGNMatrixInfer
from .player import Player


__all__ = [
    "IGNMatrix",
    "Player",
    "create_ignmatrix"
]


def create_ignmatrix(cfg: Config) -> IGNMatrix:
    """Creates a new IGNMatrix object from a list of fixed IGNs and the IGNMatrix Mode"""
    if len(cfg.team0) + len(cfg.team1) == 10:
        return IGNMatrixFixed(cfg.team0, cfg.team1, cfg.ignmatrix.ratio_threshold)

    else:
        return IGNMatrixInfer(cfg.team0, cfg.team1, cfg.ignmatrix.ratio_threshold)
