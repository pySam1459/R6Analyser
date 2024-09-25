from typing import TypeAlias

from .enums import GameType


BBox_t: TypeAlias = tuple[int, int, int, int]
GameTypeRoundMap: TypeAlias = dict[GameType, int]
