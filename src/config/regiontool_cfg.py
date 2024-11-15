from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from utils import BBox_t
from utils.enums import GameType

from .common import CaptureCommon
from .utils import validate_config


__all__ = [
    "RTConfig",
    "RTRegionsCFG",
    "create_regiontool_config"
]


class RTRegionParams(BaseModel):
    num_kf_lines: Optional[int]   = None
    kf_buf:       Optional[int]   = None
    score_width:  Optional[float] = None
    side_width:   Optional[float] = None
    t1_offset:    Optional[int]   = None

    model_config = ConfigDict(extra="ignore")


class RTRegionsCFG(RTRegionParams):
    timer:   Optional[BBox_t] = None
    kf_line: Optional[BBox_t] = None

    model_config = ConfigDict(extra="ignore")


class RTCaptureCFG(CaptureCommon):
    regions: Optional[RTRegionsCFG] = None

    model_config = ConfigDict(extra="forbid")


class RTConfig(BaseModel): ## Region tool config
    config_path: Path = Field(exclude=True)

    name:        Optional[str] = None
    game_type:   GameType
    spectator:   bool
    capture:     RTCaptureCFG

    model_config = ConfigDict(extra="allow", use_enum_values=True)


def create_regiontool_config(config_path: Path) -> RTConfig | list[RTConfig]:
    return validate_config(RTConfig.model_validate, config_path)
