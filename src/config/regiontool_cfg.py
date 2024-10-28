from pathlib import Path
from re import match
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from typing import Any, Optional, Self

from settings import Settings
from utils import Timestamp, BBox_t
from utils.constants import TIMESTAMP_PATTERN
from utils.enums import CaptureMode, GameType

from .common import CaptureCommon
from .utils import validate_config


__all__ = [
    "RTConfig",
    "RTRegionsCFG",
    "create_regiontool_config"
]


class RTRegionParams(BaseModel):
    num_kf_lines: Optional[int] = None
    kf_buf:       Optional[int] = None
    kf_buf_mult:  Optional[float] = None

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


def create_regiontool_config(config_path: Path, settings: Settings) -> RTConfig | list[RTConfig]:
    return validate_config(RTConfig.model_validate, config_path, settings)
