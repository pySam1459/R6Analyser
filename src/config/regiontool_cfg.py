from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Optional, Self

from settings import Settings
from utils import BBox_t
from utils.enums import CaptureMode, GameType
from .utils import validate_config


__all__ = [
    "RTConfig",
    "create_regiontool_config"
]


class RTRegionsCFG(BaseModel):
    timer:   Optional[BBox_t] = None
    kf_line: Optional[BBox_t] = None

    model_config = ConfigDict(extra="forbid")


class RTCaptureCFG(BaseModel):
    mode:    CaptureMode
    regions: Optional[RTRegionsCFG] = None
    file:    Optional[Path] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def file_exists(self) -> Self:
        if self.mode == CaptureMode.VIDEOFILE:
            if self.file is None:
                raise ValueError(f"videofile capture mode requires `file` field to be specified")

            elif not self.file.exists():
                raise ValueError(f"Video file {self.file} does not exist")
        return self


class RTConfig(BaseModel): ## Region tool config
    config_path: Path

    game_type:   GameType
    spectator:   bool
    capture:     RTCaptureCFG

    model_config = ConfigDict(extra="allow", use_enum_values=True)


def create_regiontool_config(config_path: Path, settings: Settings) -> RTConfig | list[RTConfig]:
    return validate_config(RTConfig.model_validate, config_path, settings)
