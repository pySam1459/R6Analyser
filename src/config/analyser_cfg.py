import os
from collections import Counter
from functools import partial
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, computed_field, model_validator, ConfigDict
from typing import Any, Optional, Self, Type

from settings import Settings
from utils import load_file, recursive_union, gen_default_name, GameTypeRoundMap, BBox_t
from utils.enums import GameType, CaptureMode, IGNMatrixMode, Team, SaveFileType
from utils.constants import DEFAULT_SAVE_DIR, IGN_REGEX, RED, WHITE

from .region_models import TimerRegion, KFLineRegion
from .utils import validate_config, VALID_URLS


__all__ = [
    "Config",
    "create_analyser_config",
]


class DebugCfg(BaseModel):
    config_keys:    bool = False
    red_percentage: bool = False
    headshot_perc:  bool = False
    infer_time:     bool = False

    model_config = ConfigDict(extra="ignore")


class SaveCfg(BaseModel):
    file_type: SaveFileType
    save_dir:  Path
    path:      Optional[Path] = Field(default=None)

    model_config = ConfigDict(extra="ignore")


class CaptureDefaults(BaseModel):
    scale_by: float
    period:  float


class Defaults(BaseModel):
    capture: CaptureDefaults
    save:    SaveCfg

    max_rounds_map:      GameTypeRoundMap
    rounds_per_side_map: GameTypeRoundMap
    overtime_rounds_map: GameTypeRoundMap

    defuser_timer: int

    model_config = ConfigDict(extra="ignore")


class RegionsParser(TimerRegion, KFLineRegion):
    model_config = ConfigDict(extra="allow")


class CaptureParser(BaseModel):
    mode:    CaptureMode
    regions: RegionsParser
    file:    Optional[Path] = None
    url:     Optional[str] = None

    scale_by: float = Field(default=2, ge=0.5, le=8)
    period:   float = Field(default=0.5, ge=0.0, le=2.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def file_exists(self) -> Self:
        if self.mode == CaptureMode.VIDEOFILE:
            if self.file is None:
                raise ValueError(f"videofile capture mode requires `file` field to be specified")
            elif not self.file.exists():
                raise ValueError(f"Video file {self.file} does not exist")
            elif not os.access(self.file, os.R_OK):
                raise ValueError(f"Permission Error: Invalid permissions to read video file {self.file}")
        return self
    
    @model_validator(mode="after")
    def validate_url(self) -> Self:
        if self.mode not in [CaptureMode.YOUTUBE, CaptureMode.TWITCH]:
            return self
        
        if self.url is None:
            raise ValueError(f"{self.mode.value} capture mode required `url` field to be specified")

        if self.mode not in VALID_URLS:
            return self

        valid_urls = VALID_URLS[self.mode]
        for vurl in valid_urls:
            if self.url.startswith(vurl):
                return self
        else:
            valid_urls_str = ", ".join(valid_urls)
            raise ValueError(f"Invalid {self.mode.value} URL: {self.url}\nIt must start with {valid_urls_str}")


class SaveParser(BaseModel):
    file_type: SaveFileType   = SaveFileType.XLSX
    save_dir:  Path           = DEFAULT_SAVE_DIR
    path:      Optional[Path] = None

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if self.path is None:
            return self
        if not self.path.parent.exists():
            raise ValueError(f"Directory {self.path.parent} does not exist!")
        return self
    
    @model_validator(mode="after")
    def path_override(self) -> Self:
        if self.path is None:
            return self

        self.save_dir = self.path.parent
        self.file_type = SaveFileType(self.path.suffix[1:])
        return self


class ConfigParser(BaseModel):
    name:        Optional[str]         = Field(default=None, min_length=1, max_length=64)
    config_path: Path

    ## required
    game_type:           GameType
    spectator:           bool
    capture:             CaptureParser
    team0:               list[str]     = Field(default_factory=list, min_length=0, max_length=5)
    team1:               list[str]     = Field(default_factory=list, min_length=0, max_length=5)

    save: SaveParser
    
    ## inferred/optional
    max_rounds:          Optional[int] = Field(default=None, ge=1, le=25)
    rounds_per_side:     Optional[int] = Field(default=None, ge=1, le=6)
    overtime_rounds:     Optional[int] = Field(default=None, ge=0, le=3)
    last_winner:         Team          = Team.UNKNOWN

    ## defaults
    max_rounds_map:      GameTypeRoundMap
    rounds_per_side_map: GameTypeRoundMap
    overtime_rounds_map: GameTypeRoundMap

    defuser_timer: int

    ## debug
    debug: DebugCfg

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    @computed_field(return_type=IGNMatrixMode)
    @property
    def ign_mode(self) -> IGNMatrixMode:
        if len(self.team0) + len(self.team1) == 10:
            return IGNMatrixMode.FIXED
        else:
            return IGNMatrixMode.INFER


    @field_validator('team0', "team1")
    @classmethod
    def validate_teams(cls, ign_list: list[str]) -> list[str]:
        invalid_igns = [ign for ign in ign_list if not IGN_REGEX.fullmatch(ign)]
        if len(invalid_igns) > 0:
            raise ValueError("Invalid IGNs: {}".format(", ".join(invalid_igns)))

        ConfigParser.check_duplicates(ign_list, "Duplicate IGN/s: {}")
        return ign_list

    @model_validator(mode="after")
    def check_igns_duplicates(self) -> Self:
        ConfigParser.check_duplicates(self.team0 + self.team1, "Duplicate IGN/s: {}")
        return self
    
    @staticmethod
    def check_duplicates(values: list[Any], err_msg: str) -> list[Any]:
        dups = [v for v,c in Counter(values).items() if c > 1]
        if len(dups) > 0:
            raise ValueError(err_msg.format(", ".join(dups)))
        return dups
 
    @model_validator(mode="after")
    def infer_round_info(self) -> Self:
        prop_maps = {
            "max_rounds": self.max_rounds_map,
            "rounds_per_side": self.rounds_per_side_map,
            "overtime_rounds": self.overtime_rounds_map
        }
        for prop, round_map in prop_maps.items():
            if getattr(self, prop) is not None:
                continue

            if self.game_type == GameType.CUSTOM:
                raise ValueError(f"Custom gametypes must have `{prop}` provided")

            setattr(self, prop, round_map.get(self.game_type))

        return self
    
    @model_validator(mode="after")
    def create_name(self) -> Self:
        if self.name is not None:
            return self
        
        if self.capture.mode == CaptureMode.VIDEOFILE and self.capture.file is not None:
            self.name = self.capture.file.stem
            return self
        
        self.name = gen_default_name()
        return self


class RegionsCfg(BaseModel):
    timer:        BBox_t
    kf_line:      BBox_t
    kf_lines:     list[BBox_t]

    num_kf_lines: int
    kf_buf:       int

    team1_score:  BBox_t
    team2_score:  BBox_t
    team1_side:   BBox_t
    team2_side:   BBox_t

    model_config = ConfigDict(extra="ignore")


class CaptureCfg(BaseModel):
    mode:    CaptureMode
    regions: RegionsCfg
    file:    Optional[Path] = None
    url:     Optional[str] = None

    scale_by: float
    period:  float

    model_config = ConfigDict(extra="ignore")


class Config(BaseModel):
    name:              str
    config_path:       Path

    game_type:         GameType
    spectator:         bool
    capture:           CaptureCfg
    team0:             list[str]
    team1:             list[str]

    save:              SaveCfg

    ign_mode:          IGNMatrixMode = Field(exclude=True)

    last_winner:       Team          = Field(exclude=True)
    max_rounds:        int           = Field(exclude=True)
    rounds_per_side:   int           = Field(exclude=True)
    overtime_rounds:   int           = Field(exclude=True)

    defuser_timer:     int           = Field(exclude=True)

    debug:             DebugCfg      = Field(exclude=True)

    model_config = ConfigDict(extra="ignore", use_enum_values=True)


def _load_model_dict(path: Path, model_cls: Type[BaseModel]) -> dict[str, Any]:
    raw_data = load_file(path)
    try:
        config = model_cls.model_validate_json(raw_data)
        return config.model_dump()
    except Exception as e:
        print(f"{RED}CONFIG PARSE ERROR{WHITE} - Invalid config file at {path}\n{str(e)}")
        raise e


def validate_analyser_dict(cfg: dict[str, Any],
                           df:  dict[str, Any],
                           dbg: dict[str, Any]) -> Config:

    cfg = recursive_union(df, cfg)
    cfg_parsed = ConfigParser.model_validate(cfg | dbg)
    return Config.model_validate(cfg_parsed.model_dump(exclude_none=True))


def create_analyser_config(config_path: Path, settings: Settings) -> Config | list[Config]:
    df = _load_model_dict(settings.defaults_filepath, Defaults)
    dbg = {"debug": _load_model_dict(settings.debug_filepath, DebugCfg)}

    validate_func = partial(validate_analyser_dict, df=df, dbg=dbg)
    return validate_config(validate_func, config_path, settings)
