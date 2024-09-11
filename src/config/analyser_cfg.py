import os
from collections import Counter
from functools import partial
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, computed_field, model_validator, ConfigDict
from typing import Any, Optional, Self

from settings import Settings
from utils import load_file, gen_default_name, GameTypeRoundMap, BBox_t
from utils.enums import GameType, CaptureMode, IGNMatrixMode, Team, SaveFileType
from utils.constants import DEFAULT_SAVE_DIR, IGN_REGEX, RED, WHITE
from .utils import validate_config


__all__ = [
    "Config",
    "create_analyser_config"
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


class Defaults(BaseModel):
    screenshot_resize: int   = Field(default=4)
    screenshot_period: float = Field(default=0.5)

    save: SaveCfg

    max_rounds_map:      GameTypeRoundMap
    rounds_per_side_map: GameTypeRoundMap
    overtime_rounds_map: GameTypeRoundMap

    model_config = ConfigDict(extra="ignore")


class RegionsParser(BaseModel):
    timer:        BBox_t
    kf_line:      BBox_t

    num_kf_lines: int   = Field(default=3,   ge=0, le=4)
    kf_buf:       int   = Field(default=4,   ge=0)
    kf_buf_mult:  float = Field(default=1.4, ge=1.0)

    model_config = ConfigDict(extra="allow")

    @field_validator('timer', 'kf_line')
    @classmethod
    def validate_bounding_box(cls, v: Any):
        if v is None:
            return v
        if not isinstance(v, tuple) or len(v) != 4 or not all(isinstance(el, int) for el in v):
            raise ValueError("must be of length=4 and type Tuple[int, int, int, int]")
        if any(el < 0 for el in v):
            raise ValueError("elements must be positive integers")
        return v
    
    @computed_field
    @property
    def team1_score(self) -> BBox_t:
        return (self.timer[0] - self.timer[2]//2, self.timer[1], self.timer[2]//2, self.timer[3])

    @computed_field
    @property
    def team2_score(self) -> BBox_t:
        return (self.timer[0] + self.timer[2], self.timer[1], self.timer[2]//2, self.timer[3])

    @computed_field
    @property
    def team1_side(self) -> BBox_t:
        return (self.timer[0] - int(self.timer[2]*0.95), self.timer[1], self.timer[2]//2, self.timer[3])

    @computed_field
    @property
    def team2_side(self) -> BBox_t:
        return (self.timer[0] + int(self.timer[2]*1.45), self.timer[1], self.timer[2]//2, self.timer[3])

    @computed_field
    @property
    def kf_lines(self) -> list[BBox_t]:
        x, y, w, h = self.kf_line
        return [
            (x, y - int(h * self.kf_buf_mult * i) - self.kf_buf, w, h + self.kf_buf * 2)
            for i in range(self.num_kf_lines)
        ]


class CaptureParser(BaseModel):
    mode:    CaptureMode
    regions: RegionsParser
    file:    Optional[Path] = None

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


class SaveParser(BaseModel):
    file_type: SaveFileType   = Field(default=SaveFileType.XLSX)
    save_dir:  Optional[Path] = Field(default=DEFAULT_SAVE_DIR)
    path:      Optional[Path] = None

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if self.path is None:
            return self
        if not self.path.parent.exists():
            raise ValueError(f"Directory {self.path.parent} does not exist!")
        if not os.access(self.path, os.W_OK):
            raise ValueError(f"Permission Error: Cannot write to location {self.path}")

        return self
    
    @model_validator(mode="after")
    def path_override(self) -> Self:
        if self.path is None:
            return self

        self.save_dir = self.path.parent
        self.file_type = SaveFileType(self.path.suffix[1:])
        return self


class ConfigParser(BaseModel):
    name:        str = Field(default_factory=gen_default_name)
    config_path: Path

    ## required
    game_type:           GameType
    spectator:           bool
    capture:             CaptureParser
    team0:               list[str]     = Field(default_factory=list, min_length=0, max_length=5)
    team1:               list[str]     = Field(default_factory=list, min_length=0, max_length=5)
    
    ## inferred/optional
    max_rounds:          Optional[int] = Field(default=None, ge=1, le=25)
    rounds_per_side:     Optional[int] = Field(default=None, ge=1, le=6)
    overtime_rounds:     Optional[int] = Field(default=None, ge=0, le=3)
    last_winner:         Team          = Team.UNKNOWN

    ## defaults
    screenshot_resize:   int           = Field(ge=1, le=4)
    screenshot_period:   float         = Field(ge=0.0, le=2.0)
    max_rounds_map:      GameTypeRoundMap
    rounds_per_side_map: GameTypeRoundMap
    overtime_rounds_map: GameTypeRoundMap

    save: SaveParser
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

    model_config = ConfigDict(extra="ignore")


class Config(BaseModel):
    name:              str
    config_path:       Path

    game_type:         GameType
    spectator:         bool
    capture:           CaptureCfg
    team0:             list[str]
    team1:             list[str]

    ign_mode:          IGNMatrixMode = Field(exclude=True)

    last_winner:       Team          = Field(exclude=True)
    max_rounds:        int           = Field(exclude=True)
    rounds_per_side:   int           = Field(exclude=True)
    overtime_rounds:   int           = Field(exclude=True)

    screenshot_resize: int
    screenshot_period: float

    save:              SaveCfg
    debug:             DebugCfg      = Field(exclude=True)

    model_config = ConfigDict(extra="ignore", use_enum_values=True)


def _get_defaults(df_path: Path) -> dict[str, Any]:
    df_raw = load_file(df_path)

    try:
        df_config = Defaults.model_validate_json(df_raw)
        return df_config.model_dump()
    except Exception as e:
        print(f"{RED}CONFIG PARSE ERROR{WHITE} - Invalid DEFAULTS file\n{str(e)}")
        raise e


def _get_debug(dbg_path: Path) -> dict[str, Any]:
    df_raw = load_file(dbg_path)

    try:
        dbg_config = DebugCfg.model_validate_json(df_raw)
        return { "debug": dbg_config.model_dump() }
    except Exception as e:
        print(f"{RED}CONFIG PARSE ERROR{WHITE} - Invalid DEBUG file\n{str(e)}")
        raise e


def validate_analyser_dict(cfg: dict[str, Any],
                           df:  dict[str, Any],
                           dbg: dict[str, Any]) -> Config:

    cfg_parsed = ConfigParser.model_validate(df | cfg | dbg)
    return Config.model_validate(cfg_parsed.model_dump(exclude_none=True))


def create_analyser_config(config_path: Path, settings: Settings) -> Config | list[Config]:
    validate_func = partial(validate_analyser_dict,
                            df =_get_defaults(settings.defaults_filepath),
                            dbg=_get_debug(settings.debug_filepath))

    return validate_config(validate_func, config_path, settings)
