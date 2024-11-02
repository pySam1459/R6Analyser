import os
from functools import partial
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Any, Optional, Self, Type

from settings import Settings
from utils import Timestamp, load_file, recursive_union, gen_default_name, check_duplicates_werr, BBox_t
from utils.enums import GameType, CaptureMode, Team, SaveFileType
from utils.constants import *
from params import *

from .common import CaptureCommon
from .region_models import TimerRegion, KFLineRegion
from .utils import validate_config


__all__ = [
    "Config",
    "create_analyser_config",
]

GAME_MAPS_ATTR = "_game_map"


class SaveCfg(BaseModel):
    file_type: SaveFileType
    save_dir:  Path
    path:      Optional[Path] = None

    model_config = ConfigDict(extra="ignore")


class _GameSettings(BaseModel):
    max_rounds:      dict[str, int]
    rounds_per_side: dict[str, int]
    overtime_rounds: dict[str, int]

    model_config = ConfigDict(extra="ignore")


class _GameParams(BaseModel):
    defuser_timer: int = Field(default=45, ge=1,   le=60)

    model_config = ConfigDict(extra="ignore")


class Defaults(BaseModel):
    save:      SaveCfg
    scheduler: SchedulerParams
    scoreline: ScorelineParams
    ignmatrix: IGNMatrixParams
    ocr:       OCRParams
    game:      _GameParams

    model_config = ConfigDict(extra="ignore")


class RegionsParser(TimerRegion, KFLineRegion):
    model_config = ConfigDict(extra="allow")


class CaptureParser(CaptureCommon):
    regions: RegionsParser

    model_config = ConfigDict(extra="forbid")


class SaveParser(BaseModel):
    file_type: SaveFileType   = SaveFileType.XLSX
    save_dir:  Path           = DEFAULT_SAVE_DIR
    path:      Optional[Path] = None

    @model_validator(mode="after")
    def path_override(self) -> Self:
        if self.path is None:
            return self

        self.save_dir = self.path.parent
        self.file_type = SaveFileType(self.path.suffix[1:])
        return self
    
    @model_validator(mode="after")
    def make_save_dir(self) -> Self:
        os.makedirs(self.save_dir, exist_ok=True)
        return self


class ConfigParser(BaseModel):
    name:                Optional[str] = Field(default=None, min_length=1, max_length=64)
    config_path:         Path

    game_type:           GameType
    spectator:           bool
    capture:             CaptureParser
    team0:               list[str]     = Field(default_factory=list, min_length=0, max_length=5)
    team1:               list[str]     = Field(default_factory=list, min_length=0, max_length=5)

    last_winner:         Team          = Team.UNKNOWN

    game:                GameParams
    save:                SaveParser
    scheduler:           SchedulerParams
    scoreline:           ScorelineParams
    ignmatrix:           IGNMatrixParams
    ocr:                 OCRParams
    debug:               DebugParams

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    @model_validator(mode="before")
    @classmethod
    def check_custom_params_exist(cls, values: dict[str, Any]):
        game_type   = values.get('game_type', None)
        game_params = values.get('game', {})

        ## game-settings.json
        gset_maps: dict[str, dict[str, int]] = values.pop(GAME_MAPS_ATTR)

        game_params_set = set(game_params)
        dc_set = set(GameParams.__dataclass_fields__)
        missing_params = (dc_set ^ game_params_set) - game_params_set

        if game_type is None:
            raise ValueError("No game_type parameter provided")
        elif game_type not in GameType._value2member_map_:
            raise ValueError(f"Invalid game_type: {game_type}")

        ## check if game_params isn't missing anything when gameType is custom
        if game_type == GameType.CUSTOM:
            if len(missing_params) != 0:
                missing_params_s = ', '.join(missing_params)
                raise ValueError(f"valid game parameters must be provided when game_type is 'custom'\nPlease provide: {missing_params_s}")
            return values

        ## create game_params from game-settings maps
        for prop, pmap in gset_maps.items():
            assert game_type in pmap, f"Game settings has been modified, {prop} has no {game_type}"
            game_params[prop] = pmap[game_type]
        
        return values | {"game": game_params}

    @field_validator('team0', "team1")
    @classmethod
    def validate_teams(cls, ign_list: list[str]) -> list[str]:
        invalid_igns = [ign for ign in ign_list if not IGN_REGEX.fullmatch(ign)]
        if len(invalid_igns) > 0:
            raise ValueError("Invalid IGNs: {}".format(", ".join(invalid_igns)))

        check_duplicates_werr(ign_list, "Duplicate IGN/s: {}")
        return ign_list

    @model_validator(mode="after")
    def check_igns_duplicates(self) -> Self:
        check_duplicates_werr(self.team0 + self.team1, "Duplicate IGN/s: {}")
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
    score_width:  float
    side_width:   float
    t1_offset:    int

    team0_score:  BBox_t
    team1_score:  BBox_t
    team0_side:   BBox_t
    team1_side:   BBox_t

    model_config = ConfigDict(extra="ignore")


class CaptureCfg(BaseModel):
    mode:    CaptureMode
    regions: RegionsCfg
    file:    Optional[Path]
    url:     Optional[str]
    start:   Optional[Timestamp]

    @field_validator("start", mode="before")
    @classmethod
    def parse_start(cls, v: Any) -> Optional[Timestamp]:
        if v is None or isinstance(v, Timestamp):
            return v
        elif isinstance(v, str):
            return Timestamp.from_str(v)
        raise ValueError(f"Cannot parse capture.start, {v}")

    model_config = ConfigDict(extra="ignore")


class Config(BaseModel):
    name:                str
    config_path:         Path = Field(exclude=True)

    game_type:           GameType
    spectator:           bool
    capture:             CaptureCfg
    team0:               list[str]
    team1:               list[str]

    last_winner:         Team

    game:                GameParams
    save:                SaveCfg
    scheduler:           SchedulerParams
    scoreline:           ScorelineParams
    ignmatrix:           IGNMatrixParams
    ocr:                 OCRParams

    debug:               DebugParams

    model_config = ConfigDict(extra="ignore", use_enum_values=True)


def _load_model_dict(path: Path, model_cls: Type[BaseModel]) -> dict[str, Any]:
    if path.exists():
        raw_data = load_file(path)
    else:
        raw_data = "{}\n"

    try:
        config = model_cls.model_validate_json(raw_data)
        return config.model_dump()
    except Exception as e:
        print(f"{RED}CONFIG PARSE ERROR{WHITE} - Invalid settings file at {path}\n{str(e)}")
        raise e


def validate_analyser_dict(cfg:  dict[str, Any],
                           base: dict[str, Any]) -> Config:

    ## config propreties recursively insert/replace into defaults
    cfg = recursive_union(base, cfg)

    ## debug props are unioned on top
    cfg_parsed = ConfigParser.model_validate(cfg)
    return Config.model_validate(cfg_parsed.model_dump())


def create_analyser_config(config_path: Path, settings: Settings) -> Config | list[Config]:
    defaults = _load_model_dict(settings.defaults_filepath, Defaults)

    game_maps = {GAME_MAPS_ATTR: _load_model_dict(settings.gsettings_filepath, _GameSettings)}
    debug     = {"debug": _load_model_dict(settings.debug_filepath, DebugParams)}

    base = defaults | game_maps | debug
    validate_func = partial(validate_analyser_dict, base=base)
    return validate_config(validate_func, config_path, settings)
