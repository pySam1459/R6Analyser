from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass
class OCRParams:
    sl_scalex:        float = 0.4
    sl_scaley:        float = 0.5
    sl_bbox_zscore:   float = 2

    inf_th:           float = 0.75

    hue_offset:       int   = 38
    hue_std:          float = 13
    sat_std:          float = 4
    col_zscore:       float = 2.5

    seg_min_area:     float = 0.025
    seg_mask_th:      float = 0.25
    seg_min_width:    float = 0.1
    seg_black_th:     int   = 24
    seg_black_clip:   int   = 4
    seg_dist_th:      float = 0.5

    hs_wide_sf:       float = 1.35
    hs_th:            float = 0.5


class SchedulerParams(BaseModel):
    scoreline: int = Field(default=1000, ge=0)
    timer:     int = Field(default=500,  ge=0)
    killfeed:  int = Field(default=300,  ge=0)

    model_config = ConfigDict(extra="ignore")


@dataclass
class ScorelineParams:
    majority_threshold: int = Field(default=3, ge=1, le=10)
    ballet_box_size:    int = Field(default=5, ge=1)


@dataclass
class IGNMatrixParams:
    ratio_threshold: float  = Field(default=0.7, ge=0.0, le=1.0)


@dataclass
class GameParams:
    max_rounds:      int = Field(ge=1, le=25)
    rounds_per_side: int = Field(ge=1, le=6)
    overtime_rounds: int = Field(ge=0, le=3)

    defuser_timer:   int = Field(default=45, ge=1,   le=60)


class DebugParams(BaseModel):
    config_keys:    bool = False
    red_percentage: bool = False
    headshot_match: bool = False
    infer_time:     bool = False

    model_config = ConfigDict(extra="ignore")
