from pydantic import BaseModel, ConfigDict, ValidationInfo, Field, computed_field, field_validator
from typing import Any

from utils import BBox_t

from .utils import InBBox_t


__all__ = [
    "TimerRegion",
    "KFLineRegion"
]


class TimerRegion(BaseModel):
    timer: InBBox_t

    score_width: float = Field(default=0.45, ge=0.0, le=1.0)
    side_width:  float = Field(default=0.45, ge=0.0, le=1.0)
    t1_offset:   int   = Field(default=6,    ge=0)

    model_config = ConfigDict(extra="ignore")

    @field_validator("score_width", "side_width", mode="before")
    @classmethod
    def convert_widths(cls, v: Any, info: ValidationInfo) -> float:
        if isinstance(v, float) and 0.0 <= v <= 1.0:
            return v
        elif isinstance(v, int):
            width: int = info.data["timer"][2]
            return v / width
        raise ValueError(f"Cannot parse timer region hyperparameter {info.field_name}, invalid value {v}")


    @computed_field
    @property
    def team0_score(self) -> BBox_t:
        score_width = int(self.timer[2] * self.score_width)
        return (self.timer[0] - score_width, self.timer[1], score_width, self.timer[3])

    @computed_field
    @property
    def team1_score(self) -> BBox_t:
        x = self.timer[0] + self.timer[2] + self.t1_offset
        score_width = int(self.timer[2] * self.score_width)
        return (x, self.timer[1], score_width, self.timer[3])

    @computed_field
    @property
    def team0_side(self) -> BBox_t:
        side_width = int(self.timer[2] * self.side_width)
        return (self.team0_score[0] - side_width, self.timer[1], side_width, self.timer[3])

    @computed_field
    @property
    def team1_side(self) -> BBox_t:
        x = self.team1_score[0]+self.team1_score[2]
        side_width = int(self.timer[2] * self.side_width)
        return (x, self.timer[1], side_width, self.timer[3])


class KFLineRegion(BaseModel):
    kf_line:      InBBox_t

    num_kf_lines: int   = Field(default=3, ge=0, le=4)
    kf_buf:       int   = Field(default=4, ge=0)

    model_config = ConfigDict(extra="ignore")

    @computed_field
    @property
    def kf_lines(self) -> list[BBox_t]:
        x, y, w, h = self.kf_line
        return [
            (x, y - int(h + self.kf_buf) * i, w, h)
            for i in range(self.num_kf_lines)
        ]
