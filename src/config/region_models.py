from pydantic import BaseModel, ConfigDict, Field, field_validator, computed_field
from typing import Any

from utils import BBox_t


__all__ = [
    "TimerRegion",
    "KFLineRegion"
]


@classmethod
def validate_bounding_box(cls, v: Any):
    if v is None:
        return v
    if not isinstance(v, tuple) or len(v) != 4 or not all(isinstance(el, int) for el in v):
        raise ValueError("must be of length=4 and type Tuple[int, int, int, int]")
    if any(el < 0 for el in v):
        raise ValueError("elements must be positive integers")
    return v


class TimerRegion(BaseModel):
    timer:        BBox_t

    model_config = ConfigDict(extra="forbid")

    validate_timer_bbox = field_validator("timer")(validate_bounding_box)

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


class KFLineRegion(BaseModel):
    kf_line:      BBox_t

    num_kf_lines: int   = Field(default=3,   ge=0, le=4)
    kf_buf:       int   = Field(default=4,   ge=0)
    kf_buf_mult:  float = Field(default=1.4, ge=1.0)

    model_config = ConfigDict(extra="forbid")

    validate_kf_line_bbox = field_validator("kf_line")(validate_bounding_box)

    @computed_field
    @property
    def kf_lines(self) -> list[BBox_t]:
        x, y, w, h = self.kf_line
        return [
            (x, y - int(h * self.kf_buf_mult * i) - self.kf_buf, w, h + self.kf_buf * 2)
            for i in range(self.num_kf_lines)
        ]
