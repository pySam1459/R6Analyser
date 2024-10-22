from pydantic import BaseModel, ConfigDict, Field, computed_field

from utils import BBox_t

from .utils import InBBox_t


__all__ = [
    "TimerRegion",
    "KFLineRegion"
]


class TimerRegion(BaseModel):
    timer: InBBox_t

    model_config = ConfigDict(extra="ignore")

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
    kf_line:      InBBox_t

    num_kf_lines: int   = Field(default=3,   ge=0, le=4)
    kf_buf:       int   = Field(default=4,   ge=0)
    kf_buf_mult:  float = Field(default=1.4, ge=1.0)

    model_config = ConfigDict(extra="ignore")

    @computed_field
    @property
    def kf_lines(self) -> list[BBox_t]:
        x, y, w, h = self.kf_line
        return [
            (x, y - int(h * self.kf_buf_mult * i), w, h)
            for i in range(self.num_kf_lines)
        ]
