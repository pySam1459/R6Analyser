from pydantic import BaseModel, ConfigDict, Field, computed_field

from utils import BBox_t

from .utils import InBBox_t


__all__ = [
    "TimerRegion",
    "KFLineRegion"
]


class TimerRegion(BaseModel):
    timer: InBBox_t

    score_width: float = Field(default=0.4, ge=0.0, le=1.0)
    side_width:  float = Field(default=0.4, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="ignore")

    @computed_field
    @property
    def team0_score(self) -> BBox_t:
        score_width = int(self.timer[2] * self.score_width)
        return (self.timer[0] - score_width, self.timer[1], score_width, self.timer[3])

    @computed_field
    @property
    def team1_score(self) -> BBox_t:
        score_width = int(self.timer[2] * self.score_width)
        return (self.timer[0] + self.timer[2], self.timer[1], score_width, self.timer[3])

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
