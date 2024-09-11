import numpy as np
from math import inf
from pydantic import BaseModel, ConfigDict, computed_field, model_validator
from typing import Optional, Self, TypeAlias, cast

from utils import BBox_t


__all__ = [
    "RegionBBoxes",
    "RegionImages",
    "InPersonRegions",
    "SpectatorRegions",
    "ImageRegions_t",
]


class RegionBBoxes(BaseModel):
    timer:       Optional[BBox_t] = None
    kf_lines:    Optional[list[BBox_t]] = None

    team1_score: Optional[BBox_t] = None
    team2_score: Optional[BBox_t] = None
    team1_side:  Optional[BBox_t] = None
    team2_side:  Optional[BBox_t] = None

    model_config = ConfigDict(extra="ignore")

    @computed_field
    @property
    def max_bounds(self) -> BBox_t:
        """This field is the bbox containing all other bounding boxes"""
        tl, br = [inf, inf], [-inf, -inf] ## [left,top],[right,bottom]
        single_boxes = [v for v in self.__dict__.values() if isinstance(v, tuple)]
        list_boxes = [el for v in self.__dict__.values() if isinstance(v, list) for el in v]

        all_boxes = single_boxes + list_boxes
        if len(all_boxes) == 0:
            raise ValueError("No boxes defined")

        for bbox in all_boxes:
            tl[:] = [min(tl[0], bbox[0]), min(tl[1], bbox[1])]
            br[:] = [max(br[0], bbox[0]+bbox[2]), max(br[1], bbox[1]+bbox[3])]
        
        return cast(BBox_t, (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1]))


class RegionImages(BaseModel):
    image: np.ndarray
    region_bboxes: RegionBBoxes

    timer:       Optional[np.ndarray] = None
    kf_lines:    Optional[list[np.ndarray]] = None
    team1_score: Optional[np.ndarray] = None
    team2_score: Optional[np.ndarray] = None
    team1_side:  Optional[np.ndarray] = None
    team2_side:  Optional[np.ndarray] = None

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_cropped_regions(self) -> Self:
        for name, bbox in self.region_bboxes.__dict__.items():
            if isinstance(bbox, tuple) and len(bbox) == 4:
                setattr(self, name, self._crop_to_bbox(bbox))
            elif isinstance(bbox, list):
                setattr(self, name, [self._crop_to_bbox(inner_bbox) for inner_bbox in bbox])

        return self
    
    def _crop_to_bbox(self, bbox: BBox_t) -> np.ndarray:
        tlx, tly = self.region_bboxes.max_bounds[:2]
        w,h = bbox[2:]
        x,y = bbox[0]-tlx, bbox[1]-tly
        return self.image[y:y+h,x:x+w]


class InPersonRegions(BaseModel):
    timer:       np.ndarray
    kf_lines:    list[np.ndarray]
    team1_score: np.ndarray
    team2_score: np.ndarray
    team1_side:  np.ndarray
    team2_side:  np.ndarray

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


class SpectatorRegions(BaseModel):
    timer:      np.ndarray
    kf_lines:   list[np.ndarray]

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


ImageRegions_t: TypeAlias = InPersonRegions | SpectatorRegions
