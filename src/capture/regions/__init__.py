import numpy as np
from math import inf
from pydantic import BaseModel, ConfigDict, computed_field
from typing import Optional, cast

from utils import BBox_t
from utils.cv import crop2bbox, offset_bbox


__all__ = [
    "RegionBBoxes",
    "Regions"
]


class RegionBBoxes(BaseModel):
    timer:       Optional[BBox_t] = None
    kf_lines:    Optional[list[BBox_t]] = None

    team0_score: Optional[BBox_t] = None
    team1_score: Optional[BBox_t] = None
    team0_side:  Optional[BBox_t] = None
    team1_side:  Optional[BBox_t] = None

    model_config = ConfigDict(extra="ignore")

    @computed_field
    @property
    def max_bounds(self) -> BBox_t:
        """This field is the bbox containing all other bounding boxes"""
        top, left, bot, right = inf, inf, -inf, -inf
        single_boxes = [bbox
                        for bbox in self.__dict__.values()
                        if isinstance(bbox, tuple)]
        list_boxes  = [bbox
                       for bbox_list in self.__dict__.values()
                       for bbox in bbox_list
                       if isinstance(bbox_list, list) and isinstance(bbox_list[0], tuple)]

        all_boxes = single_boxes + list_boxes
        if len(all_boxes) == 0:
            raise ValueError("No boxes defined")

        for bbox in all_boxes:
            top   = min(top,   bbox[0])
            left  = min(left,  bbox[1])
            bot   = max(bot,   bbox[1]+bbox[3])
            right = max(right, bbox[0]+bbox[2])
        
        return cast(BBox_t, (top, left, right-left, bot-top))

    def crop(self, image: np.ndarray) -> dict[str, np.ndarray | list[np.ndarray]]:
        bboxes_dump: dict[str, BBox_t | list[BBox_t]] = self.model_dump(exclude_none=True)
        single_bboxes = {k: v for k,v in bboxes_dump.items() if not isinstance(v, list)}
        list_bboxes   = {k: v for k,v in bboxes_dump.items() if isinstance(v, list)}

        offset = self.max_bounds[:2]
        return (
            {name:  crop2bbox(image, offset_bbox(bbox, offset)) for name, bbox in single_bboxes.items()} |
            {name: [crop2bbox(image, offset_bbox(el,   offset)) for el in bbox] for name, bbox in list_bboxes.items()}
        )


class Regions(BaseModel):
    timer:       np.ndarray
    kf_lines:    list[np.ndarray]

    team0_score: np.ndarray
    team1_score: np.ndarray
    team0_side:  np.ndarray
    team1_side:  np.ndarray

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
