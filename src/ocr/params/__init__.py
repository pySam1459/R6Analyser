from pydantic.dataclasses import dataclass

"""
OCRParams are here to resolve a circular import issue,
OCRParams is used in config, which is then imported everywhere, including in ocr engine
"""

__all__ = [
    "OCRParams"
]


@dataclass
class OCRParams:
    sl_scalex:        float = 0.4
    sl_scaley:        float = 0.5
    sl_clip_around:   float = 0.1

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
