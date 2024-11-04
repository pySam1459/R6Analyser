import cv2
import numpy as np
from dataclasses import dataclass as odataclass
from typing import Optional

from params import OCRParams
from utils import BBox_t
from utils.enums import Team

from .utils import HSVColourRange, cvt_rgb2hsv


@odataclass
class Segment:
    image: np.ndarray
    rect:  list[int]

    @staticmethod
    def create(image: np.ndarray, rect: list[int]) -> "Segment":
        x,y,w,h = rect
        return Segment(image=image[y:y+h,x:x+w], rect=rect)


@odataclass
class KfLineSegments:
    left:   Optional[Segment]
    right:  Segment
    middle: Segment
    left_team: Team
    right_team: Team


def rle(mask: np.ndarray) -> tuple[list[int], list[int]]:
    """Run length encoding of a binary mask"""
    n = mask.shape[0]
    y = mask[1:] != mask[:-1]         # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1) # must include last element posi
    z = np.diff(np.append(-1, i))     # run lengths
    p = np.cumsum(z)                  # positions
    return (z.tolist(), p.tolist())


def clean_dist(dist: np.ndarray, rls: list[int], pos: list[int], params: OCRParams) -> np.ndarray:
    n = len(rls)
    min_area = dist.shape[0] * params.seg_min_area
    for i in range(n-1):
        rl, p, p2 = rls[i+1], pos[i], pos[i+1]
        if dist[p] == 0 and rl < min_area:
            dist[p:p2] = 1
    return dist


def get_mask_lr(mask: np.ndarray, params: OCRParams) -> Optional[tuple[int, int]]:
    dist0 = np.mean(mask, axis=0) > params.seg_mask_th
    rls, pos = rle(dist0)
    dist0 = clean_dist(dist0, rls, pos, params)
    rls, pos = rle(dist0)

    width = dist0.shape[0]
    for i in range(len(rls)-1, 0, -1):
        run, p = rls[i], pos[i-1]
        if dist0[p] == 1 and run > width * params.seg_min_width:
            return pos[i-1], pos[i]

    return None

def get_mask_tb(mask: np.ndarray, params: OCRParams) -> tuple[int,int]:
    dist = np.mean(mask, axis=1) > params.seg_mask_th
    indices = np.where(dist)[0]
    return indices[0], indices[-1]

def get_seg_box(line: np.ndarray, cols: HSVColourRange, params: OCRParams) -> Optional[BBox_t]:
    mask = cv2.inRange(line, cols.low, cols.high)
    lr = get_mask_lr(mask, params)
    if lr is None:
        return None

    tb = get_mask_tb(mask[:,lr[0]:lr[1]], params)
    return (lr[0], tb[0], lr[1], tb[1])


## --- Black Section ---
def get_black_left(kf_line: np.ndarray, params: OCRParams) -> Optional[int]:
    black_gray: np.ndarray = cv2.cvtColor(kf_line, cv2.COLOR_RGB2GRAY)
    mask = (black_gray < params.seg_black_th).astype(np.uint8)
    dist = np.mean(mask, axis=0) > params.seg_dist_th
    
    width = dist.shape[0]
    rls, pos = rle(dist)
    for i in range(len(rls)-2, -1, -1):
        run, p = rls[i], pos[i]
        if dist[p-1] == 0 and run > width * params.seg_min_width:
            return p
    return None


def create_full_line_segment(kfline_img: np.ndarray,
                             box0: BBox_t,
                             box1: BBox_t,
                             params: OCRParams) -> Optional[KfLineSegments]:
    l0,t0,r0,b0 = box0
    l1,t1,r1,b1 = box1
    if l0 < l1 < r0 or l1 < l0 < r1:
        ## segments cannot intersect
        return None

    if r0 < l1:
        x, x2 = r0, l1
    else:
        x, x2 = r1, l0

    y = min(t0, t1)
    h = max(b0-t0, b1-t1)

    seg0    = Segment.create(kfline_img, [l0, y, r0-l0, h])
    seg1    = Segment.create(kfline_img, [l1, y, r1-l1, h])
    seg_mid = Segment.create(kfline_img, [x,  y, x2-x,  h])

    if r0 < l1:
        return KfLineSegments(left=seg0, right=seg1, middle=seg_mid,
                              left_team=Team.TEAM0, right_team=Team.TEAM1)
    else:
        return KfLineSegments(left=seg1, right=seg0, middle=seg_mid,
                              left_team=Team.TEAM1, right_team=Team.TEAM0)


def create_part_line_segment(kfline_img: np.ndarray,
                             box: BBox_t,
                             team: Team,
                             params: OCRParams) -> Optional[KfLineSegments]:
    width = kfline_img.shape[1]
    l,t,r,b = box
    if r < width * 0.95:
        return None
    
    black_left = get_black_left(kfline_img[:,:l], params)
    if black_left is None:
        return None
    
    right_seg = Segment.create(kfline_img, [l, t, r-l, b-t])
    mid_seg   = Segment.create(kfline_img, [black_left, t, l-black_left, b-t])

    return KfLineSegments(left=None, right=right_seg, middle=mid_seg,
                          left_team=Team.UNKNOWN, right_team=team)


def segment(kfline_img: np.ndarray,
            t0_cols: HSVColourRange,
            t1_cols: HSVColourRange,
            params: OCRParams) -> Optional[KfLineSegments]:
    
    kfline_hsv = cvt_rgb2hsv(kfline_img, params.hue_offset)
    box0 = get_seg_box(kfline_hsv, t0_cols, params)
    box1 = get_seg_box(kfline_hsv, t1_cols, params)
    if box0 is None and box1 is None:
        return None

    elif box0 is not None and box1 is not None:
        return create_full_line_segment(kfline_img, box0, box1, params)

    elif box0 is not None:
        return create_part_line_segment(kfline_img, box0, Team.TEAM0, params)
    elif box1 is not None:
        return create_part_line_segment(kfline_img, box1, Team.TEAM1, params)
