import cv2
import numpy as np
from dataclasses import dataclass as odataclass
from typing import Optional

from utils.enums import Team

from .utils import OCRParams, HSVColourRange


@odataclass
class Segment:
    image: np.ndarray
    rect:  list[int]


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


def get_seg_lr(line: np.ndarray, cols: HSVColourRange, params: OCRParams) -> Optional[tuple[int,int]]:
    mask = cv2.inRange(line, cols.low, cols.high)
    # print(cols.low, cols.high)
    return get_mask_lr(mask, params)


## --- Crop Black Section ---
def get_dist(black_section: np.ndarray, threshold: int, dist_th: float) -> np.ndarray:
    black_gray: np.ndarray = cv2.cvtColor(black_section, cv2.COLOR_RGB2GRAY)
    mask = (black_gray < threshold).astype(np.uint8)
    return np.mean(mask, axis=0) > dist_th


def get_black_left(black_section: np.ndarray, params: OCRParams) -> Optional[int]:
    dist = get_dist(black_section, params.seg_black_th, params.seg_dist_th)
    w = dist.shape[0]
    dist = dist[:-int(w*0.9)]
    diffs = np.diff(dist)
    indices = np.where(diffs != 0)[0]
    if len(indices) == 0:
        return None
    return indices[-1]


def get_black_tb(black_section: np.ndarray, params: OCRParams) -> tuple[int,int]:
    black_section_vert = np.transpose(black_section, (1, 0, 2))
    vert_dist = get_dist(black_section_vert, params.seg_black_th, params.seg_dist_vert_th)
    
    indices = np.where(vert_dist)[0]
    if len(indices) < 2:
        return (0, black_section.shape[0])

    else:
        top, bot = indices[0], indices[-1]
        return (top, min(black_section.shape[0], bot-top+1))


## --- Create Segments ---
def create_segment(image: np.ndarray, rect: list[int]) -> Segment:
    x,y,w,h = rect
    return Segment(image=image[y:y+h,x:x+w], rect=rect)


def create_full_line_segment(kfline_img: np.ndarray,
                             lr0: tuple[int,int],
                             lr1: tuple[int,int],
                             params: OCRParams) -> Optional[KfLineSegments]:
    l0,r0 = lr0
    l1,r1 = lr1
    if l0 < l1 < r0 or l1 < l0 < r1:
        ## segments cannot intersect
        return None

    if r0 < l1:
        x, x2 = r0, l1
    else:
        x, x2 = r1, l0

    black_section = kfline_img[:,x:x2]
    y,h = get_black_tb(black_section, params)

    seg0    = create_segment(kfline_img, [l0, y, r0-l0, h])
    seg1    = create_segment(kfline_img, [l1, y, r1-l1, h])
    seg_mid = create_segment(kfline_img, [x,  y, x2-x,  h])

    if r0 < l1:
        return KfLineSegments(left=seg0, right=seg1, middle=seg_mid,
                              left_team=Team.TEAM0, right_team=Team.TEAM1)
    else:
        return KfLineSegments(left=seg1, right=seg0, middle=seg_mid,
                              left_team=Team.TEAM1, right_team=Team.TEAM0)


def create_part_line_segment(kfline_img: np.ndarray,
                             lr: tuple[int,int],
                             team: Team,
                             params: OCRParams) -> Optional[KfLineSegments]:
    width = kfline_img.shape[1]
    l,r = lr
    if r < width * 0.95:
        return None
    
    black_left = get_black_left(kfline_img[:,l:r], params)
    if black_left is None:
        return None
    
    black_section = kfline_img[:,black_left:l]
    y,h = get_black_tb(black_section, params)
    
    right_seg = create_segment(kfline_img, [l, y, r-l, h])
    mid_seg   = create_segment(kfline_img, [black_left, y, l, h])

    return KfLineSegments(left=None, right=right_seg, middle=mid_seg,
                          left_team=Team.UNKNOWN, right_team=team)


def segment(kfline_img: np.ndarray,
            t0_cols: HSVColourRange,
            t1_cols: HSVColourRange,
            params: OCRParams) -> Optional[KfLineSegments]:
    
    kfline_hsv = cv2.cvtColor(kfline_img, cv2.COLOR_RGB2HSV)
    lr0 = get_seg_lr(kfline_hsv, t0_cols, params)
    lr1 = get_seg_lr(kfline_hsv, t1_cols, params)
    if lr0 is None and lr1 is None:
        return None

    if lr0 is not None and lr1 is not None:
        return create_full_line_segment(kfline_img, lr0, lr1, params)

    if lr0 is not None:
        return create_part_line_segment(kfline_img, lr0, Team.TEAM0, params)
    elif lr1 is not None:
        return create_part_line_segment(kfline_img, lr1, Team.TEAM1, params)
