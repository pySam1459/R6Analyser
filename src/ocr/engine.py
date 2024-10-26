import cv2
import numpy as np
from dataclasses import dataclass as odataclass
from re import match
from tesserocr import PSM
from typing import Optional, Callable

from assets import Assets
from settings import Settings
from utils import Timestamp, resize_height, filter_none, clip_around, argmin
from utils.enums import Team
from utils.constants import *

from .base import BaseOCREngine
from .segment import segment
from .utils import *


__all__ = [
    "OCREngine",
    "OCRLineResult"
]

@odataclass
class OCRLineResult:
    left:     Optional[str]
    right:    str
    headshot: bool

    left_team: Team
    right_team: Team

    left_image:   Optional[np.ndarray] = None
    right_image:  Optional[np.ndarray] = None
    middle_image: Optional[np.ndarray] = None


def get_headshot_match(middle_segment: np.ndarray,
                       hs_asset: np.ndarray,
                       params: OCRParams):
    """Uses template matching to determine if a kf line has a headshot"""
    ## TODO: could potentially cache the hs_asset resized, with wide version for speedup
    middle_gray = cv2.cvtColor(middle_segment, cv2.COLOR_RGB2GRAY)

    hs_asset = resize_height(hs_asset, middle_segment.shape[0])
    result = cv2.matchTemplate(middle_gray, hs_asset, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    ## try matching against a wider template
    hs_asset_wide = cv2.resize(hs_asset, None, fx=params.hs_wide_sf, fy=1.0, interpolation=cv2.INTER_LINEAR)
    result_wide = cv2.matchTemplate(middle_gray, hs_asset_wide, cv2.TM_CCOEFF_NORMED)
    _, max_val_wide, _, _ = cv2.minMaxLoc(result_wide)

    max_val = max(max_val, max_val_wide)
    return max_val


class OCREngine(BaseOCREngine):
    team0_colours: HSVColourRange
    team1_colours: HSVColourRange

    _debug_vars: dict

    def __init__(self, params: OCRParams, settings: Settings, assets: Assets, _debug_print: Optional[Callable] = None) -> None:
        super(OCREngine, self).__init__(settings, _debug_print)
        self.__assets = assets
        self.params = params

        self.__has_colours = False

        self.__last_timer: Optional[Timestamp] = None
    
    ## Team colour methods
    def read_team_colour(self, team_score_image: np.ndarray) -> tuple[int,int,int]:
        """Reads the team colour my averaging the colour over the threshold mask of the team score"""
        th_img = self._read_threshold(team_score_image)

        mask = np.asarray(th_img) == 0
        masked_colours = team_score_image[mask]
        avg_colour: np.ndarray = np.mean(masked_colours, axis=0)
        r,g,b = avg_colour.astype(np.int64)
        return (r, g, b)
    
    def _set_colours(self, rang0: HSVColourRange, rang1: HSVColourRange) -> None:
        self.team0_colours = rang0
        self.team1_colours = rang1
        self.__has_colours = True
    
    def set_colours(self, team0_score: np.ndarray, team1_score: np.ndarray) -> None:
        stds = (self.params.hue_std, self.params.sat_std)
        rgb = self.read_team_colour(team0_score)
        colours0 = get_colour_range(rgb, stds, self.params.col_zscore)

        rgb = self.read_team_colour(team1_score)
        colours1 = get_colour_range(rgb, stds, self.params.col_zscore)

        self._set_colours(colours0, colours1)

    @property
    def has_colours(self) -> bool:
        return self.__has_colours
    
    def read_score(self, score: np.ndarray) -> Optional[str]:
        score = clip_around(score, self.params.sl_clip_around)
        score_median = cv2.medianBlur(score, 3)
        score_resize = cv2.resize(score_median,
                           None,
                           fx=self.params.sl_scalex,
                           fy=self.params.sl_scaley,
                           interpolation=cv2.INTER_CUBIC)

        text_char = self.readtext(score_resize, PSM.SINGLE_CHAR, DIGITS)
        if match(SCORELINE_PATTERN, text_char):
            return text_char

        return None

    def read_kfline(self, kfline_img: np.ndarray, charlist = IGN_CHARLIST) -> Optional[OCRLineResult]:
        """Reads a single killfeed line and returns an OCRLineResult instance"""
        if not self.has_colours:
            return None
        
        segment_output = segment(kfline_img,
                                 self.team0_colours,
                                 self.team1_colours,
                                 self.params)
        if segment_output is None:
            return None
        
        right_image = segment_output.right.image
        right_text = self.readtext(~right_image, OCReadMode.WORD, charlist)
        if len(right_text) <= 2:
            return None
        
        middle_image = segment_output.middle.image

        headshot_match = get_headshot_match(middle_image, self.__assets["headshot_mask"], self.params)
        is_headshot = headshot_match >= self.params.hs_th
        self.debug_print("headshot_match", headshot_match)

        if segment_output.left is None:
            return OCRLineResult(None, right_text, is_headshot,
                                 Team.UNKNOWN, segment_output.right_team,
                                 None, right_image, middle_image)

        left_image = segment_output.left.image
        left_text = self.readtext(~left_image, OCReadMode.WORD, charlist)
        if len(left_text) <= 2:
            return None

        return OCRLineResult(left_text, right_text, is_headshot,
                             segment_output.left_team, segment_output.right_team,
                             left_image, right_image, middle_image)


    def read_timer(self, timer_img: np.ndarray) -> tuple[Optional[Timestamp], bool]:
        """Returns (timer: Optional[Timestamp], is_bomb_countdown: bool)"""
        if self.get_is_bomb_countdown(timer_img):
            self.__last_timer = None
            return None, True

        not_timer_img = ~cv2.cvtColor(timer_img, cv2.COLOR_RGB2GRAY) # type: ignore
        denoised_image = cv2.medianBlur(not_timer_img, 3)
        th_image = denoised_image > OCR_TIMER_THRESHOLD # type: ignore

        results = self.readtext([denoised_image, th_image], OCReadMode.LINE, TIMER_CHARLIST)
        self.__last_timer = self.__pick_timer_result(results)
        return self.__last_timer, False

    def get_is_bomb_countdown(self, timer_img: np.ndarray) -> bool:
        red_perc = get_timer_redperc(timer_img)
        self.debug_print("red_percentage", f"{red_perc=}")

        return red_perc > BOMB_COUNTDOWN_RT

    def __pick_timer_result(self, results: list[str]) -> Optional[Timestamp]:
        converted_results = [self.__convert_raw_to_ts(res) for res in results]
        filtered_results = filter_none(converted_results)
        if len(filtered_results) == 0:
            return None
        
        if self.__last_timer is None:
            return filtered_results[0]
        
        lt_int = self.__last_timer.to_int()
        arg_idx = argmin(filtered_results, lambda ts: abs(ts.to_int() - lt_int + 1))
        return filtered_results[arg_idx]

    def __convert_raw_to_ts(self, raw_result: str) -> Optional[Timestamp]:
        timer_match = match(r"(\d?\d)([:\.])(\d\d)", raw_result)
        if timer_match is None:
            return None

        if timer_match.group(2) == ":":
            return Timestamp(
                minutes=int(timer_match.group(1)),
                seconds=int(timer_match.group(3))
            )
        elif timer_match.group(2) == ".":
            return Timestamp(
                minutes=0,
                seconds=int(timer_match.group(1))
            )
        ## TODO: could improve this? maybe assume '220' == '2m 20s', infer from previous timer?

        return None
