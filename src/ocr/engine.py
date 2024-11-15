import cv2
import numpy as np
from dataclasses import dataclass as odataclass
from re import match
from typing import Optional, Callable

from assets import Assets
from params import OCRParams
from utils import Timestamp, filter_none, argmin
from utils.cv import resize_height, gen_gaussian2d, guassian_threshold
from utils.enums import Team
from utils.constants import *

from .base import BaseOCREngine, OCRMode
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


class OCREngine(BaseOCREngine):
    team0_colours: HSVColourRange
    team1_colours: HSVColourRange

    _debug_vars: dict

    def __init__(self, params: OCRParams,
                 assets: Assets,
                 _debug_print: Optional[Callable] = None) -> None:
        super(OCREngine, self).__init__(_debug_print)
        self.params = params

        self.__assets = assets
        self.__gaussian = 255 * gen_gaussian2d(96, 3.0)

        self.__has_colours = False

        self.__last_timer: Optional[Timestamp] = None
    
    ## Team colour methods
    def average_mask_colour(self, image: np.ndarray) -> tuple[int,int]:
        """Reads the team colour my averaging the colour over the threshold mask of the team score"""
        th_img = self._read_threshold(image)

        mask = th_img == 0
        masked_score = image[mask]

        masked_colours = masked_score.reshape(1, -1, 3)
        masked_hsv = cvt_rgb2hsv(masked_colours, self.params.hue_offset)

        avg_colour: np.ndarray = np.mean(masked_hsv[0,:,:2], axis=0).astype(np.uint8)
        return (avg_colour[0], avg_colour[1])
    
    def _set_colours(self, rang0: HSVColourRange, rang1: HSVColourRange) -> None:
        self.team0_colours = rang0
        self.team1_colours = rang1
        self.__has_colours = True
    
    def set_colours(self, team0_score: np.ndarray, team1_score: np.ndarray) -> None:
        stds = (self.params.hue_std, self.params.sat_std)
        hs = self.average_mask_colour(team0_score)
        colours0 = get_hsv_range(hs, stds, self.params.col_zscore)

        hs = self.average_mask_colour(team1_score)
        colours1 = get_hsv_range(hs, stds, self.params.col_zscore)

        self._set_colours(colours0, colours1)

    @property
    def has_colours(self) -> bool:
        return self.__has_colours
    
    def read_score(self, score: np.ndarray, side: Optional[np.ndarray] = None, save = None) -> Optional[str]:
        """Reads a score from the scoreline, pass the side region as well for better image thresholding"""
        image = score
        if side is not None:
            image = np.hstack((score, side))

        th_score = self._read_threshold(image)
        th_score = th_score[:, :score.shape[1]] ## remove side image before ocr
        th_median = cast(np.ndarray, cv2.medianBlur(th_score, 3))

        th_gaussian = guassian_threshold(~th_median, self.__gaussian)
        th_clipped = 255 * (~(th_gaussian > 127)).astype(np.uint8)

        if save:
            cv2.imwrite(save, th_clipped)


        text_char = self.readtext(th_clipped, OCRMode.CHAR, SCORE_CHARLIST)
        if match(SCORELINE_PATTERN, text_char):
            return text_char.replace("O", "0")

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
        right_text = self.readtext(~right_image, OCRMode.WORD, charlist).lstrip(IGN_EXTRA_CHARS)
        if len(right_text) <= 2:
            return None
        
        middle_image = segment_output.middle.image

        headshot_asset = self.__assets["headshot_mask"]
        headshot_match = self.get_headshot_match(middle_image, headshot_asset)
        is_headshot = headshot_match >= self.params.hs_th
        self.debug_print("headshot_match", headshot_match)

        if segment_output.left is None:
            return OCRLineResult(None, right_text, is_headshot,
                                 Team.UNKNOWN, segment_output.right_team,
                                 None, right_image, middle_image)

        left_image = segment_output.left.image
        left_text = self.readtext(~left_image, OCRMode.WORD, charlist).lstrip(IGN_EXTRA_CHARS)
        if len(left_text) <= 2:
            return None

        return OCRLineResult(left_text, right_text, is_headshot,
                             segment_output.left_team, segment_output.right_team,
                             left_image, right_image, middle_image)

    def get_headshot_match(self, middle_segment: np.ndarray,
                                 hs_asset: np.ndarray):
        """Uses template matching to determine if a kf line has a headshot"""
        ## TODO: could potentially cache the hs_asset resized, with wide version for speedup
        middle_gray = cv2.cvtColor(middle_segment, cv2.COLOR_RGB2GRAY)

        vals = []
        for mask in [hs_asset, hs_asset[2:-2,:]]:
            mask = resize_height(mask, middle_segment.shape[0])
            result = cv2.matchTemplate(middle_gray, mask, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            vals.append(max_val)

            ## try matching against a wider template
            mask_wide = cv2.resize(mask,
                                    None,
                                    fx=self.params.hs_wide_sf,
                                    fy=1.0,
                                    interpolation=cv2.INTER_LINEAR)
            result_wide = cv2.matchTemplate(middle_gray, mask_wide, cv2.TM_CCOEFF_NORMED)
            _, max_val_wide, _, _ = cv2.minMaxLoc(result_wide)
            vals.append(max_val_wide)

        return max(vals)


    def read_timer(self, timer_img: np.ndarray) -> tuple[Optional[Timestamp], bool]:
        """Returns (timer: Optional[Timestamp], is_bomb_countdown: bool)"""
        gray_image = cv2.cvtColor(timer_img, cv2.COLOR_RGB2GRAY)
        if self.__is_inf(gray_image):
            ## infinite symbol shows when game waits at end pick phase
            return Timestamp(minutes=0, seconds=0), False

        not_timer_img = ~gray_image # type: ignore
        denoised_image = cast(np.ndarray, cv2.medianBlur(not_timer_img, 3))
        th_image = denoised_image > OCR_TIMER_THRESHOLD

        results = self.readtext([denoised_image, th_image], OCRMode.LINE, TIMER_CHARLIST)
        self.__last_timer = self.__pick_timer_result(results)

        red_perc = get_timer_redperc(timer_img)
        self.debug_print("red_percentage", f"{red_perc=}")

        return self.__last_timer, red_perc > BOMB_COUNTDOWN_RT
    
    def __is_inf(self, timer_image: np.ndarray) -> bool:
        template = resize_height(self.__assets["timer_inf"], timer_image.shape[0]//2)
        result = cv2.matchTemplate(timer_image, template, cv2.TM_CCOEFF_NORMED)
        max_result = cast(float, np.max(result))
        return max_result > self.params.inf_th


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
