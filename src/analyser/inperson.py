import cv2
import numpy as np
from os import mkdir
from os.path import join, exists
from re import search, match
from Levenshtein import ratio
from typing import Optional, cast

from capture import InPersonRegions, RegionBBoxes
from config import Config
from history import KFRecord
from ignmatrix import Player_t
from ocr import OCRLineResult
from utils.cli import AnalyserArgs
from utils.enums import Team, WinCondition
from utils import *
from utils.constants import *

from .base import Analyser, State
from .killfeed import SmartKillfeed, KFRecord_t
from .utils import is_headshot, get_timer_redperc


__all__ = ["InPersonAnalyser"]


class InPersonAnalyser(Analyser):
    NUM_LAST_SECONDS  = 4  ## number of seconds to continue reading killfeed after round end (reliability reasons)
    END_ROUND_SECONDS = 12 ## number of seconds to check no timer to determine round end
    
    SCREENSHOT_REGIONS = ["timer",
                          "team1_score", "team2_score", "team1_side", "team2_side",
                          "kf_lines"]
    NON_NAMES = ["has found the bomb",
                 "Friendly Fire has been activated for",
                 "Friendly Fire turned off until Action Phase"]
    NON_NAME_THRESHOLD = 0.5

    SCORELINE_PROB = 0.25
    TIMER_PROB = 0.35
    KF_PROB = 0.10

    PROXIMITY_DIST = 35

    def __init__(self, args: AnalyserArgs, config: Config) -> None:
        super(InPersonAnalyser, self).__init__(args, config)

        self.last_kf_seconds = None
        self.end_round_seconds = None

        self.skf = SmartKillfeed(self.config, self.history, self._add_record)
    
    def _get_regions(self) -> RegionBBoxes:
        return RegionBBoxes.model_validate(self.config.capture.regions.model_dump(exclude_none=True))

    ## ----- SCORELINE -----
    def _handle_scoreline(self, regions: InPersonRegions) -> None:
        """Extracts the current scoreline visible and determines when a new rounds starts"""
        if not self.running or not self.state.end_round: return

        scoreline = self.__read_scoreline(regions.team1_score, regions.team2_score)
        if not scoreline:
            return

        new_roundn = scoreline.left + scoreline.right + 1
        if new_roundn in self.history:
            self._fix_state()
            return

        self._new_round(scoreline)

        self.history.cround.atk_side = self.__read_atkside(regions.team1_side, regions.team2_side)
        self._verbose_print(1, f"Atk Side: {self.history.cround.atk_side}")

    def __read_scoreline(self, slimg_left: np.ndarray, slimg_right: np.ndarray) -> Optional[Scoreline]:
        """Reads the scoreline from the 2 TEAM1/2_SCORE_REGION screenshot"""
        left_img = self._screenshot_preprocess(slimg_left, to_gray=True, squeeze_width=0.65)
        right_img = self._screenshot_preprocess(slimg_right, to_gray=True, squeeze_width=0.65)

        results = flatten(self._readbatch([left_img, right_img],
                                          InPersonAnalyser.SCORELINE_PROB,
                                          allowlist=DIGITS))
        if (len(results) != 2 or
                not match(r"^\d+$", results[0]) or
                not match(r"^\d+$", results[1])):
            return None

        return Scoreline(left=int(results[0]), right=int(results[1]))

    def __read_atkside(self, side1: np.ndarray, side2: np.ndarray) -> Team:
        """Matches the side icon next to the scoreline with `res/swords.jpg` to determine which side is attack."""
        icon1 = self._screenshot_preprocess(side1, to_gray=True, denoise=True)
        icon2 = self._screenshot_preprocess(side2, to_gray=True, denoise=True)

        res_icon1 = cv2.matchTemplate(self.assets.atkside_icon, icon1, cv2.TM_CCOEFF_NORMED)
        res_icon2 = cv2.matchTemplate(self.assets.atkside_icon, icon2, cv2.TM_CCOEFF_NORMED)
        ## TODO: prob only need to match 1 side, then threshold
        # Get the maximum match value for each icon
        _, max_val_icon1, _, _ = cv2.minMaxLoc(res_icon1)
        _, max_val_icon2, _, _ = cv2.minMaxLoc(res_icon2)

        # Decide which icon matches best
        if max_val_icon1 >= max_val_icon2:
            return Team.TEAM0
        else:
            return Team.TEAM1


    ## ----- TIMER FUNCTION -----
    def _handle_timer(self, regions: InPersonRegions) -> None:
        """Reads and handles the timer information, used to determine when the bomb is planted and when the round ends"""
        if not self.running or not self.history.is_ready: return

        red_perc = get_timer_redperc(regions.timer)
        self._debug_print("red_percentage", f"{red_perc=}")

        new_time = self.__read_timer(regions.timer, red_perc)

        ## timer is showing
        if new_time is not None:
            self.__handle_timer_shown(new_time)
        
        ## The bomb uses a separate countdown timer as the visual timer cannot be accurately tracked
        elif red_perc > BOMB_COUNTDOWN_RT:
            self.__handle_bomb_countdown()

        elif self.last_kf_seconds is None and self.end_round_seconds is None and self.state.in_round:
            self.last_kf_seconds = self.timer.time()
            self.end_round_seconds = self.timer.time()
        
        if self.last_kf_seconds is not None and self.last_kf_seconds + InPersonAnalyser.NUM_LAST_SECONDS < self.timer.time():
            self.last_kf_seconds = None

        if (self.end_round_seconds is not None and
                self.end_round_seconds + InPersonAnalyser.END_ROUND_SECONDS < self.timer.time()
                and self.__read_scoreline(regions.team1_score, regions.team2_score) is None):
            self._end_round()
            self.end_round_seconds = None

    def __read_timer(self, image: np.ndarray, red_perc: float) -> Optional[Timestamp]:
        """
        Reads the current time displayed in the region `TIMER`
        If the timer is not present, None is returned
        """
        image = self._screenshot_preprocess(image, to_gray=True, denoise=True, squeeze_width=0.75)
        results = self._readtext(image, prob=InPersonAnalyser.TIMER_PROB, allowlist=TIMER_CHATLIST)
        if len(results) == 0:
            return None

        result = results[0] if len(results) == 1 else "".join(results)
        timer_match = search(r"(\d?\d).?(\d\d)", result)

        if timer_match is None:
            return None

        if red_perc < TIMER_LAST_SECONDS_RT:
            minutes = int(timer_match.group(1))
            seconds = int(timer_match.group(2))
        else:
            minutes = 0
            seconds = int(timer_match.group(1))
        
        return Timestamp(minutes=minutes, seconds=seconds)

    def __handle_timer_shown(self, new_time: Timestamp) -> None:
        self.current_time = new_time
        self.defuse_countdown_timer = None

        self.last_kf_seconds = None
        self.end_round_seconds = None
        
        self.prog_bar.set_time(new_time)
        self.prog_bar.refresh()

    def __handle_bomb_countdown(self) -> None:
        if self.defuse_countdown_timer is None: ## bomb planted
            self.defuse_countdown_timer = self.timer.time()
            self.state.bomb_planted = True

            self.history.cround.bomb_planted_at = self.current_time
            self._verbose_print(1, f"Bomb planted at: {self.current_time}")
            self.prog_bar.bomb()
        
        elif self.history.cround.bomb_planted_at is not None:
            bpat_int = self.history.cround.bomb_planted_at.to_int()
            tdelta = int(self.timer.time() - self.defuse_countdown_timer)
            self.current_time = Timestamp.from_int(bpat_int - tdelta)
            self.prog_bar.set_time(int(self.config.defuser_timer-tdelta))
            self.prog_bar.refresh()


    ## ----- KILL FEED -----
    def _handle_feed(self, regions: InPersonRegions) -> None:
        """Handles the killfeed by reading the names, querying the ign matrix and the information to History"""
        if not self.running or not self.history.is_ready: return
        if not self.state.in_round and self.last_kf_seconds is None: return

        image_lines = [regions.kf_lines[i] for i in range(self.config.capture.regions.num_kf_lines)]
        for line in self.__read_feed(image_lines)[0]:
            if len(line.results) == 2:
                self.__handle_feed_normal(line)
                
            elif len(line.results) == 1:
                self.__handle_feed_suicide(line)
                
    
    def __handle_feed_normal(self, line: OCRLine) -> None:
        player = self.ign_matrix.get(line.results[0].text)
        target = self.ign_matrix.get(line.results[1].text)

        if player is None or target is None:
            return

        record = KFRecord(player, target, self.current_time, line.headshot)
        self.skf.add(record, KFRecord_t.NORMAL)

    def __handle_feed_suicide(self, line: OCRLine) -> None:
        ## suicides?
        res = line.results[0]
        player = self.ign_matrix.get(res.text)
        if player is None:
            return
        if res.rect[0] + res.rect[2] < line.size[1] * 0.8:
            return
        if compute_rect_ratio([0,0,line.size[1],line.size[0]], res.rect) < 0.1:
            return

        record = KFRecord(player, player, self.current_time, line.headshot)
        self.skf.add(record, KFRecord_t.SUICIDE)

    def _add_record(self, record: KFRecord) -> None:
        self.history.cround.killfeed.append(record)

        if record.player.type == Player_t.INFER or record.target.type == Player_t.INFER:
            self.ign_matrix.update_mats(record.player.ign, record.target.ign)
            self.ign_matrix.update_mats(record.target.ign, record.player.ign)

        self._verbose_print(2, record.to_str())
        self.prog_bar.set_desc(record.to_str(show_time=False))
        self.prog_bar.refresh()
    
    def __read_feed(self, image_lines: list[np.ndarray]) -> tuple[list[OCRLine], list[np.ndarray]]:
        image_lines = [self._screenshot_preprocess(image, denoise=True) for image in image_lines]
        headshots = [is_headshot(img_line, self.assets.headshot) for img_line in image_lines]
        nohs_il = self.__fill_in_hs(image_lines, headshots)

        ocr_results = self.ocr_engine.read_batch(nohs_il,
                            add_margin=0.15,
                            slope_ths=0.5,
                            allowlist=IGN_CHARLIST)
        ocr_lines = self.__get_ocrlines(ocr_results, image_lines)
        for line, hs in zip(ocr_lines, headshots):
            line.headshot = bool(hs)

        return ocr_lines, image_lines
    
    def __fill_in_hs(self, image_lines: list[np.ndarray], headshots: list[BBox_t | None]) -> list[np.ndarray]:
        """Fills over the headshot icon in the image line to improve OCR results"""
        out = []
        for img, hsr in zip(image_lines, headshots):
            img_cpy = img.copy()
            if hsr is not None:
                x, y, w, h = hsr
                img_cpy[y:y+h, x:x+w] = 0
            
            out.append(img_cpy)
        return out
    
    def __get_ocrlines(self, lines: tuple[list[OCResult]], image_lines: list[np.ndarray]) -> list[OCRLine]:
        ocr_lines: list[OCRLine] = []
        for raw_results, img in zip(lines, image_lines):
            ## cleaning stage 1
            line: list[OCResult] = []
            for res in raw_results:
                if res.prob < InPersonAnalyser.KF_PROB or len(res.text) < 2: continue
                if max([ratio(res.text, non_name) for non_name in InPersonAnalyser.NON_NAMES]) > InPersonAnalyser.NON_NAME_THRESHOLD: continue

                res.eval(self.ign_matrix)
                line.append(res)
            
            if len(line) == 0: continue
            elif len(line) == 2 and line[0].rect[0] > line[1].rect[0]:
                line[0], line[1] = line[1], line[0]
            elif len(line) >= 3: ## join call is required
                line = self.__join_line(line)

            ocr_lines.append(OCRLine(results=line, size=img.shape))

        return ocr_lines

    def __join_line(self, line: list[OCResult]) -> list[OCResult]:
        """Determines which OCResults to join"""
        ## TODO: check this
        prox_dist = InPersonAnalyser.PROXIMITY_DIST * (self.config.capture.scale_by / 4)
        line.sort(key=lambda ocr_res: ocr_res.rect[0])
        new_line: list[OCResult] = []
        used = []
        for i, ocr1 in enumerate(line):
            if i in used: continue
            if i == len(line)-1:
                new_line.append(ocr1)
                break

            for j, ocr2 in enumerate(line[i+1:], start=i+1):
                if j in used: continue
                
                jocr = ocr1.join(ocr2)
                jocr.eval(self.ign_matrix)

                mx_score = max(ocr1.eval_score, ocr2.eval_score) # type: ignore the line
                if jocr.eval_score > mx_score or mx_score == 0.0:
                    new_line.append(jocr)
                elif rect_collision(ocr1.rect, ocr2.rect) or rect_proximity(ocr1.rect, ocr2.rect) <= prox_dist:
                    new_line.append(jocr)
                else:
                    continue

                used += [i, j]
                break
            else:
                new_line.append(ocr1)
                used.append(i)

        if len(new_line) > 2: ## TODO: sort this out
            def _key(res: OCResult) -> float:
                if res.eval_score is None:
                    return -9999
                return res.eval_score

            new_line = sorted(new_line, key=_key, reverse=True)[:2]
            if new_line[0].rect[0] > new_line[1].rect[0]:
                new_line = [new_line[1], new_line[0]]

        return new_line


    # ----- GAME STATE -----
    def __finish_last_round(self, sl: Scoreline, save = True) -> None:
        """
        To save the correct information for each round, this method must be called just before the start of a new round,
          so that the scoreline and winner attributes can be used
        """
        if not self.history.is_ready: return

        ## infer winner of previous round based on new scoreline
        winner = self.history.cround.winner
        if winner is None and self.history.cround.scoreline is not None:
            sl_new = self.history.cround.scoreline
            winner = Team(sl_new.right < sl.right)
            self.history.cround.winner = winner ## if _score1+1 == score1, return 0

        win_con = self.history.cround.get_wincon(self.ign_matrix)
        self.history.cround.win_condition = win_con

        reat = self.history.cround.round_end_at
        if win_con == WinCondition.DISABLED_DEFUSER:
            self.history.cround.disabled_defuser_at = reat
            self._verbose_print(1, f"Disabled defuser at: {reat}")
        
        self._verbose_print(0, f"Team {winner} wins round {self.history.roundn} by {win_con.value} at {reat}.")
        if save:
            self.writer.write(self.history, self.ign_matrix)

    def _new_round(self, sl: Scoreline) -> None:
        """
        When a new round starts, this method is called, initialising a new round history
        The parameters `score1` and `score2` are the current scores displayed at the start of a new round
        """
        new_round = sl.left + sl.right + 1
        if new_round in self.history: return

        self.__finish_last_round(sl, save=True)
        self.state = State(True, False, False)

        self.history.new_round(new_round)
        self.history.cround.scoreline = sl
        self.skf.reset()

        self._verbose_print(1, f"New Round: {new_round} | Scoreline: {sl.left}-{sl.right}")
        self.prog_bar.reset()
        self.prog_bar.set_header(new_round, sl)
        self.prog_bar.refresh()
    
    def _end_round(self) -> None:
        """A game state method called when the program determines that the current round has ended"""
        self.history.cround.round_end_at = self.current_time
        self.state = State(False, True, False)

        self.prog_bar.set_time(0)
        self.prog_bar.set_desc("ROUND END")
        self.prog_bar.refresh()

        if self._is_end_of_game():
            self._end_game()

    def _is_end_of_game(self) -> bool:
        sl = self.history.cround.scoreline
        if sl is None:
            return False

        rps = self.config.rounds_per_side
        is_ot = sl.total >= self.config.max_rounds - self.config.overtime_rounds

        return (
            self.history.roundn >= self.config.max_rounds
                or (not is_ot and sl.max == rps+1)
                or (is_ot and abs(sl.left-sl.right) == 2)
        )
    
    def _end_game(self) -> None:
        """This method is called when the program determines the game has ended"""
        winner = self._get_last_winner()
        self.history.cround.winner = winner
        if (sl := self.history.cround.scoreline) is not None:
            self.__finish_last_round(sl.inc(winner), save=False)

        self.writer.write(self.history, self.ign_matrix)
        self._verbose_print(0, f"Data Saved to {self.writer.save_path}, program terminated.")
        self.stop()
    
    def _fix_state(self) -> None:
        """
        Called when the program incorrectly thinks the round ended,
        e.g. paused during death animation with now scoreline showing
        """
        self._verbose_print(1, f"Fixing State")
        self.state.in_round = True
        self.state.end_round = False

        timer_region = self.capture.get_region("timer")
        if timer_region is not None:
            red_perc = get_timer_redperc(timer_region)
            self.state.bomb_planted = red_perc > BOMB_COUNTDOWN_RT
        
        self.history.fix_round()

    ## ----- CHECK & TEST -----
    def _check_regions(self) -> None:
        """Called when the --check flag is present in the program call, saves the screenshotted regions as jpg images"""
        if not exists("images"):
            mkdir("images")

        self._verbose_print(0, "Saving check images")
        regions = cast(InPersonRegions, self.capture.next(self._get_regions()))

        def save(name: str, img: np.ndarray) -> None:
            cv2.imwrite(join("images", f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for name, img in regions.model_dump(exclude_none=True).items():
            if isinstance(img, list):
                for i, subimg in enumerate(img):
                    save(f"{name}_{i}", subimg)
            else:
                save(name, img)


    def _test_regions(self) -> None:
        """
        This method is called when the `--test-regions` flag is added to the program call,
        runs inference on a single screenshot.
        """
        regions = cast(InPersonRegions, self.capture.next(self._get_regions()))
        scoreline = self.__read_scoreline(regions.team1_score, regions.team2_score)
        atkside   = self.__read_atkside(regions.team1_side, regions.team2_side)
        time_read = self.__read_timer(regions.timer, get_timer_redperc(regions.timer))
        feed_read, image_lines = self.__read_feed(regions.kf_lines)

        headshots = [is_headshot(img_line, self.assets.headshot) for img_line in image_lines]
        self.__test_drawocr(image_lines, feed_read, headshots)

        print(f"\nTest: {scoreline=} | {atkside=} | {time_read} | ", end="")
        for line in feed_read:
            if len(line.results) == 1:
                print(f"{line.results[0]} -> {line.results[0]}", end="")
            elif len(line.results) == 2:
                headshot = "(X) " if line.headshot else ""
                print(f"{line.results[0]} -> {headshot}{line.results[1]}, ", end="")
            else:
                headshot = "(X) " if line.headshot else ""
                print(headshot, "|".join([str(res) for res in line.results]))
        print()

    def __test_drawocr(self, image_lines: list[np.ndarray], lines: list[OCRLine],
                       hs_rects: list[BBox_t | None]) -> None:
        line_cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] + [gen_random_colour() for _ in range(len(image_lines)-3)]
        for i, (image, line, col, hs) in enumerate(zip(image_lines, lines, line_cols, hs_rects)):
            rect_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            for ocres in line.results:
                cv2.rectangle(rect_img, ocres.rect[:2],  [ocres.rect[0]+ocres.rect[2], ocres.rect[1]+ocres.rect[3]], col, 2)
            if hs is not None:
                cv2.rectangle(rect_img, hs[:2],  [hs[0]+hs[2], hs[1]+hs[3]], (255, 225, 0), 2)
            
            cv2.imwrite(f"images/TEST_kfline{i+1}.jpg", rect_img)
