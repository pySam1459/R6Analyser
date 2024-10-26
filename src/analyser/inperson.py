import cv2
import numpy as np
from os import makedirs
from typing import Optional, Generator

from capture import Capture, InPersonRegions
from config import Config
from history import KFRecord
from ignmatrix import Player, Player_t
from ocr import OCRLineResult
from settings import Settings
from utils.cli import AnalyserArgs
from utils.enums import Team, WinCondition
from utils import *
from utils.constants import *

from .base import Analyser, State
from .killfeed import SmartKillfeed, KFRecord_t
from .template_matcher import TemplateMatcher


__all__ = ["InPersonAnalyser"]


class InPersonAnalyser(Analyser):
    NUM_LAST_SECONDS  = 4  ## number of seconds to continue reading killfeed after round end (reliability reasons)
    END_ROUND_SECONDS = 12 ## number of seconds to check no timer to determine round end

    def __init__(self, args: AnalyserArgs, config: Config, settings: Settings) -> None:
        super(InPersonAnalyser, self).__init__(args, config, settings, InPersonRegions)

        self.capture: Capture[InPersonRegions]
        self.skf = SmartKillfeed(self.config, self.history, self._add_record)
        self.atkside_matcher = TemplateMatcher(self.assets["atkside_template"])

        self.prog_bar = ProgressBar(add_postfix=self.config.debug.infer_time)

        self.last_kf_seconds = None
        self.end_round_seconds = None

    ## ----- SCORELINE -----
    def _handle_scoreline(self, regions: InPersonRegions) -> None:
        """Extracts the current scoreline visible and determines when a new rounds starts"""
        if not self.running or not self.state.end_round: return

        scoreline = self.__read_scoreline(regions.team0_score, regions.team1_score)
        if not scoreline:
            return

        if scoreline in self.history:
            self._fix_state()
            return

        self._new_round(scoreline)

        self.history.cround.atk_side = self.__read_atk_side(regions)
        self._verbose_print(1, f"Atk Side: {self.history.cround.atk_side}")
    
    def __read_scoreline(self, team0_score: np.ndarray, team1_score: np.ndarray) -> Optional[Scoreline]:
        left_text  = self.ocr_engine.read_score(team0_score)
        if left_text is None:
            return None

        right_text = self.ocr_engine.read_score(team1_score)
        if right_text is None:
            return None

        if not self.ocr_engine.has_colours:
            self.ocr_engine.set_colours(team0_score, team1_score)

        return Scoreline(left=int(left_text), right=int(right_text))
    
    def __read_atk_side(self, regions: InPersonRegions) -> Team:
        if self.atkside_matcher.match(regions.team1_side):
            return Team.TEAM0
        else:
            return Team.TEAM1


    ## ----- TIMER FUNCTION -----
    def _handle_timer(self, regions: InPersonRegions) -> None:
        """Reads and handles the timer information, used to determine when the bomb is planted and when the round ends"""
        if not self.running or not self.history.is_ready: return

        new_time, is_bomb_countdown = self.ocr_engine.read_timer(regions.timer)

        ## timer is showing
        if new_time is not None:
            self.__handle_timer_shown(new_time)
        
        ## The bomb uses a separate countdown timer as the visual timer cannot be accurately tracked
        elif is_bomb_countdown:
            self.__handle_bomb_countdown()

        ## TODO: sort out this seconds stuff
        elif self.last_kf_seconds is None and self.end_round_seconds is None and self.state.in_round:
            self.last_kf_seconds = self.timer.time
            self.end_round_seconds = self.timer.time
        
        if self.last_kf_seconds is not None and self.last_kf_seconds + InPersonAnalyser.NUM_LAST_SECONDS < self.timer.time:
            self.last_kf_seconds = None

        if (self.end_round_seconds is not None and
                self.end_round_seconds + InPersonAnalyser.END_ROUND_SECONDS < self.timer.time
                and self.__read_scoreline(regions.team0_score, regions.team1_score) is None):
            self._end_round()
            self.end_round_seconds = None

    def __handle_timer_shown(self, new_time: Timestamp) -> None:
        self.timer.ctime = new_time
        self.timer.defuse_countdown = None

        self.last_kf_seconds = None
        self.end_round_seconds = None
        
        self.prog_bar.set_time(new_time)
        self.prog_bar.refresh()

    def __handle_bomb_countdown(self) -> None:
        if self.timer.defuse_countdown is None: ## bomb planted
            self.timer.defuse_countdown = self.timer.time
            self.state.bomb_planted = True

            self.history.cround.bomb_planted_at = self.timer.ctime
            self._verbose_print(1, f"Bomb planted at: {self.timer.ctime}")
            self.prog_bar.bomb()
        
        elif self.history.cround.bomb_planted_at is not None:
            bpat_int = self.history.cround.bomb_planted_at.to_int()
            tdelta = int(self.timer.time - self.timer.defuse_countdown)
            self.timer.ctime = Timestamp.from_int(bpat_int - tdelta)
            self.prog_bar.set_time(int(self.config.defuser_timer-tdelta))
            self.prog_bar.refresh()


    ## ----- KILL FEED -----
    def _handle_feed(self, regions: InPersonRegions) -> None:
        """Handles the killfeed by reading the names, querying the ign matrix and the information to History"""
        if not self.running or not self.history.is_ready: return
        if not self.state.in_round and self.last_kf_seconds is None: return

        for line in self.__read_feed(regions.kf_lines):
            player, target = self.__get_players_from_ocrline(line)
            record_type = KFRecord_t.NORMAL

            if (player is None and line.left is None) and target is not None:
                player = target
                record_type = KFRecord_t.SUICIDE
            elif player is None or target is None:
                continue

            record = KFRecord(player, target, self.timer.ctime, line.headshot)
            self.skf.add(record, record_type)

    def __read_feed(self, image_lines: list[np.ndarray]) -> Generator[OCRLineResult, None, None]:
        for img_line in image_lines:
            line_out = self.ocr_engine.read_kfline(img_line, self.ign_matrix.charlist)
            if line_out is None:
                break
            yield line_out

    def __get_players_from_ocrline(self, line: OCRLineResult) -> tuple[Optional[Player], Optional[Player]]:
        target = self.ign_matrix.get(line.right)
        if line.left is None:
            return None, target

        player = self.ign_matrix.get(line.left)
        return player, target

    def _add_record(self, record: KFRecord) -> None:
        self.history.cround.killfeed.append(record)

        if record.player.type == Player_t.INFER or record.target.type == Player_t.INFER:
            ## TODO: add read_kfline teams here
            self.ign_matrix.update_mats(record.player.ign, record.target.ign)
            self.ign_matrix.update_mats(record.target.ign, record.player.ign)

        self._verbose_print(2, record.to_str())
        self.prog_bar.set_desc(record.to_str(show_time=False))
        self.prog_bar.refresh()


    # ----- GAME STATE -----
    def __finish_last_round(self, new_sl: Scoreline, save = True) -> None:
        """
        To save the correct information for each round, this method must be called just before the start of a new round,
          so that the scoreline and winner attributes can be used
        """
        if not self.history.is_ready: return

        ## infer winner of previous round based on new scoreline
        cround = self.history.cround
        if cround.winner == Team.UNKNOWN and cround.scoreline is not None:
            if new_sl.left > cround.scoreline.left:
                cround.winner = Team.TEAM0
            else:
                cround.winner = Team.TEAM1

        win_con = cround.get_wincon(self.ign_matrix)
        cround.win_condition = win_con

        reat = cround.round_end_at
        if win_con == WinCondition.DISABLED_DEFUSER:
            cround.disabled_defuser_at = reat
            self._verbose_print(1, f"Disabled defuser at: {reat}")

        self._verbose_print(0, f"Team {cround.winner} wins round {self.history.roundn} by {win_con.value} at {reat}.")
        if save:
            self.writer.write(self.history, self.ign_matrix)

    def _new_round(self, sl: Scoreline) -> None:
        """
        When a new round starts, this method is called, initialising a new round history
        The parameters `score1` and `score2` are the current scores displayed at the start of a new round
        """
        if sl in self.history:
            return

        self.__finish_last_round(sl, save=True)
        self.state = State(in_round=True, end_round=False, bomb_planted=False)

        new_round = sl.total + 1
        self.history.new_round(new_round)
        self.history.cround.scoreline = sl
        self.skf.reset()

        self._verbose_print(1, f"New Round: {new_round} | Scoreline: {sl.left}-{sl.right}")
        self.prog_bar.new_round(sl)
    
    def _end_round(self) -> None:
        """A game state method called when the program determines that the current round has ended"""
        self.history.cround.round_end_at = self.timer.ctime
        self.state = State(in_round=False, end_round=True, bomb_planted=False)

        self.prog_bar.set_time(0)
        self.prog_bar.set_desc("End of Round")
        self.prog_bar.refresh()

        if self._is_end_of_game():
            self._end_game()

    def _is_end_of_game(self) -> bool:
        sl = self.history.cround.scoreline
        if sl is None:
            return False

        max_non_ot = self.config.max_rounds - self.config.overtime_rounds
        non_ot_perside = max_non_ot // 2
        is_ot = sl.total >= max_non_ot

        return (
            self.history.roundn >= self.config.max_rounds
                or (not is_ot and sl.max == non_ot_perside+1)
                or (is_ot and sl.diff == 2)
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
            self.state.bomb_planted = self.ocr_engine.get_is_bomb_countdown(timer_region)
        
        self.history.fix_round()

    ## ----- CHECK & TEST -----
    def _check_regions(self, regions: InPersonRegions) -> None:
        """Called when the --check-regions flag is present in the program call, saves the screenshotted regions as jpg images"""
        img_dir = DEFAULT_IMAGE_DIR / self.config.name
        if not img_dir.exists():
            makedirs(img_dir)

        self._verbose_print(0, "Saving check images")

        def save(name: str, _img: np.ndarray) -> None:
            if _img.shape[0] * _img.shape[1] == 0:
                return
            save_path = img_dir / f"{name}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(_img, cv2.COLOR_RGB2BGR))

        for name, img in regions.model_dump(exclude_none=True).items():
            if isinstance(img, list):
                for i, subimg in enumerate(img):
                    save(f"{name}_{i}", subimg)
            else:
                save(name, img)

    def _test_regions(self, regions: InPersonRegions) -> None:
        """
        This method is called when the `--test-regions` flag is added to the program call,
        runs inference on a single screenshot.
        """
        scoreline = self.__read_scoreline(regions.team0_score, regions.team1_score)
        atkside   = self.__read_atk_side(regions)
        time_read, is_bomb_countdown = self.ocr_engine.read_timer(regions.timer)
        print(f"\nTest: {scoreline=} | {atkside=} | {time_read} | {is_bomb_countdown=}")

        for line in self.__read_feed(regions.kf_lines):
            if line.headshot:
                print(f"{line.left} -> (X) {line.right}")
            else:
                print(f"{line.left} -> {line.right}")
