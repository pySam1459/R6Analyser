from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import Optional, Generic, TypeVar, Type

from assets import Assets
from capture import Timer, RegionBBoxes, InPersonRegions, SpectatorRegions, create_capture
from config import Config
from history import History
from ignmatrix import create_ignmatrix
from ocr import OCREngine
from settings import create_settings
from utils.enums import Team
from utils.cli import AnalyserArgs
from writer import create_writer
from utils import *


__all__ = ["Analyser", "State"]


@dataclass
class State:
    """Program state, should only be modified by new_round, end_round, fix_state methods"""
    in_round:     bool
    end_round:    bool  ## ready's the program to start a new round
    bomb_planted: bool


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class Analyser(Generic[T], ABC):
    """Main class `Analyser`
    Operates the main inference loop `run` and records match/round information"""

    def __init__(self, args: AnalyserArgs, config: Config, region_type: Type[T]):
        self.args = args
        self.config = config
        self.settings = create_settings(args.sets_path)
        self._debug_print("config_keys", f"Config\n{self.config}")

        self.assets = self.__load_assets()
        self.capture = create_capture(self.config, region_type)
        self.ocr_engine = OCREngine(self.settings)

        self.ign_matrix = create_ignmatrix(self.config)
        self.history = History()
        self.writer = create_writer(self.config)

        self.test_and_checks()
        if self.is_a_test:
            return

        self.running = False
        self.state = State(in_round=False, end_round=True, bomb_planted=False)
        self.timer = Timer(self.capture.get_time, period=self.config.capture.period)

        self.prog_bar = ProgressBar(add_postfix=self.config.debug.infer_time)
        self.current_time: Timestamp ## TODO: move into self.timer
        self.defuse_countdown_timer: Optional[float] = None        

        self.handlers = [
            self._handle_scoreline,
            self._handle_timer,
            self._handle_feed,
        ]
    
    def __load_assets(self) -> Assets:
        """Load the assets used in image detection"""
        return (
            Assets()
                .resize_height("headshot", self.config.capture.regions.kf_line[3])
        )

    # ----- MAIN LOOP -----
    def run(self):
        """Main program loop, calls the handlers every `capture.period` seconds"""
        self.running = True
        self._verbose_print(0, "Running...")

        self.timer.update()
        while self.running:
            if not self.timer.step():
                continue

            ## time.time() is correct here since its not in-game time
            __infer_start = time()

            regions = self.capture.next()
            for handler_func in self.handlers:
                handler_func(regions)

            self._debug_infertime(time() - __infer_start)
            self.timer.update()
            self.prog_bar.refresh()

    ## ----- CHECK & TEST -----
    def test_and_checks(self) -> None:
        self.is_a_test = False
        if self.args.check_regions:
            self._check_regions()
            self.is_a_test = True

        if self.args.test_regions:
            self._test_regions()
            self.is_a_test = True

    @abstractmethod
    def _check_regions(self) -> None:
        ...

    @abstractmethod
    def _test_regions(self) -> None:
        ...

    ## ----- IN ROUND OCR FUNCTIONS -----
    @abstractmethod
    def _handle_scoreline(self, regions: T) -> None:
        ...

    @abstractmethod
    def _handle_timer(self, regions: T) -> None:
        ...

    @abstractmethod
    def _handle_feed(self, regions: T) -> None:
        ...

    ## ----- GAME STATE FUNCTIONS -----
    @abstractmethod
    def _new_round(self, sl: Scoreline) -> None:
        ...

    @abstractmethod
    def _end_round(self) -> None:
        ...

    @abstractmethod
    def _fix_state(self) -> None:
        ...    

    # ----- ENG OF PROGRAM / SAVING -----
    @abstractmethod
    def _end_game(self) -> None:
        ...
    
    def stop(self) -> None:
        self.running = False
    
    def _get_last_winner(self) -> Team:
        """The program cannot currently detect who wins the final round, so get the user to input that info"""
        if self.config.last_winner != Team.UNKNOWN:
            return self.config.last_winner

        while (winner := input("Who won the last round? (0/1) >> ")) not in "01":
            print("Invalid option, pick either 0 or 1")
            continue

        return Team(int(winner))

    ## ----- PRINT FUNCTION -----
    @property
    def __print(self) -> Callable[..., None]:
        if getattr(self, "prog_bar", None) is None:
            return print
        else:
            return self.prog_bar.print

    def _verbose_print(self, verbose_value: int, *prompt) -> None:
        if self.args.verbose > verbose_value:
            self.__print("Info:", *prompt)

    def _debug_print(self, key: str, *prompt) -> None:
        if self.config.__dict__.get(key, False):
            self.__print("Debug:", *prompt)

    def _debug_infertime(self, dt: float) -> None:
        if self.config.debug.infer_time:
            self.prog_bar.set_postfix(f"{dt:.3f}s")
            self.prog_bar.refresh()
