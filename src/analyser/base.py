from abc import ABC, abstractmethod
from time import perf_counter
from pydantic.dataclasses import dataclass
from typing import Callable

from args import AnalyserArgs
from assets import Assets
from capture import create_capture
from capture.regions import Regions
from config import Config
from history import History
from ignmatrix import create_ignmatrix
from ocr import OCREngine
from scheduler import create_scheduler
from settings import Settings
from utils.enums import Team, CaptureTimeType
from utils.keycheck import send_inc_update
from utils.constants import RED, WHITE
from writer import create_writer
from utils import *


__all__ = [
    "Analyser",
    "State"
]


@dataclass
class State:
    """Program state, should only be modified by new_round, end_round, fix_state methods"""
    in_round:     bool
    end_round:    bool  ## ready's the program to start a new round
    bomb_planted: bool


class RoundTimer:
    def __init__(self, get_time_func: Callable[[], float]) -> None:
        self._get_time = get_time_func

        self._current_time: Timestamp
        self._defuse_countdown_timer: Optional[float] = None

    @property
    def time(self) -> float:
        return self._get_time()

    @property
    def ctime(self) -> Timestamp:
        return self._current_time
    
    @ctime.setter
    def ctime(self, value: Timestamp) -> None:
        self._current_time = value
    
    @property
    def defuse_countdown(self) -> Optional[float]:
        return self._defuse_countdown_timer
    
    @defuse_countdown.setter
    def defuse_countdown(self, value: Optional[float]) -> None:
        self._defuse_countdown_timer = value


class Analyser(ABC):
    """Main class `Analyser`
    Operates the main inference loop `run` and records match/round information"""

    def __init__(self, args: AnalyserArgs,
                 config: Config,
                 settings: Settings):

        self.args = args
        self.config = config
        self.settings = settings
        self._debug_print("config_keys", f"Config\n{config}")

        self.assets = Assets(self.settings.assets_path)
        self.capture = create_capture(config)
        self.ocr_engine = OCREngine(config.ocr, settings, self.assets)

        self.ign_matrix = create_ignmatrix(config)
        self.history = History()
        self.writer = create_writer(config)

        self.running = False
        self.state = State(in_round=False, end_round=True, bomb_planted=False)
        self.timer = RoundTimer(self.capture.get_time)

        self.prog_bar: ProgressBar

        self.scheduler = create_scheduler(config, self.capture, {
            "scoreline": self._handle_scoreline,
            "timer":     self._handle_timer,
            "killfeed":  self._handle_feed
        })

    # ----- MAIN LOOP -----
    def run(self):
        self.running = True
        self._verbose_print(0, "Running...")

        while self.running:
            start = perf_counter()
            stop_signal = self.scheduler.tick()
            if stop_signal:
                self._end_game()
                return
            
            if (infer_time := perf_counter() - start) > 0.001:
                self._debug_postfix(f"{infer_time:.3f}s | {self.capture.get_time()}")
            self.prog_bar.refresh()


    ## ----- CHECK & TEST -----
    def __get_start(self) -> Timestamp:
        if self.args.start is not None:
            return self.args.start

        while True:
            timestamp_s = input("Timestamp >> ")
            if not timestamp_s:
                exit()

            ts = Timestamp.from_str(timestamp_s)
            if ts is not None:
                return ts
            else:
                print("Invalid Timestamp")

    def test_and_checks(self) -> None:
        if self.capture.time_type == CaptureTimeType.FPS:
            dt = self.__get_start().to_int()
            regions = self.capture.next(dt, jump=True)
        else:
            regions = self.capture.next()

        if regions is None:
            self._verbose_print(0, "Error Cannot capture regions!")
            return

        if self.args.check_regions:
            self._check_regions(regions)

        if self.args.test_regions:
            self._test_regions(regions)

    @abstractmethod
    def _check_regions(self, regions: Regions) -> None:
        ...

    @abstractmethod
    def _test_regions(self, regions: Regions) -> None:
        ...

    ## ----- IN ROUND OCR FUNCTIONS -----
    @abstractmethod
    def _handle_scoreline(self, regions: Regions) -> None:
        ...

    @abstractmethod
    def _handle_timer(self, regions: Regions) -> None:
        ...

    @abstractmethod
    def _handle_feed(self, regions: Regions) -> None:
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
        self.capture.stop()
        self.ocr_engine.stop()
    
    def _get_last_winner(self) -> Team:
        """The program cannot currently detect who wins the final round, so get the user to input that info"""
        if self.config.last_winner != Team.UNKNOWN:
            return self.config.last_winner

        while (winner := input("Who won the last round? (0/1) >> ")) not in "01":
            print("Invalid option, pick either 0 or 1")
            continue

        return Team(int(winner))

    def _send_key_update(self) -> None:
        if not send_inc_update(self.args.software_key, ncalls=1):
            print(f"\n\n{RED}Could not validate software key!{WHITE}\nMake sure you are connected to the internet.\n")
            exit()

    ## ----- PRINT FUNCTION -----
    @property
    def __print(self) -> Callable[..., None]:
        if getattr(self, "prog_bar", None) is None:
            return print
        else:
            return self.prog_bar.print

    def _verbose_print(self, verbose_value: int, *prompt, **kwargs) -> None:
        if self.args.verbose > verbose_value:
            self.__print("Info:", *prompt, **kwargs)

    def _debug_print(self, key: str, *prompt, **kwargs) -> None:
        if self.config.__dict__.get(key, False):
            self.__print("Debug:", *prompt, **kwargs)
    
    def _debug_postfix(self, postfix: str) -> None:
        self.prog_bar.set_postfix(postfix)
        self.prog_bar.refresh()

    def _debug_infertime(self, dt: float) -> None:
        if self.config.debug.infer_time:
            self._debug_postfix(f"{dt:.3f}s")
