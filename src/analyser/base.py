import cv2
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import Optional

from capture import create_capture, RegionBBoxes, ImageRegions_t
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


class Analyser(ABC):
    """Main class `Analyser`
    Operates the main inference loop `run` and records match/round information"""

    def __init__(self, args: AnalyserArgs, config: Config):
        self.args = args
        self.config = config
        self.settings = create_settings(args.settings_path)
        self._debug_print("config_keys", f"Config\n{self.config}")

        self.capture = create_capture(self.config)

        if self.test_and_checks():
            return

        self.running = False
        self.tdelta = self.config.screenshot_period

        self.state = State(False, True, False)
        self.history = History()
        self.writer = create_writer(self.config)

        self.prog_bar = ProgressBar(add_postfix=self.config.debug.infer_time)
        self.current_time: Timestamp
        self.defuse_countdown_timer: Optional[float] = None

        self.ign_matrix = create_ignmatrix(self.config)
        self.ocr_engine = OCREngine.new(self.settings.ocr_engine, "en")
        self._verbose_print(0, self.ocr_engine.load_msg())

    ## ----- CHECK & TEST -----
    def test_and_checks(self) -> bool:
        ret_true = False
        if self.args.check_regions:
            self._check_regions()
            ret_true = True
        if self.args.test_regions:
            self._test_regions()
            ret_true = True

        return ret_true

    @abstractmethod
    def _check_regions(self) -> None:
        ...

    @abstractmethod
    def _test_regions(self) -> None:
        ...

    # ----- MAIN LOOP -----
    def run(self):
        """Main program loop, calls each _handle method every `tdelta` seconds"""
        self.running = True
        self._verbose_print(0, "Running...")

        self.timer = time()
        while self.running:
            if self.tdelta > 0 and self.timer + self.tdelta > time():
                continue

            infer_start = time()

            regions = self.capture.next(self._get_regions())
            self._handle_scoreline(regions)
            self._handle_timer(regions)
            self._handle_feed(regions)

            self._debug_infertime(time() - infer_start)
            self.timer = time()
            self.prog_bar.refresh()
    
    def _debug_infertime(self, dt: float) -> None:
        if self.config.debug.infer_time:
            self.prog_bar.set_postfix(f"{dt:.3f}s")
            self.prog_bar.refresh()
    
    ## ----- OCR -----
    def _screenshot_preprocess(self,
                               image: np.ndarray,
                               to_gray: bool = True,
                               denoise: bool = False,
                               squeeze_width: float = 1.0) -> np.ndarray:
        """
        To increase the accuracy of the EasyOCR readtext function, a few preprocessing techniques are used
          - RGB to Grayscale conversion
          - Denoise the image using fastNlMeansDenoising
          - Resize by factor `Config.SCREENSHOT_RESIZE` (normally 2-4)
          - Squeeze the width of the image, useful for scoreline OCR
        """
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if denoise:
            image = cv2.fastNlMeansDenoising(image, None, 5, 7, 21)

        scale_factor = self.config.screenshot_resize
        if squeeze_width != 1.0:
            sf_w, sf_h = scale_factor * squeeze_width, scale_factor
        else:
            sf_w = sf_h = scale_factor

        new_width = int(image.shape[1] * sf_w)
        new_height = int(image.shape[0] * sf_h)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _readtext(self, image: np.ndarray, prob = 0.0, allowlist: Optional[str] = None) -> list[str]:
        """Performs the EasyOCR inference and cleans the output based on the model's assigned probabilities and a threshold"""
        results = self.ocr_engine.read(image, allowlist=allowlist)
        return [out.text for out in results if out.prob > prob]
    
    def _readbatch(self, images: list[np.ndarray], prob = 0.0, allowlist: Optional[str] = None) -> list[list[str]]:
        results = self.ocr_engine.read_batch(images, allowlist=allowlist)
        return [[out.text for out in res if out.prob > prob] for res in results]


    ## ----- IN ROUND OCR FUNCTIONS -----
    @abstractmethod
    def _get_regions(self) -> RegionBBoxes:
        ...

    @abstractmethod
    def _handle_scoreline(self, regions: ImageRegions_t) -> None:
        ...
    
    @abstractmethod
    def _handle_timer(self, regions: ImageRegions_t) -> None:
        ...
    
    @abstractmethod
    def _handle_feed(self, regions: ImageRegions_t) -> None:
        ...

    ## ----- GAME STATE FUNCTIONS -----
    @abstractmethod
    def _new_round(self, score1: int, score2: int) -> None:
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
    
    def _stop_analyser(self) -> None:
        self.running = False
    
    def _get_last_winner(self) -> int:
        """The program cannot currently detect who wins the final round, so get the user to input that info"""
        if self.config.last_winner != Team.UNKNOWN:
            return self.config.last_winner

        while (winner := input("Who won the last round? (0/1) >> ")) not in "01":
            print("Invalid option, pick either 0 or 1")
            continue

        return int(winner)

    ## ----- PRINT FUNCTION -----
    def _verbose_print(self, verbose_value: int, *prompt) -> None:
        if self.args.verbose > verbose_value:
            self.prog_bar.print("Info:", *prompt)

    def _debug_print(self, key: str, *prompt) -> None:
        if self.config.__dict__.get(key, False):
            self.prog_bar.print("Debug:", *prompt)
