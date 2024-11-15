from args import AnalyserArgs
from capture.regions import Regions
from config import Config
from utils import *

from .base import Analyser


class SpectatorAnalyser(Analyser):
    def __init__(self, args: AnalyserArgs, config: Config) -> None:
        super(SpectatorAnalyser, self).__init__(args, config)

        self.prog_bar = ProgressBar(add_postfix=self.config.debug.infer_time)

    ## ----- IN ROUND OCR FUNCTIONS -----
    def _handle_scoreline(self, regions: Regions) -> None:
        ...
    
    def _handle_timer(self, regions: Regions) -> None:
        ...
    
    def _handle_feed(self, regions: Regions) -> None:
        ...

    ## ----- GAME STATE FUNCTIONS -----
    def _new_round(self, sl: Scoreline) -> None:
        ...

    def _end_round(self) -> None:
        ...    
    
    def _end_game(self) -> None:
        ...
    
    def _fix_state(self) -> None:
        ...
    
    ## ----- CHECK & TEST -----
    def _check_regions(self) -> None:
        ...
    
    def _test_regions(self) -> None:
        ...
