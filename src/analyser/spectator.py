from capture import RegionBBoxes, SpectatorRegions
from config import Config
from utils.cli import AnalyserArgs
from utils import *

from .base import Analyser


class SpectatorAnalyser(Analyser):
    def __init__(self, args: AnalyserArgs, config: Config) -> None:
        super(SpectatorAnalyser, self).__init__(args, config)

    def _get_regions(self) -> RegionBBoxes:
        return RegionBBoxes()

    ## ----- IN ROUND OCR FUNCTIONS -----
    def _handle_scoreline(self, regions: SpectatorRegions) -> None:
        ...
    
    def _handle_timer(self, regions: SpectatorRegions) -> None:
        ...
    
    def _handle_feed(self, regions: SpectatorRegions) -> None:
        ...

    ## ----- GAME STATE FUNCTIONS -----
    def _new_round(self, score1: int, score2: int) -> None:
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
