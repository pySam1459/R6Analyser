import cv2
import easyocr
import numpy as np
import pyautogui
import sys
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from io import StringIO
from os import mkdir
from os.path import join, exists
from re import search, fullmatch
from time import time
from tqdm import tqdm
from typing import TypeAlias, Optional

from ignmatrix import IGNMatrix
from history import History, KFRecord, WinCondition
from writer import Writer
from utils import *


class TimerFormat(Enum):
    """Enum used to declare what type of formatting the timer requires"""
    FULL    = 0   ## 2:59 - 0:10
    SECONDS = 1   ## 9:99 - 0:00   -> 0:09 - 0:00


@dataclass
class State:
    """Program state, should only be modified by new_round, end_round, fix_state methods"""
    in_round:     bool
    end_round:    bool ## ready's the program to start a new round
    bomb_planted: bool


class ProgressBar:
    def __init__(self, verbose: int, is_test: bool = False) -> None:
        bar_format = "{desc}|{bar}"
        if verbose == 3:
            bar_format += "|{postfix}"

        self.__tqdmbar = tqdm(total=180, bar_format=bar_format)
        self.__header = ""
        self.__time = ""
        self.__value = "-"
        
        if is_test: ## for a slightly cleaner console output during --test
            self.__tqdmbar.refresh = lambda *args, **kwargs: None

    def set_total(self, value: int) -> None:
        self.__tqdmbar.total = value

    def set_desc(self, value: str) -> None:
        self.__value = value
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {value} ", refresh=False)

    def set_time(self, value: Timestamp | int) -> None:
        assert type(value) in [Timestamp, int], f"Invalid value type: {type(value)}"
        if type(value) == Timestamp:
            self.__time = str(value)
            value = value.to_int()
        elif type(value) == int:
            self.__time = str(Timestamp.from_int(value))

        self.__tqdmbar.n = value
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {self.__value} ", refresh=False)

    def set_header(self, nround: int, s1: int, s2: int) -> None:
        self.__header = f"{nround}/{s1}:{s2}"
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {self.__value} ", refresh=False)
    
    def set_postfix(self, value: str) -> None:
        self.__tqdmbar.set_postfix_str(value, refresh=False)

    def refresh(self) -> None:
        self.__tqdmbar.refresh()

    def close(self) -> None:
        self.__tqdmbar.close()

    def reset(self) -> None:
        self.__tqdmbar.n = 180
        self.__tqdmbar.total = 180
        self.__value = "-"
        self.__time = "3:00"
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {self.__value} ")

    def bomb(self) -> None:
        self.set_time(45)
        self.set_total(45)
        self.refresh()

    @staticmethod
    def print(*prompt: object, sep: Optional[str] = " ", end: Optional[str] = "\n", flush: bool = False) -> None:
        """Replaces the builtin `print` function with one which works with the tqdm progress bar"""
        temp_out = StringIO()
        sys.stdout = temp_out
        print(*prompt, sep=sep, end=end, flush=flush)
        sys.stdout = sys.__stdout__
        tqdm.write(temp_out.getvalue(), end='')


@dataclass
class OCResult:
    """
    A dataclass containing the data of a single easyOCR reading, the data stored includes
        rect - rectangle bounding the text, text, prob - probability assigned by the easyOCR engine, eval_score - IGNMatrix.evaluate score
    """
    rect: list[int]
    text: str
    prob: float
    eval_score: Optional[float] = None

    def __init__(self, ocr_result: tuple[list[list[int]], str, float]):
        self.rect = bbox_to_rect(ocr_result[0])
        self.text = ocr_result[1]
        self.prob = ocr_result[2]

    def eval(self, ign_matrix: IGNMatrix) -> float:
        """Sets the eval_score of self.text given an IGNMatrix"""
        self.eval_score = ign_matrix.evaluate(self.text)
        return self.eval_score
    
    def join(self, other: 'OCResult') -> 'OCResult':
        """
        Combines 2 OCResult objects into 1,
          a solution to a issue with the EasyOCR's detection engine where it wouldn't properly detect a name with an underscore in it
        """
        x, y = min(self.rect[0], other.rect[0]), min(self.rect[1], other.rect[1])
        x2, y2 = max(self.rect[0]+self.rect[2], other.rect[0]+other.rect[2]), max(self.rect[1]+self.rect[3], other.rect[1]+other.rect[3])
        join_box = [[x, y], [x2, y], [x2, y2], [x, y2]]

        join_str  = self.text + "_" + other.text
        join_prob = min(self.prob, other.prob)
        return OCResult((join_box, join_str, join_prob))
    
    def __str__(self) -> str:
        if self.eval_score is not None: return f"{self.text}|eval={self.eval_score*100:.2f}"
        else: return f"{self.text}|prob={self.prob*100:.2f}"

    __repr__ = __str__


@dataclass
class OCRLine:
    """A helper dataclass containing the OCResults and other info from a single killfeed line"""
    ocr_results: list[OCResult]
    headshot: bool = False


class Analyser(ABC):
    """
    Main class `Analyser`
    Operates the main inference loop `run` and records match/round information
    """
    KF_ALLOWLIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    
    def __init__(self, args: Namespace):
        self.config: dict = args.config
        self.verbose: int = args.verbose
        self.prog_args = args
        self._debug_print(f"Config Keys -", list(self.config.keys()))

        if args.check:
            self.check()
            if not args.test: sys.exit()

        self.running = False
        self.tdelta: float = self.config["SCREENSHOT_PERIOD"]

        self.state = State(False, True, False)
        self.history = History()
        self.writer = Writer.new(args.save, self.config, args.append_save)
        self.tempfeed: dict[KFRecord, int] = {}

        self.prog_bar = ProgressBar(self.verbose, args.test)
        self.current_time: Timestamp
        self.defuse_countdown_timer: Optional[float] = None
        
        self.ign_matrix = IGNMatrix.new(self.config["IGNS"], self.config["IGN_MODE"])
        self.reader = easyocr.Reader(['en'], gpu=not args.cpu)
        self._verbose_print(0, "EasyOCR Reader model loaded")

    ## ----- CHECK & TEST-----
    @abstractmethod
    def check(self) -> None:
        ...

    @abstractmethod
    def test(self) -> None:
        ...

    # ----- SCREENSHOTS -----
    @abstractmethod
    def _get_ss_region_keys(self) -> list[str]:
        ...

    Regions_t: TypeAlias = dict[str, np.ndarray]
    def _get_ss_regions(self, region_keys: Optional[list[str]] = None) -> Regions_t:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        if region_keys is None:
            region_keys = self._get_ss_region_keys()

        screenshot = pyautogui.screenshot(allScreens=True)
        return {
            region: np.array(screenshot.crop(Analyser.convert_region(self.config[region])), copy=False) # type: ignore the line
            for region in region_keys
        }

    @staticmethod
    def convert_region(region: list[int]) -> tuple[int,int,int,int]:
        """Converts (X,Y,W,H) -> (Left,Top,Right,Bottom)"""
        left, top, width, height = region
        return (left, top, left + width, top + height)

    # ----- MAIN LOOP -----
    def run(self):
        """Main program loop, calls each _handle method every `tdelta` seconds"""
        self.running = True
        self._verbose_print(0, "Running...")

        self.timer = time()
        while self.running:
            if self.tdelta > 0 and self.timer + self.tdelta > time():
                continue

            __infer_start = time()

            regions = self._get_ss_regions()
            self._handle_scoreline(regions)
            self._handle_timer(regions)
            self._handle_feed(regions)

            self._debug_infertime(time() - __infer_start)
            self.timer = time()
            self.prog_bar.refresh()
    
    def _debug_infertime(self, dt: float) -> None:
        if self.verbose == 3:
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

        scale_factor = self.config["SCREENSHOT_RESIZE"]
        if squeeze_width != 1.0:
            sf_w, sf_h = scale_factor * squeeze_width, scale_factor
        else:
            sf_w = sf_h = scale_factor

        new_width = int(image.shape[1] * sf_w)
        new_height = int(image.shape[0] * sf_h)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _readtext(self, image: np.ndarray, prob: float = 0.0, allowlist: Optional[str] = None) -> list[str]:
        """Performs the EasyOCR inference and cleans the output based on the model's assigned probabilities and a threshold"""
        results = self.reader.readtext(image, allowlist=allowlist)
        return [out[1] for out in results if out[2] > prob]


    ## ----- IN ROUND OCR FUNCTIONS -----
    @abstractmethod
    def _handle_scoreline(self, regions: Regions_t) -> None:
        ...
    
    @abstractmethod
    def _handle_timer(self, regions: Regions_t) -> None:
        ...
    
    @abstractmethod
    def _handle_feed(self, regions: Regions_t) -> None:
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
    
    def _ask_winner(self) -> int:
        """The program cannot currently detect who wins the final round, so get the user to input that info"""
        while (winner := input("Who won the last round? (0/1) >> ")) not in "01":
            print("Invalid option, pick either 0 or 1")
            continue

        return int(winner)

    ## ----- PRINT FUNCTION -----
    def _verbose_print(self, verbose_value: int, *prompt) -> None:
        if self.verbose > verbose_value:
            try:
                self.prog_bar.print("Info:", *prompt)
            except AttributeError:
                print("Info:", *prompt)

    def _debug_print(self, *prompt) -> None:
        if self.verbose == 3:
            try:
                self.prog_bar.print("Debug:", *prompt)
            except AttributeError:
                print("Debug:", *prompt)


class InPersonAnalyser(Analyser):
    NUM_LAST_SECONDS = 4   ## number of seconds to continue reading killfeed after round end (reliability reasons)
    END_ROUND_SECONDS = 12 ## number of seconds to check no timer to determine round end
    
    NUM_KF_LINES = 3
    KF_BUF = 4  ## number of pixels buffer around each KF_LINE
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION",
                          "TEAM1_SIDE_REGION", "TEAM2_SIDE_REGION",
                          "TIMER_REGION"] + [f"KF_LINE{i+1}_REGION" for i in range(NUM_KF_LINES)]
    NON_NAMES = ["has found the bomb", "Friendly Fire has been activated for"]

    SCORELINE_PROB = 0.25
    TIMER_PROB = 0.35
    KF_PROB = 0.10

    PROXIMITY_DIST = 35

    RED_THRESHOLD = 0.73
    RED_RGB_SPACE = np.array([ ## Defines the range for red color in HSV space
        [240, 10, 10],
        [255, 35, 35]])
    
    ASSET_LIST = ["atkside_icon.jpg", "headshot.jpg"]

    def __init__(self, args) -> None:
        super(InPersonAnalyser, self).__init__(args)

        self.last_kf_seconds = None
        self.end_round_seconds = None
        self.last_seconds_count = 0
        self.timer_format = TimerFormat.FULL

        self.__set_kfline_config()
        self.assets = self.__load_assets()
    
    def __load_assets(self) -> dict[str, np.ndarray]:
        """Load the assets used in image detection"""
        assets: dict[str, np.ndarray] = {}
        for asset_filename in InPersonAnalyser.ASSET_LIST:
            filepath = join("assets", asset_filename)
            assert exists(filepath), f"{filepath} does not exist!"
            
            filename, _ = asset_filename.rsplit(".", maxsplit=1)
            assets[filename] = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        def __resize(key: str, dsize: list[int]) -> np.ndarray:
            return cv2.resize(assets[key], dsize, interpolation=cv2.INTER_LINEAR)
        
        assets["atkside_icon"] = __resize("atkside_icon", self.config["TEAM1_SIDE_REGION"][2:])
        
        h,w = assets["headshot"].shape
        sf = self.config["SCREENSHOT_RESIZE"] * self.config["KF_LINE_REGION"][3] / h
        assets["headshot"] = __resize("headshot", [int(sf*w), int(sf*h)])

        return assets
    
    def __set_kfline_config(self) -> None:
        """Create k new regions for each killfeed line"""
        x,y,w,h = self.config["KF_LINE_REGION"]
        buf = InPersonAnalyser.KF_BUF
        for i in range(InPersonAnalyser.NUM_KF_LINES):
            region = [x, y-int(h*1.25*i)-buf, w, h+buf*2]
            self.config[f"KF_LINE{i+1}_REGION"] = region

    def _get_ss_region_keys(self) -> list[str]:
        return InPersonAnalyser.SCREENSHOT_REGIONS


    ## ----- SCORELINE -----
    def _handle_scoreline(self, regions: Analyser.Regions_t) -> None:
        """Extracts the current scoreline visible and determines when a new rounds starts"""
        if not self.state.end_round: return

        scores = self.__read_scoreline(regions["TEAM1_SCORE_REGION"], regions["TEAM2_SCORE_REGION"])
        if not scores: return

        new_roundn = sum(scores)+1
        if new_roundn in self.history:
            self._fix_state()
            return

        self._new_round(*scores)

        atkside = self.__read_atkside(regions["TEAM1_SIDE_REGION"], regions["TEAM2_SIDE_REGION"])
        self.history.cround.atk_side = atkside
        self._verbose_print(1, f"Atk Side: {atkside}")
    
    def __read_scoreline(self, scoreline1: np.ndarray, scoreline2: np.ndarray) -> Optional[tuple[int, int]]:
        """Reads the scoreline from the 2 TEAM1/2_SCORE_REGION screenshot"""
        img1 = self._screenshot_preprocess(scoreline1, to_gray=True, squeeze_width=0.65)
        img2 = self._screenshot_preprocess(scoreline2, to_gray=True, squeeze_width=0.65)

        allowlist = "0123456789"
        results = self._readtext(img1, InPersonAnalyser.SCORELINE_PROB, allowlist=allowlist) + self._readtext(img2, InPersonAnalyser.SCORELINE_PROB, allowlist=allowlist)
        if len(results) != 2:
            return None
        if not fullmatch(r"^\d+$", results[0]) or not fullmatch(r"^\d+$", results[1]):
            return None
        
        return (int(results[0]), int(results[1]))
    
    def __read_atkside(self, side1: np.ndarray, side2: np.ndarray) -> Optional[int]:
        """Matches the side icon next to the scoreline with `res/swords.jpg` to determine which side is attack."""
        icon1 = self._screenshot_preprocess(side1, to_gray=True, denoise=True)
        icon2 = self._screenshot_preprocess(side2, to_gray=True, denoise=True)

        res_icon1 = cv2.matchTemplate(self.assets["atkside_icon"], icon1, cv2.TM_CCOEFF_NORMED)
        res_icon2 = cv2.matchTemplate(self.assets["atkside_icon"], icon2, cv2.TM_CCOEFF_NORMED)
        ## TODO: prob only need to match 1 side, then threshold
        # Get the maximum match value for each icon
        _, max_val_icon1, _, _ = cv2.minMaxLoc(res_icon1)
        _, max_val_icon2, _, _ = cv2.minMaxLoc(res_icon2)

        # Decide which icon matches best
        if max_val_icon1 >= max_val_icon2:
            return 0
        elif max_val_icon1 < max_val_icon2:
            return 1


    ## ----- TIMER FUNCTION -----
    def _handle_timer(self, regions: Analyser.Regions_t) -> None:
        """Reads and handles the timer information, used to determine when the bomb is planted and when the round ends"""
        if not self.history.is_ready: return

        timer_image = regions["TIMER_REGION"]
        new_time = self.__read_timer(timer_image)

        ## timer is showing
        if new_time is not None:
            self.current_time = new_time
            self.defuse_countdown_timer = None

            self.last_kf_seconds = None
            self.end_round_seconds = None
            
            self.prog_bar.set_time(new_time)
            self.prog_bar.refresh()
        
        ## The bomb uses a separate countdown timer as the visual timer cannot be accurately tracked
        elif self.__is_bomb_countdown(timer_image):
            if self.defuse_countdown_timer is None: ## bomb planted
                self.defuse_countdown_timer = time()
                self.state.bomb_planted = True

                self.history.cround.bomb_planted_at = self.current_time
                self._verbose_print(1, f"Bomb planted at: {self.current_time}")
                self.prog_bar.bomb()
            
            elif self.history.cround.bomb_planted_at is not None:
                bpat_int = self.history.cround.bomb_planted_at.to_int()
                tdelta = int(time() - self.defuse_countdown_timer)
                self.current_time = Timestamp.from_int(bpat_int - tdelta)
                self.prog_bar.set_time(int(45-tdelta))
                self.prog_bar.refresh()

        elif self.last_kf_seconds is None and self.end_round_seconds is None and self.state.in_round:
            self.last_kf_seconds = time()
            self.end_round_seconds = time()
        
        if self.last_kf_seconds is not None and self.last_kf_seconds + InPersonAnalyser.NUM_LAST_SECONDS < time():
            self.last_kf_seconds = None

        if self.end_round_seconds is not None and self.end_round_seconds + InPersonAnalyser.END_ROUND_SECONDS < time() \
                and self.__read_scoreline(regions["TEAM1_SCORE_REGION"], regions["TEAM2_SCORE_REGION"]) is None:
            self._end_round()
            self.end_round_seconds = None

    
    def __read_timer(self, image: np.ndarray) -> Optional[Timestamp]:
        """
        Reads the current time displayed in the region `TIMER_REGION`
        If the timer is not present, None is returned
        """
        image = self._screenshot_preprocess(image, to_gray=True, denoise=True, squeeze_width=0.75)
        results = self._readtext(image, prob=InPersonAnalyser.TIMER_PROB, allowlist="0123456789:.")
        if len(results) == 0: return None

        result = results[0] if len(results) == 1 else "".join(results)
        time1 = search(r"([0-2]).?([0-5]\d)$", result) ## 2:59 - 0:10
        time2 = search(r"(\d).?\d\d", result)          ## 9:99 - 0:00

        if time1 is None and time2 is not None and self.timer_format != TimerFormat.SECONDS:
            self.last_seconds_count += 1
            if self.last_seconds_count > 4:
                self.timer_format = TimerFormat.SECONDS
                self.last_seconds_count = 0

        if self.timer_format == TimerFormat.FULL:
            if time1 is not None:
                return Timestamp(int(time1.group(1)), int(time1.group(2)))
        elif self.timer_format == TimerFormat.SECONDS or time1 is None:
            if time2 is not None:
                return Timestamp(0, int(time2.group(1)))

        return None

    
    def __is_bomb_countdown(self, image: np.ndarray) -> bool:
        """
        When a bomb is planted, the timer is replaced with a majority red circular countdown
        This method detects when the bomb defuse countdown is shown using a majority red threshold
        """
        mask: np.ndarray = cv2.inRange(image, InPersonAnalyser.RED_RGB_SPACE[0], InPersonAnalyser.RED_RGB_SPACE[1])

        # Calculate the percentage of red in the image
        red_percentage = np.sum(mask > 0) / mask.size
        self._debug_print(f"{red_percentage=}")
        return bool(red_percentage > InPersonAnalyser.RED_THRESHOLD)


    ## ----- KILL FEED -----
    def _handle_feed(self, regions: Analyser.Regions_t) -> None:
        """Handles the killfeed by reading the names, querying the ign matrix and the information to History"""
        if not self.history.is_ready: return
        if not self.state.in_round and self.last_kf_seconds is None: return

        image_lines = [regions[f"KF_LINE{i+1}_REGION"] for i in range(InPersonAnalyser.NUM_KF_LINES)]
        for line in self.__read_feed(image_lines)[0]:
            if len(line.ocr_results) == 1:
                ## suicides?
                player = self.ign_matrix.get(line.ocr_results[0].text)
                target = self.ign_matrix.get(line.ocr_results[0].text)

            elif len(line.ocr_results) == 2:
                player = self.ign_matrix.get(line.ocr_results[0].text)
                target = self.ign_matrix.get(line.ocr_results[1].text)
                
            if player is None or target is None:
                continue ## invalid igns

            record = KFRecord(player, target, self.current_time, line.headshot)
            if record not in self.tempfeed: ## a record requires at least 2 instances to add to kf
                self.tempfeed[record] = 1
            
            elif self.tempfeed[record] < 1: ## TODO: make variable for accuracy control
                self.tempfeed[record] += 1

            elif record not in self.history.cround.killfeed:
                self.history.cround.killfeed.append(record)
                self.history.cround.deaths.append(target.idx)
                self.ign_matrix.update_team_table(player.idx, target.idx)

                self._verbose_print(2, record.to_str())
                self.prog_bar.set_desc(record.to_str())
                self.prog_bar.refresh()
    
    OCResult_t: TypeAlias = tuple[list[list], str, float]
    def __read_feed(self, image_lines: list[np.ndarray]) -> tuple[list[OCRLine], list[np.ndarray]]:
        """
        Reading of the killfeed requires a few steps
        1. Preprocessing with grayscale, fastNlMeansDenoising and width squeezing
        2. EasyOCR reading and cleaning
        3. Line Sorting & Cleaning
        """
        image_lines = [self._screenshot_preprocess(image, denoise=True)#0.75)
                       for image in image_lines]
        
        ocr_results = self.reader.readtext_batched(image_lines,
                            add_margin=0.15,
                            slope_ths=0.5,
                            allowlist=Analyser.KF_ALLOWLIST)
        headshots = [self.__is_headshot(img_line) for img_line in image_lines]
        ocr_lines = self.__get_ocrlines(ocr_results, headshots)
        return ocr_lines, image_lines
    
    def __get_ocrlines(self, lines: list[list[OCResult_t]], headshot_rects: list[list[int]]) -> list[OCRLine]:
        output = []
        for rawline, hsr in zip(lines, headshot_rects):
            ## cleaning stage 1
            results: list[OCResult] = []
            for res in rawline:
                res = OCResult(res)
                if res.prob < InPersonAnalyser.KF_PROB or len(res.text) < 2: continue
                if max([IGNMatrix.compare_names(res.text, non_name) for non_name in InPersonAnalyser.NON_NAMES]) > 0.5: continue
                
                # if hsr and compute_iou(hsr, res.rect) > 0.75:
                #     continue

                res.eval(self.ign_matrix)
                results.append(res)
            
            if len(results) == 0: continue
            elif len(results) == 2 and results[0].rect[0] > results[1].rect[0]:
                results[0], results[1] = results[1], results[0]
            elif len(results) >= 3: ## join call is required
                results = self.__join_line(results)

            output.append(OCRLine(results, headshot=bool(hsr)))

        return output

    def __join_line(self, line: list[OCResult]) -> list[OCResult]:
        """Determines which OCResults to join"""
        prox_dist = InPersonAnalyser.PROXIMITY_DIST * (self.config["SCREENSHOT_RESIZE"] / 4)
        line.sort(key=lambda ocr_res: ocr_res.rect[0])
        new_line = []
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
            new_line = sorted(new_line, key=lambda res: res.eval_score, reverse=True)[:2]
            if new_line[0].rect[0] > new_line[1].rect[0]:
                new_line = [new_line[1], new_line[0]]

        return new_line

    def __is_headshot(self, image_line: np.ndarray, threshold: float = 0.6) -> list[int]:
        hs_img = self.assets["headshot"]
        result = cv2.matchTemplate(image_line, hs_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, best_pt = cv2.minMaxLoc(result)
        
        if max_val < threshold:
            return []

        h,w = hs_img.shape
        return [best_pt[0], best_pt[1], w, h]

    # ----- GAME STATE -----
    def __pre_new_round(self, score2: int, save: bool = True) -> None:
        """
        To save the correct information for each round, this method must be called just before the start of a new round,
          so that the scoreline and winner attributes can be used
        """
        if not self.history.is_ready: return

        ## infer winner of previous round based on new scoreline
        winner = self.history.cround.winner
        if winner is None and self.history.cround.scoreline is not None:
            _, _score2 = self.history.cround.scoreline
            winner = int(_score2 < score2)
            self.history.cround.winner = winner ## if _score1+1 == score1, return 0

        win_con = self._get_wincon()
        self.history.cround.win_condition = win_con

        reat = self.history.cround.round_end_at
        if win_con == WinCondition.DISABLED_DEFUSER:
            self.history.cround.disabled_defuser_at = reat
            self._verbose_print(1, f"Disabled defsuer at: {reat}")
        
        self._verbose_print(0, f"Team {winner} wins round {self.history.roundn} by {win_con.value} at {reat}.")
        if save:
            self.writer.write(self.history, self.ign_matrix)

    def _new_round(self, score1: int, score2: int) -> None:
        """
        When a new round starts, this method is called, initialising a new round history
        The parameters `score1` and `score2` are the current scores displayed at the start of a new round
        """
        new_round = score1 + score2 + 1
        if new_round in self.history: return

        self.__pre_new_round(score2, save=True)
        self.state = State(True, False, False)
        self.timer_format = TimerFormat.FULL
        self.last_seconds_count = 0

        self.history.new_round(new_round)
        self.history.cround.scoreline = [score1, score2]
        self.tempfeed.clear()

        self._verbose_print(1, f"New Round: {new_round} | Scoreline: {score1}-{score2}")
        self.prog_bar.reset()
        self.prog_bar.set_header(new_round, score1, score2)
        self.prog_bar.refresh()
    
    def _end_round(self) -> None:
        """A game state method called when the program determines that the current round has ended"""
        self.history.cround.round_end_at = self.current_time
        self.state = State(False, True, False)
        self.last_seconds_count = 0
        self.prog_bar.set_time(0)
        self.prog_bar.set_desc("ROUND END")
        self.prog_bar.refresh()

        mx_rnd = self.config["MAX_ROUNDS"]
        rps = self.config["ROUNDS_PER_SIDE"]
        scoreline = self.history.cround.scoreline = [0, 0]
        is_ot = not self.config["SCRIM"] and scoreline[0] >= rps and scoreline[1] >= rps
        if self.history.roundn >= mx_rnd \
                or (not is_ot and max(scoreline) == rps+1) \
                or (is_ot and abs(scoreline[0]-scoreline[1]) == 2):
            self._end_game()
    
    def _get_wincon(self) -> WinCondition:
        """Returns the win condition from the round history"""
        bpat    = self.history.cround.bomb_planted_at
        winner  = self.history.cround.winner
        atkside = self.history.cround.atk_side
        if winner is None or atkside is None:
            return WinCondition.UNKNOWN

        defside = 1-atkside
        if bpat is not None and winner == defside:
            return WinCondition.DISABLED_DEFUSER
        elif bpat is not None:
            return WinCondition.DEFUSED_BOMB

        elif 0 <= self.current_time.to_int() <= 1 \
                and winner == defside \
                and self.__wincon_alive_count(atkside) > 0 \
                and self.__wincon_alive_count(defside) > 0:
            return WinCondition.TIME

        elif self.__wincon_alive_count(1-winner) == 0:
            return WinCondition.KILLED_OPPONENTS
        
        return WinCondition.UNKNOWN
    
    def __wincon_alive_count(self, side: int) -> int:
        """Returns the number of alive players on a particular side"""
        alive = 5 ## TODO: history, ign matrix team count
        for d_idx in self.history.cround.deaths:
            if self.ign_matrix.get_team_from_idx(d_idx) == side:
                alive -= 1
        return alive
    
    def _end_game(self) -> None:
        """This method is called when the program determines the game has ended"""
        winner = self._ask_winner()
        self.history.cround.winner = winner
        if self.history.cround.scoreline is not None:
            scoreline = self.history.cround.scoreline[:]
            scoreline[winner] += 1
            self.__pre_new_round(scoreline[1], save=False)

        self.writer.write(self.history, self.ign_matrix)
        self._verbose_print(0, f"Data Saved to {self.prog_args.save}, program terminated.")
        sys.exit()
    
    def _fix_state(self) -> None:
        """
        Called when the program incorrectly thinks the round ended, e.g. paused during death animation with now scoreline showing
        """
        self._verbose_print(1, f"Fixing State")
        self.state.in_round = True
        self.state.end_round = False

        timer_region = self._get_ss_regions(["TIMER_REGION"])["TIMER_REGION"]
        self.state.bomb_planted = self.__is_bomb_countdown(timer_region)
        
        self.history.fix_round()

    ## ----- CHECK & TEST -----
    def check(self) -> None:
        """Called when the --check flag is present in the program call, saves the screenshotted regions as jpg images"""
        if not exists("images"):
            mkdir("images")

        self._verbose_print(0, "Saving check images")
        self.__set_kfline_config()
        region_keys = self._get_ss_region_keys()
        regions = self._get_ss_regions(region_keys)
        for name, img in regions.items():
            cv2.imwrite(join("images", f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
    def test(self) -> None:
        """
        This method is called when the `--test` flag is added to the program call,
        runs inference on a single screenshot.
        """
        regions   = self._get_ss_regions()
        scoreline = self.__read_scoreline(regions["TEAM1_SCORE_REGION"], regions["TEAM2_SCORE_REGION"])
        atkside   = self.__read_atkside(regions["TEAM1_SIDE_REGION"], regions["TEAM2_SIDE_REGION"])
        time_read = self.__read_timer(regions["TIMER_REGION"])
        feed_read, image_lines = self.__read_feed([regions[f"KF_LINE{i+1}_REGION"] for i in range(InPersonAnalyser.NUM_KF_LINES)])

        headshots = [self.__is_headshot(img_line) for img_line in image_lines]
        self.__test_drawocr(image_lines, feed_read, headshots)

        print(f"\nTest: {scoreline=} | {atkside=} | {time_read} | ", end="")
        for line in feed_read:
            if len(line.ocr_results) == 1:
                print(f"{line.ocr_results[0]} -> {line.ocr_results[0]}", end="")
            elif len(line.ocr_results) == 2:
                headshot = "(X) " if line.headshot else ""
                print(f"{line.ocr_results[0]} -> {headshot}{line.ocr_results[1]}, ", end="")
            else:
                headshot = "(X) " if line.headshot else ""
                print(headshot, "|".join([str(res) for res in line.ocr_results]))
        print()

    def __test_drawocr(self, image_lines: list[np.ndarray], lines: list[OCRLine], hs_rects: list[list[int]]) -> None:
        line_cols: list[tuple[int, int, int]] = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i, (image, line, col, hs) in enumerate(zip(image_lines, lines, line_cols, hs_rects)):
            rect_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            for ocres in line.ocr_results:
                cv2.rectangle(rect_img, ocres.rect[:2],  [ocres.rect[0]+ocres.rect[2], ocres.rect[1]+ocres.rect[3]], col, 2)
            if hs:
                cv2.rectangle(rect_img, hs[:2],  [hs[0]+hs[2], hs[1]+hs[3]], (255, 225, 0), 2)
            
            cv2.imwrite(f"images/TEST_KFLINE{i+1}.jpg", rect_img)


class SpectatorAnalyser(Analyser):
    NUM_LAST_SECONDS = 4
    END_ROUND_SECONDS = 12
    
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TIMER_REGION", "KILLFEED_REGION"]

    def __init__(self, args) -> None:
        super(SpectatorAnalyser, self).__init__(args)

    def _get_ss_region_keys(self) -> list[str]:
        return SpectatorAnalyser.SCREENSHOT_REGIONS

    ## ----- IN ROUND OCR FUNCTIONS -----
    def _handle_scoreline(self, team1_scoreline: np.ndarray, team2_scoreline: np.ndarray) -> None:
        ...
    
    def _handle_timer(self, timer: np.ndarray) -> None:
        ...
    
    def _handle_feed(self, feed: np.ndarray) -> None:
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
    def check(self) -> None:
        ...
    
    def test(self) -> None:
        ...


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
