import argparse
import json
import pyautogui
import cv2
import numpy as np
import easyocr
from re import search
from time import sleep, time
from os.path import join, exists
from dataclasses import dataclass
from enum import Enum
from Levenshtein import ratio as leven_ratio


class IGNMatrixMode(Enum):
    FIXED = "fixed"
    INFER = "infer"
    OPPOSITION = "opposition"

    @classmethod
    def from_string(cls, value: str):
        """
        Class method to convert a string to an Enum value, with validity checks.
        """
        for enum_member in cls:
            if enum_member.value == value:
                return enum_member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


class IGNMatrix:
    """
    IGN Matrix infers the true IGN from the EasyOCR reading of the IGN (pseudoIGN) from the killfeed.
    The matrix can be initialised with a prior list of 'fixed' IGNs (IGNs known before starting the game)
    If the matrix is not provided with 10 fixed IGNs, the matrix's output will depend on its initialised mode
    - fixed:      will return None for all non-fixed IGNs
    - infer:      will infer the non-fixed IGNs from the OCR's output
    - opposition: will return init param opp_value  [default='OPPOSITION'] for non-fixed IGNs, used when you only care about a single team's statistics
    The matrix will return an index/ID and the true/most-seen IGN when requested using the `get` method
    Notes:
    - Fixing the IGNs prior to recording is recommended and more accurate
        (None, None) will be returned if the IGN requested is not present in a fully-fixed matrix (all 10 IGNs fixed)
    - The matrix uses Levenshtein distance to determine whether two IGNs are the same (could be improved?)
    - The matrix records the number of occurrences of a pseudoIGN to determine its true IGN
    - When requested and not fixed, the matrix will compare a pseudoIGN against all occurrences of pseudoIGNs
        if the requested pseudoIGN matches with a known-pseudoIGN, it will return this pseudoIGN's index
        otherwise, the pseudoIGN will be added to the matrix
    """
    VALID_THRESHOLD = 0.75 ## threshold for Levenshtein distance to determine equality
    
    def __init__(self, igns: list[str], mode: IGNMatrixMode, opp_value: str = "OPPOSITION") -> None:
        self.__igns: list[str|dict] = igns
        self.__fixed: int = len(igns)
        self.__mode = mode
        self.__opp_value = opp_value
        
    def get(self, pseudoIGN: str) -> tuple[int|None, str|None]:
        """
        This method is used to request the index/ID and true/most-seen IGN from the pseudoIGN argument.
        If the matrix is fully-fixed, the method will return (None, None) if the pseudoIGN is not present.
        """
        ## Check the fixed igns
        if self.__fixed > 0:
            fixed_igns = self.__igns[:self.__fixed]
            for i, name in enumerate(fixed_igns):
                if IGNMatrix.__compare_names(pseudoIGN, name) > IGNMatrix.VALID_THRESHOLD:
                    return i, name
        
        if self.__fixed == 10 or self.__mode == IGNMatrixMode.FIXED:
            return None, None ## not a valid IGN

        if self.__mode == IGNMatrixMode.OPPOSITION:
            return -1, self.__opp_value

        ## Check the unfixed, infered igns
        unfixed_igns = self.__igns[self.__fixed:]
        for i, names_dict in enumerate(unfixed_igns, start=self.__fixed):
            if pseudoIGN in names_dict:
                names_dict[pseudoIGN] += 1
                return i, max(names_dict, key=lambda k: names_dict[k])
            
            scores = [IGNMatrix.__compare_names(pseudoIGN, name) for name in names_dict.keys()]
            if max(scores) > IGNMatrix.VALID_THRESHOLD:
                names_dict[pseudoIGN] = 1
                return i, max(names_dict, key=lambda k: names_dict[k])
        
        ## if ign has not been seen, add to matrix
        idx = len(self.__igns)
        self.__igns.append({pseudoIGN: 1})
        return idx, pseudoIGN

    def get_mode(self) -> IGNMatrixMode:
        return self.__mode

    @staticmethod
    def new(igns: list[str], mode: str) -> 'IGNMatrix':
        """Creates a new IGNMatrix object from a list of fixed IGNs"""
        if type(igns) != list:
            raise ValueError(f"Invalid Config IGN list, argument is not a list")

        if len(igns) == 0:
            return IGNMatrix([], IGNMatrixMode.INFER)

        for i, el in enumerate(igns):
            if type(el) != str:
                raise ValueError(f"Invalid Config IGN list, element {i} is not a string")
        
        if len(igns) > 10:
            igns = igns[:10]
            print("Warning: Config IGN list has more than 10 IGNs, will only use first 10")

        _mode = IGNMatrixMode.from_string(mode)
        return IGNMatrix(igns, _mode)
            
    
    @staticmethod
    def __compare_names(name1: str, name2: str) -> float:
        """Compares two IGN's (pseudo/non-pseudo) using Levenshtein distance, output in the range [0-1]"""
        return leven_ratio(name1.lower(), name2.lower())


@dataclass
class KFRecord:
    """
    Dataclass to record an player interaction, who killed who and at what time.
      player: killer, target: dead
    """
    player: str
    player_idx: int
    target: str
    target_idx: str
    time: str
    
    def __eq__(self, other: 'KFRecord') -> bool:
        return self.player_idx == other.player_idx and self.target_idx == other.target_idx


class RHRecord:
    """
    Class to record a piece of information about a round.
    """
    def __init__(self,
                 bomb_planted_time=None,
                 round_scoreline=None,
                 **kwargs):
        attributes = [bomb_planted_time, round_scoreline, *kwargs.values()]
        n_attrs = sum(attr is not None for attr in attributes)
        if n_attrs != 1:
            raise ValueError("RHRecord must only have one property set at a time.")

        self.bomb_planted_time = bomb_planted_time
        self.round_scoreline = round_scoreline
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        active_property = next(((k, v) for k, v in self.__dict__.items() if v is not None), ('none', None))
        return f"RHRecord({active_property[0]}={active_property[1]!r})"


@dataclass
class State:
    in_round: bool
    bomb_planted: bool


class Analyser:
    """
    Main class `Analyser`
    Operates the main inference loop `run` and records match/round information
    """
    PROB_THRESHOLD = 0.5
    RED_THRESHOLD = 0.9   ## Note: avg red_perc for bomb-defuse countdown ~0.93
    
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TIMER_REGION", "KILL_FEED_REGION"]
    
    def __init__(self, args: argparse.Namespace):
        self.config: dict = args.config
        self.gpu: bool    = not args.cpu
        self.verbose: int = args.verbose
        self.__debug_print(f"Config Keys -", list(self.config.keys()))

        self.ign_matrix = IGNMatrix.new(self.config["IGNS"], self.config["IGN_MODE"])
        self.kill_feed: list[KFRecord] = []
        self.round_history: list[RHRecord] = []

        self.running = False
        self.tdelta: float = self.config.get("SCREENSHOT_PERIOD", 1.0)
        
        self.reader = easyocr.Reader(['en'], gpu=self.gpu)
        if self.verbose > 0:
            print("Info: EasyOCR Reader model loaded")
        
        self.state = State(False, False)
        self.current_scoreline = None
        self.current_time = 0
        self.no_timer = 0
        self.defuse_countdown_timer = None
                
        self.__red_hsv_space = np.array([  # Define the range for red color in HSV space
            [0, 70, 50],
            [10, 255, 255],
            [170, 70, 50],
            [180, 255, 255]])
        
        self.__saved = True

    def run(self):
        self.running = True
        if self.verbose > 0:
            print("Info: Running...")

        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time():
                continue
            
            __inference_start = time()
            team1_scoreline, team2_scoreline, timer, feed = self.__get_ss_regions(Analyser.SCREENSHOT_REGIONS)
            
            self.__handle_timer(timer)

            self.__read_scoreline(team1_scoreline, team2_scoreline)

            self.__read_feed(feed)

            self.__debug_print(f"Inference time {time()-__inference_start:.2f}s")
            
            self.timer = time()

    def __get_ss_regions(self, regions: list[str]) -> list[np.ndarray]:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        screenshot = pyautogui.screenshot()
        return [np.array(screenshot.crop(Analyser.convert_region(self.config[region])), copy=False) for region in regions]
    
    @staticmethod
    def convert_region(region: list[int]) -> list[int]:
        """Converts (X,Y,W,H) -> (Left,Top,Right,Bottom)"""
        left, top, width, height = region
        return (left, top, left + width, top + height)

    def __screenshot_preprocess(self, image: np.ndarray, to_gray: bool=True) -> np.ndarray:
        """
        To increase the accuracy of the EasyOCR readtext function, a few preprocessing techniques are used
          - RGB to Grayscale conversion
          - Resize by factor `Config.SCREENSHOT_RESIZE_FACTOR` (normally 2-4)
        """
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        scale_factor = self.config.get("SCREENSHOT_RESIZE_FACTOR", 2)
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def __readtext(self, image: np.ndarray) -> list[str]:
        """Performs the EasyOCR inference and cleans the output based on the model's assigned probabilities and a threshold"""
        results = self.reader.readtext(image)
        return [out[1] for out in results if out[2] > Analyser.PROB_THRESHOLD]


    def __read_scoreline(self, team1_scoreline: np.ndarray, team2_scoreline: np.ndarray):
        scores = np.hstack([self.__screenshot_preprocess(team1_scoreline, to_gray=False), 
                            self.__screenshot_preprocess(team2_scoreline, to_gray=False)])
        if not self.__saved:
            gray1 = self.__screenshot_preprocess(team1_scoreline)
            gray2 = self.__screenshot_preprocess(team2_scoreline)
            
            cv2.imwrite('test/gray1.png', gray1.astype(np.uint8))
            cv2.imwrite('test/gray2.png', gray2.astype(np.uint8))

            self.__saved = True

        self.__debug_print(self.__readtext(scores))
        # results1 = self.__readtext(self.__screenshot_preprocess(team1_scoreline))
        # results2 = self.__readtext(self.__screenshot_preprocess(team2_scoreline))
        # self.__debug_print(results1, results2)
        
    
    def __handle_timer(self, timer_image: np.ndarray) -> None:
        new_time = self.__read_timer(timer_image)
        
        if new_time is not None: ## timer is showing
            self.current_time = new_time
            self.defuse_countdown_timer = None
            self.state.in_round = True ## might get confused with pick phase timer?
        
        elif self.__is_bomb_countdown(timer_image):
            if self.defuse_countdown_timer is None: ## bomb planted
                self.defuse_countdown_timer = time()
                self.round_history.append(RHRecord(bomb_planted_time=self.current_time))

                self.state.bomb_planted = True
                if self.verbose > 0:
                    print(f"{self.current_time}: BOMB PLANTED")
        else:
            self.state.in_round = False
            self.state.bomb_planted = False

    
    def __read_timer(self, image: np.ndarray) -> str | None:
        """
        Reads the current time displayed in the region `TIMER_REGION`
        If the timer is not present, None is returned
        """
        image = self.__screenshot_preprocess(image)
        results = self.__readtext(image)
        for read_time in results:
            if (time := search(r"(\d?\d)[ \.:](\d\d)", read_time)):
                return f"{time.group(1)}:{time.group(2)}"
        
        return None
    
    def __is_bomb_countdown(self, image: np.ndarray) -> bool:
        """
        When a bomb is planted, the timer is replaced with a majority red circular countdown
        This method detects when the bomb defuse countdown is shown using a majority red threshold
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create masks for red color
        mask1 = cv2.inRange(hsv, self.__red_hsv_space[0], self.__red_hsv_space[1])
        mask2 = cv2.inRange(hsv, self.__red_hsv_space[2], self.__red_hsv_space[3])
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate the percentage of red in the image
        red_percentage = np.sum(red_mask > 0) / red_mask.size
        self.__debug_print(f"{red_percentage=}")
        return red_percentage > Analyser.RED_THRESHOLD

    
    def __get_time(self) -> str:
        if self.defuse_countdown_timer is None:
            return self.current_time
        else:
            time_past = time() - self.defuse_countdown_timer
            return f"!0:{int(45-time_past)}"

    def __read_feed(self, image: np.ndarray) -> None:
        if not self.state.in_round: return

        image = self.__screenshot_preprocess(image)
        results = self.__readtext(image)
        if len(results) % 2 != 0: return None  ## TODO: could cause an issue when someone c4's themselves
        
        for i in range(0, len(results), 2):
            player_ign_raw = results[i]
            target_ign_raw = results[i+1]
            p_idx, p_name = self.ign_matrix.get(player_ign_raw)
            t_idx, t_name = self.ign_matrix.get(target_ign_raw)
            
            if p_idx is None or t_idx is None: continue
            if self.ign_matrix.get_mode == IGNMatrixMode.OPPOSITION and (p_idx == -1 and t_idx == -1):
                continue ## disregard opp on opp / invalid igns
            
            record = KFRecord(p_name, p_idx, t_name, t_idx, self.__get_time())
            if record not in self.kill_feed:
                self.kill_feed.append(record)

                if self.verbose >= 2:
                    print(f"{record.time}: {record.player}/{record.player_idx} -> {record.target}/{record.target_idx}")
                elif self.verbose > 0:
                    print(f"{record.time}: {record.player} -> {record.target}")


    def __debug_print(self, *prompt: str) -> None:
        if self.verbose == 3:
            print("Debug:", *prompt)


def __infer_key(config: dict, key: str) -> None:
    match key:
        case "TEAM1_SCORE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM1_SCORE_REGION"] = [tr[0] - tr[2]//2, tr[1], tr[2]//2, tr[3]]
        case "TEAM2_SCORE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM2_SCORE_REGION"] = [tr[0] + tr[2], tr[1], tr[2]//2, tr[3]]
        case _:
            ...


def __parse_config(arg: str) -> dict:
    ## argument checks
    if not (exists(arg) or exists(join("configs", arg))):
        raise argparse.ArgumentError(f"Config file '{arg}' cannot be found!")
    
    if not exists(arg):
        arg = join("configs", arg)
    
    ## config keys
    REQUIRED_CONFIG_KEYS = ["TIMER_REGION", "KILL_FEED_REGION"]
    OPTIONAL_CONFIG_KEYS = ["SCREENSHOT_RESIZE_FACTOR", "SCREENSHOT_PERIOD", "IGNS", "IGN_MODE"]
    INFER_CONFIG_KEYS    = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION"]
    DEFAULT_CONFIG_FILENAME = "defaults.json"
    with open(arg, "r", encoding="utf-8") as f_in:
        config = json.load(f_in)

    ## check if all required keys are contained in config file
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise argparse.ArgumentError(f"Config file does not contain key '{key}'!")
    
    print(f"Info: Loaded configuration file '{arg}'")
    
    ## if the optional keys weren't provided, use defaults from default.json
    to_add = [key for key in OPTIONAL_CONFIG_KEYS if key not in config]
    if len(to_add) > 0:
        if not exists(DEFAULT_CONFIG_FILENAME):
            raise argparse.ArgumentError("'defaults.json' does not exist!")

        with open(DEFAULT_CONFIG_FILENAME, "r", encoding="utf-8") as f_in:
            default_config = json.load(f_in)
        
        __log = ""
        for key in to_add:
            if key not in default_config:
                raise Exception("defaults.json has been modified, key '{key}' has been removed!")
            
            config[key] = default_config[key]
            __log += f"{key}, "
        print(f"Info: Loaded default config keys {__log}")
    
    to_add = [key for key in INFER_CONFIG_KEYS if key not in config]
    if len(to_add) > 0:
        for key in INFER_CONFIG_KEYS:
            __infer_key(config, key)

    return config


def __parse_verbose(arg: str) -> int:
    try:
        x = int(arg)
        if 0 <= x <= 3:
            return x
        raise argparse.ArgumentError("Verbose argument out of range [0,2]")

    except ValueError:
        raise argparse.ArgumentError(f"Invalid Verbose argument {arg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="R6 Analyser",
        description="A Rainbow Six Siege VOD Analyser to record live information from a game.")

    parser.add_argument("config",
                        type=__parse_config,
                        help="Filename of the .json config file containing information bounding boxes")
    parser.add_argument("-d", "--delay",
                        type=int,
                        help="Time delay between starting the program and recording",
                        dest="delay",
                        default=2)
    parser.add_argument("-v", "--verbose",
                        type=__parse_verbose,
                        help="Determines how detailed the console output is, 0-nothing, 1-some, 2-all, 3-debug",
                        dest="verbose",
                        default=1)
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Flag for only cpu execution, if your machine does not support gpu acceleration")

    args = parser.parse_args()
    if args.delay > 0:
        sleep(args.delay)
    
    Analyser(args).run()
