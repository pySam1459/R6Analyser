import argparse
import json
import pyautogui
import cv2
import numpy as np
import easyocr
from PIL import Image
from re import search, fullmatch
from time import sleep, time
from datetime import datetime
from os import mkdir
from os.path import join, exists
from sys import exit as sys_exit
from dataclasses import dataclass
from enum import Enum
from Levenshtein import ratio as leven_ratio


class StrEnum(Enum):
    @classmethod
    def from_string(cls, value: str):
        """
        Class method to convert a string to an Enum value, with validity checks.
        """
        for enum_member in cls:
            if enum_member.value == value:
                return enum_member
        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


class IGNMatrixMode(StrEnum):
    FIXED = "fixed"
    INFER = "infer"
    OPPOSITION = "opposition"


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
        self.__igns: list[str|dict] = [ign for ign in igns if ign is not None]
        self.__fixed: int = len(self.__igns)
        self.__mode = mode
        self.__opp_value = opp_value

        self.__team_table = { self.__igns.index(ign): int(idx >= 5) for idx, ign in enumerate(igns) if ign is not None }

    def from_idx(self, index: int) -> str | None:
        """
        Returns the IGN of a player with the specified index, if the IGN was inferred, the highest recorded occurrence IGN is returned
        """
        if 0 <= index < len(self.__igns):
            el = self.__igns[index]
            if type(el) == str:
                return el
            elif type(el) == dict: ## names_dict
                return IGNMatrix.__max_dict(el)

        return None
    
    def get_index(self, pseudoIGN: str) -> int | None:
        return self.get(pseudoIGN)[0]

    def get(self, pseudoIGN: str, threshold: float = VALID_THRESHOLD) -> tuple[int|None, str|None]:
        """
        This method is used to request the index/ID and true/most-seen IGN from the pseudoIGN argument.
        If the matrix is fully-fixed, the method will return (None, None) if the pseudoIGN is not present.
        """
        ## Check the fixed igns
        if pseudoIGN in self.__igns:
            return self.__igns.index(pseudoIGN), pseudoIGN

        if self.__fixed > 0:
            fixed_igns = self.__igns[:self.__fixed]
            for i, name in enumerate(fixed_igns):
                if IGNMatrix.__compare_names(pseudoIGN, name) > threshold:
                    return i, name
        
        if self.__fixed == 10 or self.__mode == IGNMatrixMode.FIXED:
            return None, None ## not a valid IGN

        if self.__mode == IGNMatrixMode.OPPOSITION:
            return -1, self.__opp_value

        ## Check the unfixed, infered igns
        unfixed_igns = self.__igns[self.__fixed:]
        for i, names_dict in enumerate(unfixed_igns, start=self.__fixed):
            if pseudoIGN in names_dict:
                names_dict[pseudoIGN] += 1 # increase occurrence count for pseudoIGN
                return i, IGNMatrix.__max_dict(names_dict)
            
            scores = [IGNMatrix.__compare_names(pseudoIGN, name) for name in names_dict.keys()]
            if max(scores) > threshold:
                names_dict[pseudoIGN] = 1 # add to names_dict as a possible true IGN
                return i, IGNMatrix.__max_dict(names_dict)
        
        ## if ign has not been seen, add to matrix
        idx = len(self.__igns)
        self.__igns.append({ pseudoIGN: 1 })
        return idx, pseudoIGN

    def get_team(self, ign: str | int) -> int | None:
        """
        Fixed:
            first 5 igns provided are team idx 0, last 5 are team idx 1
        Infer:
            a team index table will be created, and records 
        """
        idx = ign if type(ign) == int else self.get_index(ign)

        if self.__fixed >= 5:
            return int(idx >= 5)
        
        ## Infer from table
        if idx in self.__team_table:
            kill_diff = self.__team_table[idx]
            if kill_diff[0] < kill_diff[1]:
                return 0
            elif kill_diff[1] < kill_diff[0]:
                return 1

        return None ## team idx is unknown
    
    def update_team_table(self, player: str | int, target: str | int) -> None:
        if self.__fixed == 10: return ## team_table update is unneccessary

        pidx = player if type(player) == int else self.get_index(player)
        tidx = target if type(target) == int else self.get_index(target)

        p_killdiff = self.__team_table.get(pidx, None)
        t_killdiff = self.__team_table.get(tidx, None)
        pknown = type(p_killdiff) == int
        tknown = type(t_killdiff) == int
        if pknown and tknown: ## if player's team is already known
            return
        elif pknown or tknown:
            if pidx is not self.__team_table:
                self.__team_table[pidx] = 1 - tknown
            elif tidx is not self.__team_table:
                self.__team_table[tidx] = 1 - pknown

        elif False: ## TODO fix, inference
            if pidx not in self.__team_table:
                self.__team_table[pidx] = [0, 0]
            if tidx not in self.__team_table:
                self.__team_table[tidx] = [0, 0]

            pidx_team = self.get_team(pidx)
            tidx_team = self.get_team(tidx)
            self.__team_table[pidx][tidx_team] += 1  


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
    def __max_dict(_dict: dict[str: int]) -> str:
        return max(_dict, key=_dict.get)
    
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
    
    def to_string(self) -> str:
        return f"{self.time}: {self.player} -> {self.target}"

    def to_json(self, ign_matrix: IGNMatrix = None) -> dict:
        if ign_matrix is None:
            player = self.player
            target = self.target
        else:
            player = ign_matrix.from_idx(self.player_idx) or self.player
            target = ign_matrix.from_idx(self.target_idx) or self.target

        return {
            "time": self.time,
            "player": player,
            "target": target
        }

    __str__ = to_string
    __repr__ = to_string # to_json


@dataclass
class State:
    in_round: bool
    last_ten: bool
    bomb_planted: bool


class WinCondition(StrEnum):
    KILLED_OPPONENTS = "KilledOpponents"
    TIME = "Time"
    DEFUSED_BOMB = "DefusedBomb"
    DISABLED_DEFUSER = "DisabledDefuser"


class Analyser:
    """
    Main class `Analyser`
    Operates the main inference loop `run` and records match/round information
    """
    PROB_THRESHOLD = 0.5
    RED_THRESHOLD = 0.9    ## Note: avg red_perc for bomb-defuse countdown ~0.93

    NUM_LAST_SECONDS = 4   ## number of seconds to continue reading killfeed after round end (reliability reasons)
    END_ROUND_SECONDS = 12 ## number of seconds to check no timer to determine round end
    
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TIMER_REGION", "KILL_FEED_REGION"]

    RED_HSV_SPACE = np.array([ ## Defines the range for red color in HSV space
        [0, 120, 70],
        [10, 255, 255],
        [170, 170, 70],
        [180, 255, 255]])

    SHARPEN_KERNEL = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]])
    SHARPEN_KERNEL2 = np.array([
        [0, -1, 0],
        [-1,  5, -1],
        [0, -1, 0]])
    
    def __init__(self, args: argparse.Namespace):
        self.config: dict = args.config
        self.verbose: int = args.verbose
        self.prog_args = args
        self.__debug_print(f"Config Keys -", list(self.config.keys()))
        
        if args.check:
            self.check()

        self.running = False
        self.tdelta: float = self.config.get("SCREENSHOT_PERIOD", 1.0)
        
        self.ign_matrix = IGNMatrix.new(self.config["IGNS"], self.config["IGN_MODE"])
        self.reader = easyocr.Reader(['en'], gpu=not args.cpu)
        self.__verbose_print(0, "EasyOCR Reader model loaded")
        
        self.state = State(False, False, False)
        self.history = {}
        self.__temp_history = self.__new_history_round()
        self.current_round = None
        self.current_time = None

        self.check_scoreline = True        
        self.defuse_countdown_timer = None
        self.last_kf_seconds = None
        self.end_round_seconds = None


    def check(self) -> None:
        regions = self.__get_ss_regions(Analyser.SCREENSHOT_REGIONS)
        self.__save_regions(regions)
        sys_exit()
    
    def __save_regions(self, regions: list[np.ndarray]) -> None:
        if not exists("images"):
            mkdir("images")

        print("Info: Saving check images")
        for region_name, region_image in zip(Analyser.SCREENSHOT_REGIONS, regions):
            if region_name == "KILL_FEED_REGION":
                region_image = self.__killfeed_preprocess(region_image)
            Image.fromarray(region_image).save(join("images", f"{region_name}.png"))


    def run(self):
        self.running = True
        self.__verbose_print(0, "Running...")

        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time():
                continue
            
            __inference_start = time()
            regions = self.__get_ss_regions(Analyser.SCREENSHOT_REGIONS)
            
            team1_scoreline, team2_scoreline, timer, feed = regions
            self.__handle_scoreline(team1_scoreline, team2_scoreline)
            self.__handle_timer(timer)
            self.__read_feed(feed)

            # self.__debug_print(f"Inference time {time()-__inference_start:.2f}s")
            
            self.timer = time()

    def __get_ss_regions(self, regions: list[str]) -> list[np.ndarray]:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        screenshot = pyautogui.screenshot(allScreens=True)
        return [np.array(screenshot.crop(Analyser.convert_region(self.config[region])), copy=False) for region in regions]
    
    @staticmethod
    def convert_region(region: list[int]) -> list[int]:
        """Converts (X,Y,W,H) -> (Left,Top,Right,Bottom)"""
        left, top, width, height = region
        return (left, top, left + width, top + height)

    
    def __history_set(self, key: str, value) -> None:
        """Sets the values of elements in the history attributes; appending values to the `killfeed` element"""
        APPEND_KEYS = ["killfeed"]
        if self.current_round not in self.history:
            self.check_scoreline = True
            history = self.__temp_history
        else:
            history = self.history[self.current_round]
        
        if key in APPEND_KEYS:
            history[key].append(value)
        else:
            history[key] = value
        
        self.__verbose_print(1, history)
    
    ## ----- OCR -----
    def __killfeed_preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.fastNlMeansDenoising(image, None, 5, 7, 21)
    
        # image = cv2.filter2D(image, -1, Analyser.SHARPEN_KERNEL2)
        sf = self.config.get("SCREENSHOT_RESIZE_FACTOR", 2)
        new_width = int(image.shape[1] * sf)
        new_height = int(image.shape[0] * sf)
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def __screenshot_preprocess(self,
                                image: np.ndarray,
                                to_gray: bool = True,
                                sharpen: bool = False,
                                squeeze_width: float = -1) -> np.ndarray:
        """
        To increase the accuracy of the EasyOCR readtext function, a few preprocessing techniques are used
          - RGB to Grayscale conversion
          - Resize by factor `Config.SCREENSHOT_RESIZE_FACTOR` (normally 2-4)
          - squeeze the width of the image, useful for scoreline OCR
        """
        scale_factor = self.config.get("SCREENSHOT_RESIZE_FACTOR", 2)
        if squeeze_width != -1:
            sf_w, sf_h = scale_factor * squeeze_width, scale_factor
        else:
            sf_w = sf_h = scale_factor

        new_width = int(image.shape[1] * sf_w)
        new_height = int(image.shape[0] * sf_h)
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        if sharpen:
            image = cv2.filter2D(image, -1, Analyser.SHARPEN_KERNEL)
        
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def __readtext(self, image: np.ndarray, prob: float=PROB_THRESHOLD, out: bool=False) -> list[str]:
        """Performs the EasyOCR inference and cleans the output based on the model's assigned probabilities and a threshold"""
        results = self.reader.readtext(image)
        if out: print(results)
        return [out[1] for out in results if out[2] > prob]


    ## ----- SCORELINE -----
    def __handle_scoreline(self, team1_scoreline: np.ndarray, team2_scoreline: np.ndarray) -> None:
        """Extracts the current scoreline visible and determines when a new rounds starts"""
        if not self.check_scoreline: return

        img1 = self.__screenshot_preprocess(team1_scoreline, squeeze_width=0.5)
        img2 = self.__screenshot_preprocess(team2_scoreline, squeeze_width=0.5)

        results = self.__readtext(img1) + self.__readtext(img2)
        if len(results) != 2:
            return
        if not fullmatch(r"^\d+$", results[0]) or not fullmatch(r"^\d+$", results[1]):
            return
        
        ## create update for new round start
        score1, score2 = map(int, results)
        if score1 + score2 + 1 == self.current_round and self.current_round in self.history:
            self.check_scoreline = False
            return
        
        self.__new_round(score1, score2)
        self.check_scoreline = False


    ## ----- TIMER FUNCTION -----
    def __handle_timer(self, timer_image: np.ndarray) -> None:
        new_time = self.__read_timer(timer_image)
        
        if new_time is not None: ## timer is showing
            self.current_time = new_time
            self.defuse_countdown_timer = None

            self.last_kf_seconds = None
            self.end_round_seconds = None

            if not self.state.in_round:
                self.check_scoreline = True
                self.state.in_round = True ## might get confused with pick phase timer?
        
        elif self.__is_bomb_countdown(timer_image):
            if self.defuse_countdown_timer is None: ## bomb planted
                self.defuse_countdown_timer = time()
                self.__history_set("bomb_planted_at", self.current_time)

                self.state.bomb_planted = True
                self.__verbose_print(0, f"{self.current_time}: BOMB PLANTED")

        elif self.last_kf_seconds is None and self.end_round_seconds is None:
            self.last_kf_seconds = time()
            self.end_round_seconds = time()
        
        if self.last_kf_seconds is not None and self.last_kf_seconds + Analyser.NUM_LAST_SECONDS < time():
            self.last_kf_seconds = None

        if self.end_round_seconds is not None and self.end_round_seconds + Analyser.END_ROUND_SECONDS < time():
            self.__end_round()
            self.end_round_seconds = None

    
    def __read_timer(self, image: np.ndarray) -> str | None:
        """
        Reads the current time displayed in the region `TIMER_REGION`
        If the timer is not present, None is returned
        """
        image = self.__screenshot_preprocess(image)
        results = self.__readtext(image, prob=0.4)
        for read_time in results:
            if (time := search(r"([0-2]).{1,2}([0-5],?\d)", read_time)) and not self.state.last_ten: ## 2:59-0:10
                seconds = time.group(2).replace(",", "")
                return f"{time.group(1)}:{seconds}"
            elif (time := search(r"(\d).{1,2}(\d,?\d)", read_time)): ## 9:99-0:00
                return f"0:0{time.group(1)}"
        
        return None
    
    def __is_bomb_countdown(self, image: np.ndarray) -> bool:
        """
        When a bomb is planted, the timer is replaced with a majority red circular countdown
        This method detects when the bomb defuse countdown is shown using a majority red threshold
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create masks for red color
        mask1 = cv2.inRange(hsv, Analyser.RED_HSV_SPACE[0], Analyser.RED_HSV_SPACE[1])
        mask2 = cv2.inRange(hsv, Analyser.RED_HSV_SPACE[2], Analyser.RED_HSV_SPACE[3])
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate the percentage of red in the image
        red_percentage = np.sum(red_mask > 0) / red_mask.size
        self.__debug_print(f"{red_percentage=}")
        return 0.97 > red_percentage > Analyser.RED_THRESHOLD


    ## ----- KILL FEED -----
    def __get_time(self) -> str:
        if self.defuse_countdown_timer is None:
            return self.current_time
        else:
            time_past = time() - self.defuse_countdown_timer
            return f"!0:{int(45-time_past)}"
    
    def __get_killfeed(self) -> list[dict]:
        if self.current_round not in self.history:
            return self.__temp_history.get("killfeed", [])
        else:
            return self.history[self.current_round].get("killfeed", [])

    def __read_feed(self, image: np.ndarray) -> None:
        if not self.state.in_round and self.last_kf_seconds is None: return

        image = self.__killfeed_preprocess(image)
        results = self.__readtext(image, prob=0.3)
        if len(results) % 2 != 0: return None  ## TODO: could cause an issue when someone c4's themselves
        
        for i in range(0, len(results), 2):
            p_idx, p_name = self.ign_matrix.get(results[i], 0.65)
            t_idx, t_name = self.ign_matrix.get(results[i+1], 0.65)
            
            if p_idx is None or t_idx is None: continue ## invalid igns
            if self.ign_matrix.get_mode() == IGNMatrixMode.OPPOSITION and (p_idx == -1 and t_idx == -1):
                continue ## disregard opp on opp / invalid igns
            
            record = KFRecord(p_name, p_idx, t_name, t_idx, self.__get_time())
            if record not in self.__get_killfeed():
                self.__history_set("killfeed", record)
                self.ign_matrix.update_team_table(p_idx, t_idx)

                # did_print = self.__verbose_print(1, f"{record.time}: {record.player}/{record.player_idx} -> {record.target}/{record.target_idx}")
                # if not did_print: self.__verbose_print(0, f"{record.time}: {record.player} -> {record.target}")


    ## ----- GAME STATE FUNCTIONS -----
    
    def __new_history_round(self) -> None:
        return {
            "scoreline": None,
            "bomb_planted_at": None,
            "bomb_defused_at": None,
            "round_end_at": None,
            "win_condition": None,
            "winner": None,
            "killfeed": []
        }

    def __new_round(self, score1: int, score2: int) -> None:
        """
        When a new round starts, this method is called, initialising a new round history
        The parameters `score1` and `score2` are the current scores displayed at the start of a new round
        """
        ## infer winner of previous round based on new scoreline
        if self.current_round is not None and self.history[self.current_round]["winner"] is None:
            _, _score2 = self.history[self.current_round]
            self.history[self.current_round]["winner"] = int(_score2 < score2) ## if _score1+1 == score1, return 0

        self.state.last_ten = False

        new_round = score1 + score2 + 1
        self.current_round = new_round
        self.history[new_round] = self.__new_history_round()
        self.history[new_round]["scoreline"] = [score1, score2]

        ## temp history is an attempt to be fault tolerant, if certain OCR processes (scoreline) are inaccurate
        if len(self.__temp_history) > 0:
            for key, value in self.__temp_history.items():
                self.history[new_round][key] = value

            self.__temp_history = self.__new_history_round()
        
        self.__verbose_print(1, self.history[new_round])
    
    def __end_round(self) -> None:
        if len(self.history) == 0: return ## TODO: consider if this is a appropriate clause guard

        # if self.state.bomb_planted:
        #     win_con = WinCondition.DISABLED_DEFUSER if self.state.disabled_defuser else WinCondition.DEFUSED_BOMB
        # elif self.current_time == "0:00": ## this could be doubious
        #     win_con = WinCondition.TIME
        #     ## self.__history_set("winner", )
        # else: ## killed opps
        #     ...

        # self.__history_set("win_condition", win_con)
        self.__history_set("round_end_at", self.current_time)

        self.state.in_round = False
        self.state.bomb_planted = False

        if len(self.history) >= self.config["MAX_ROUNDS"] or sum(self.history[self.current_round]["scoreline"])+1 >= self.config["MAX_ROUNDS"]:
            self.__end_game()
        elif self.prog_args.append_save:
            self.__save()
        elif self.prog_args.upload_save:
            self.__upload_save()

    def __end_game(self) -> None:
        self.__save()
        sys_exit()
    
    def __save(self) -> None:
        if not exists("saves"):
            mkdir("saves")

        match self.prog_args.save.ext:
            case "json":
                self.__save_json()
            case "xlsx":
                self.__save_xlsx()

    def __upload_save(self) -> None:
        ...
    
    def __save_json(self) -> None:
        hist_clone = repr(self.history.copy())
        with open(join("saves", self.prog_args.save), "w") as f_out:
            json.dump(hist_clone, f_out, indent=4)
    
    def __save_xlsx(self) -> None:
        ...


    ## ----- PRINT FUNCTION -----
    def __verbose_print(self, verbose_value: int, *prompt) -> bool:
        if self.verbose > verbose_value:
            print("Info:", *prompt)

        return self.verbose > verbose_value

    def __debug_print(self, *prompt: str) -> None:
        if self.verbose == 3:
            print("Debug:", *prompt)


## ----- HELPER FUNCTIONS -----
def __infer_key(config: dict, key: str) -> None:
    match key:
        case "TEAM1_SCORE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM1_SCORE_REGION"] = [tr[0] - tr[2]//2, tr[1], tr[2]//2, tr[3]]
        case "TEAM2_SCORE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM2_SCORE_REGION"] = [tr[0] + tr[2], tr[1], tr[2]//2, tr[3]]
        case "MAX_ROUNDS":
            config["MAX_ROUNDS"] = 12 if config["SCRIM"] else 15
        case _:
            ...


def __parse_config(arg: str) -> dict:
    ## argument checks
    if not (exists(arg) or exists(join("configs", arg))):
        raise argparse.ArgumentError(f"Config file '{arg}' cannot be found!")
    
    if not exists(arg):
        arg = join("configs", arg)
    
    ## config keys
    REQUIRED_CONFIG_KEYS = ["SCRIM", "TIMER_REGION", "KILL_FEED_REGION"]
    OPTIONAL_CONFIG_KEYS = ["SCREENSHOT_RESIZE_FACTOR", "SCREENSHOT_PERIOD", "IGNS", "IGN_MODE", "SPECTATOR"]
    INFER_CONFIG_KEYS    = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "MAX_ROUNDS"]
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


@dataclass
class SaveFile:
    filename: str
    ext: str

    def __str__(self) -> str: return f"{self.filename}.{self.ext}"
    __repr__ = __str__


def __parse_save(arg: str) -> SaveFile:
    filename, ext = arg.rsplit(".", maxsplit=1)
    if ext == "json" or ext == "xlsx":
        return SaveFile(filename, ext)
    raise argparse.ArgumentError(f"Invalid save file type {ext}, only json/xlsx allowed")

def __default_save() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".json"


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
    parser.add_argument("-s", "--save",
                        type=__parse_save,
                        help="Where to save the output of R6Analyser, only json or xlsx files",
                        dest="save",
                        default=__default_save())
    parser.add_argument("--append-save",
                       action="store_true",
                       help="Whether to append new round data onto the existing save file, otherwise save all data at the end of a game",
                       dest="append_save")
    parser.add_argument("--upload-save",
                       action="store_true",
                       help="Whether to upload new round data directly to the cloud, otherwise save all data at the end of a game",
                       dest="upload_save")
    parser.add_argument("--check", 
                        action="store_true",
                        help="Does not perform data extract but saves the regions of interest as images for quality check")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Flag for only cpu execution, if your machine does not support gpu acceleration")

    args = parser.parse_args()
    if args.delay > 0:
        sleep(args.delay)
    
    Analyser(args).run()
