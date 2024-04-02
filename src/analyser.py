import json
import pyautogui
import cv2
import numpy as np
import easyocr
from argparse import Namespace
from re import search, fullmatch
from time import time
from os import mkdir
from os.path import join, exists
from sys import exit as sys_exit
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
from Levenshtein import ratio as leven_ratio
from openpyxl import load_workbook, Workbook
from typing import TypeAlias


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


@dataclass
class Player:
    idx: int
    ign: str


class IGNMatrix(ABC):
    """
    IGN Matrix infers the true IGN from the EasyOCR reading of the IGN (pseudoIGN/pign) from the killfeed.
    The matrix can be initialised with a prior list of 'fixed' IGNs (IGNs known before starting the game)
    If the matrix is not provided with 10 fixed IGNs, the matrix's output will depend on its initialised mode
    - fixed:      will return None for all non-fixed IGNs
    - infer:      will infer the non-fixed IGNs from the OCR's output
    The matrix will return an index/ID and the true/most-seen IGN when requested using the `get` method
    The matrix uses Levenshtein distance to determine whether two IGNs are the same (could be improved?)
    """

    @abstractmethod
    def get(self, pseudoIGN: str, threshold: float) -> Player | None:
        ...
    
    @abstractmethod
    def from_idx(self, idx: int) -> Player | None:
        ...
    
    @abstractmethod
    def get_team(self, ign: str | int) -> int:
        ...
    
    def update_team_table(self, player: str | int, target: str | int) -> None:
        ...
    
    @abstractmethod
    def get_players(self, flip: bool = False) -> list[Player]:
        ...
    
    @staticmethod
    def new(igns: list[str], mode: IGNMatrixMode) -> 'IGNMatrix':
        """Creates a new IGNMatrix object from a list of fixed IGNs"""
        match mode:
            case IGNMatrixMode.FIXED:
                return IGNMatrixFixed.new(igns)
            case IGNMatrixMode.INFER:
                if len(igns) == 10: return IGNMatrixFixed.new(igns)
                else: return IGNMatrixInfer.new(igns)
            case _:
                raise ValueError(f"Unknown IGNMatrixMode {mode}")

    @staticmethod
    def _compare_names(name1: str, name2: str) -> float:
        """Compares two IGN's (pseudo/non-pseudo) using Levenshtein distance, output in the range [0-1]"""
        return leven_ratio(name1.lower(), name2.lower())
    
    @staticmethod
    def _check_fixed(ign_list: list[str], pign: str, threshold: float) -> Player | None:
        for i, ign in enumerate(ign_list):
            if ign == pign or IGNMatrix._compare_names(pign, ign) >= threshold:
                return Player(i, ign)
        return None


class IGNMatrixFixed(IGNMatrix):
    """
    This subclass of IGNMatrix only handles the case where all IGNs are known beforehand
    Separating the different IGN modes aims to improve efficiency, readability and modularity
    """
    VALID_THRESHOLD = 0.6 ## threshold for Levenshtein distance to determine equality

    def __init__(self, igns: list[str]) -> None:
        self.__matrix = igns
    
    def get(self, pign: str, threshold: float = VALID_THRESHOLD) -> Player | None:
        """This method is used to request the index/ID and true/most-seen IGN from the pseudoIGN argument."""
        return IGNMatrix._check_fixed(self.__matrix, pign, threshold)

    def from_idx(self, idx: int) -> Player | None:
        if 0 <= idx <= 10:
            return Player(idx, self.__matrix[idx])

        return None
    
    def get_team(self, ign: str | int) -> int | None:
        if type(ign) == str and ((pl := self.get(ign)) is not None):
            return int(pl.idx >= 5)
        elif type(ign) == int and 0 <= ign < 10:
            return int(ign >= 5)

        return None

    def get_players(self, flip: bool = False) -> list[Player]:
        players = [Player(i, ign) for i, ign in enumerate(self.__matrix)]
        return players  if not flip  else  players[5:] + players[:5] 

    @staticmethod
    def new(igns: list[str]) -> 'IGNMatrixFixed':
        """Type checking of `igns` is done by the `__cparse_IGNS` function, except for length=10 check"""
        if len(igns) != 10:
            raise ValueError(f"Invalid Fixed IGNMatrix argument, must have 10 IGNs, not {len(igns)}")
        
        return IGNMatrixFixed(igns)


class IGNMatrixInfer(IGNMatrix):
    """
    IGN Inference has 3 steps, fixed, semi-fixed, matrix
    1. Fixed      - a pseudoIGN is compared to the list of known IGNs
    2. Semi-Fixed - a pseudoIGN is tested against all of the semi-fixed IGN names-dicts
    3. Matrix     - a pseudoIGN is tested against all other pseudoIGN names-dicts
    When a pseudoIGN is found in either semi-fixed or matrix names-dicts, the occurrence of that pseudoIGN is incremented
    If a matrix names-dict reaches the Assimilation threshold of occurrences, that pseduo IGN names-dict is promoted to semi-fixed
    Once len(Fixed) + len(Semi-Fixed) == 10, no more psuedoIGN names-dict can be promoted and assumed to be invalid IGNs
    A `names-dict: dict[str, int]` is a dictionary containing the occurrency count for each pseudoIGN of an unknown true IGN
    """
    VALID_THRESHOLD = 0.75   ## threshold for Levenshtein distance to determine equality
    ASSIMILATE_THRESHOLD = 5 ## how many times a pseudoIGN has to be seen before adding to 'semi-fixed' list

    def __init__(self, igns: list[str]) -> None:
        self.__fixmat = igns
        self.__fixlen = len(igns)

        self.__semi_fixmat: dict[int,dict[str,int]] = {}
        self.__matrix:      dict[int,dict[str,int]] = {}
        self.__semi_fixlen = 0
        self.__idx_counter = self.__fixlen
    
    def get(self, pign: str, threshold: float = VALID_THRESHOLD) -> Player | None:
        ## first check the fixed igns
        if self.__fixlen > 0 and (pl := IGNMatrix._check_fixed(self.__fixmat, pign, threshold)) is not None:
            return pl
        
        ## second, check the semi-fixed igns
        for idx, names_dict in self.__semi_fixmat.items():
            if (pl := self.__in_names_dict(pign, idx, names_dict, threshold)):
                return pl
        
        ## if all fixed and semi-fixed igns have been found, assume pseudoIGN is invalid
        if self.__fixlen + self.__semi_fixlen >= 10:
            return None

        ## if not all semi-fixed igns have been found, check matrix, and assimilate if necessary
        for idx, names_dict in self.__matrix.items():
            if (pl := self.__in_names_dict(pign, idx, names_dict, threshold)):
                self.__check_assimilation(idx, names_dict)
                return pl

        ## if ign has not been seen, add to matrix
        self.__matrix[self.__idx_counter] = { pign: 1 }
        self.__idx_counter += 1
        return Player(idx, pign)

    def __in_names_dict(self, pign: str, idx: int, names_dict: dict, threshold: float) -> Player | None:
        """
        This method determines whether a pseudoIGN/pign belongs to a specified names_dict,
          and increases the occurrence counts if it does.
        """
        if pign in names_dict:
            names_dict[pign] += 1 # increase occurrence count for pseudoIGN
            return Player(idx, IGNMatrixInfer._max_dict(names_dict))
        
        for seen_pign in names_dict.keys():
            if IGNMatrix._compare_names(pign, seen_pign) > threshold:
                names_dict[pign] = 1  # add to names_dict as a possible true IGN
                return Player(idx, IGNMatrixInfer._max_dict(names_dict))

        return None
    
    def __check_assimilation(self, idx: int, names_dict: dict[str, int]) -> None:
        """
        Checks to see if a specified names_dict has passed over the Assimilation threshold, 
          and if so assimilate to semi-fixed matrix
        """
        if sum(names_dict, key=names_dict.get) >= IGNMatrixInfer.ASSIMILATE_THRESHOLD:
            self.__semi_fixmat[idx] = self.__matrix.pop(idx)
            self.__semi_fixlen += 1
        
    
    def from_idx(self, idx: int) -> str | None:
        """Returns the (most likely) IGN from a given idx"""
        if 0 <= idx < self.__fixlen:
            return self.__fixmat[idx]
        elif idx in self.__semi_fixmat:
            return IGNMatrixInfer._max_dict(self.__semi_fixmat[idx])
        elif idx in self.__matrix:
            return IGNMatrixInfer._max_dict(self.__matrix[idx])

        return None
    
    def get_team(self, ign: str | int) -> int | None:
        ...

    def get_players(self, flip: bool = False) -> list[Player]:
        igns = ... ## TODO

    @staticmethod
    def new(igns: list[str]) -> 'IGNMatrixInfer':
        """Type checking of `igns` is done by the `__cparse_IGNS` function"""
        return IGNMatrixInfer(igns)

    @staticmethod
    def _max_dict(_dict: dict[str: int]) -> str:
        return max(_dict, key=_dict.get)

"""
    self.__team_table = { self.__igns.index(ign): int(idx >= 5) for idx, ign in enumerate(igns) if ign is not None }


    def get_team(self, ign: str | int) -> int | None:
        ""
        Fixed:
            first 5 igns provided are team idx 0, last 5 are team idx 1
        Infer:
            a team index table will be created, and records 
        ""
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
"""

@dataclass
class Timestamp:
    minutes: int
    seconds: int

    def __str__(self) -> str:
        return f"{self.minutes}:{self.seconds:02}"
    
    def to_int(self) -> int:
        return self.minutes * 60 + self.seconds

    @staticmethod
    def from_int(num: int) -> 'Timestamp':
        if num < 0:
            return Timestamp(0, num)
        else:
            return Timestamp(*divmod(num, 60))



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
    end_round: bool
    bomb_planted: bool
    last_ten: int


class WinCondition(StrEnum):
    KILLED_OPPONENTS = "KilledOpponents"
    TIME = "Time"
    DEFUSED_BOMB = "DefusedBomb"
    DISABLED_DEFUSER = "DisabledDefuser"


class History(dict):
    def __init__(self, verbose: int) -> None:
        self.__roundn = -1
        self.new_round(-1)

        self.__verbose = verbose
    
    def get(self, key: str):
        return self[self.__roundn].get(key, None)
    
    @property
    def roundn(self) -> int:
        return self.__roundn
    
    def get_round(self, roundn: int = None) -> dict:
        if roundn is None: roundn = self.__roundn
        return self[roundn]
    
    def set(self, key: str, value) -> None:
        """Sets the values of elements in the history attributes; appending values to the `killfeed` element"""
        APPEND_KEYS = ["killfeed"]
        if key in APPEND_KEYS:
            self[self.__roundn][key].append(value)
        else:
            self[self.__roundn][key] = value
        self[self.__roundn]["updates"] += 1

        if self.__verbose > 1:
            print(f"{self.__roundn}|{key}: {self[self.__roundn][key]}")
        elif self.__verbose > 2:
            self.print()

    def new_round(self, round_number: int) -> None:
        self.__roundn = round_number
        self[round_number] = {
            "scoreline": None,
            "bomb_planted_at": None,
            "bomb_defused_at": None,
            "round_end_at": None,
            "win_condition": None,
            "winner": None,
            "killfeed": [],
            "updates": 0
        }
        
    def to_json(self) -> dict:
        if -1 in self and self[-1]["updates"] == 0:
            self.pop(-1)

        out = {}
        for ridx, round in self.items():
            out[ridx] = {}
            for key, value in round.items():
                match key:
                    case "killfeed":
                        out[ridx][key] = [record.to_json() for record in round[key]]
                    case "updates":
                        continue
                    case _:
                        out[ridx][key] = value
        return out

    def print(self) -> None:
        print(self.__roundn, self[self.__roundn])


@dataclass
class SaveFile:
    filename: str
    ext: str

    def __str__(self) -> str: return f"{self.filename}.{self.ext}"
    __repr__ = __str__

    def copy(self) -> 'SaveFile':
        return SaveFile(self.filename, self.ext)


class SaveManager:
    """
    This class manages everything to do with saving/uploading round data to json/xlsx/google sheets
    """
    def __init__(self, savefile: SaveFile, history: History, ign_matrix: IGNMatrix) -> None:
        self.__savefile = savefile
        self.__history = history
        self.__ignmat = ign_matrix

    def save(self, append: bool = False) -> None:
        if not exists("saves"):
            mkdir("saves")

        match self.__savefile.ext:
            case "json":
                self._save_json()
            case "xlsx":
                self._save_xlsx(append)

    def _save_json(self) -> None:
        with open(join("saves", str(self.__savefile)), "w") as f_out:
            json.dump(self.__history.to_json(), f_out, indent=4)
    

    def _save_xlsx(self, append: bool = False) -> None:
        # Create a new workbook
        savepath = join("saves", str(self.__savefile))
        if append and exists(savepath):
            try:
                workbook = load_workbook(savepath)
            except PermissionError:
                print(f"ERROR: Permission Denied! Cannot open save file {savepath}, file may already be open.")
                workbook = Workbook()
                temp_savefile = self.__savefile.copy()
                temp_savefile.filename += " - Copy"
                savepath = join("saves", str(self.__savefile))
                append = False
        else:
            workbook = Workbook()

        ## remove default worksheet
        if "Sheet" in workbook.sheetnames:
            del workbook["Sheet"]
        
        if -1 in self.__history and self.__history[-1]["updates"] == 0:
            self.__history.pop(-1)

        self.__players = self.__ignmat.get_players()
        rounds = [self.__history.roundn] if append else list(self.__history.keys())
        data = self.__get_xlsx_match() | self.__get_xlsx_rounds(rounds)

        # Add new sheets and fill them with data
        for sheet_name, data in data.items():
            if sheet_name in workbook.sheetnames:
                del workbook[sheet_name]

            sheet = workbook.create_sheet(title=sheet_name)
            for row in data:
                sheet.append(row)

        # Save the new workbook
        workbook.save(savepath)

    
    def __get_xlsx_match(self) -> dict:
        headers = [
            ["Statistics"],
            ["Player", "Team Index", "Rounds", "Kills", "Deaths"]]
        
        kills, deaths = self.__get_xlsx_kd()
        dataT = [
            [pl.ign for pl in self.__players],
            [0]*5 + [1]*5,
            [self.__history.roundn]*10,
            kills,
            deaths
        ]
        return { "Match": headers + transpose(dataT) }

    def __get_xlsx_kd(self) -> tuple[list[int], list[int]]:
        kills, deaths = [0]*10, [0]*10

        for round in self.__history.values():
            for record in round["killfeed"]: ## TODO: fix for infer cases when idx >= 10
                kills[record.player_idx] += 1
                deaths[record.target_idx] += 1

        return kills, deaths

    def __get_xlsx_rounds(self, round_nums: list[int]) -> dict:
        data = {}
        for rn in round_nums:
            round = self.__history.get_round(rn)
            data[f"Round {rn}"] = self.__get_xlsx_rdata(round)
        
        return data

    def __get_xlsx_rdata(self, round: dict) -> list[list]:
        rdata = [
            ["Statistics"],
            ["Player", "Team Index", "Kills", "Deaths", "Assissts", "Hs%", "Headshots", "1vX", "Operator"]
        ]
        ## Statistics Section
        kills, deaths = [0]*10, [False]*10
        for record in round["killfeed"]:
            kills[record.player_idx] += 1
            deaths[record.target_idx] = True

        onevx = None
        onevx_count = 0
        if (w := round["winner"]) is not None:
            if deaths[w*5:(w+1)*5].count(False) == 1: ## possible 1vX
                onevx = deaths.index(False)
                for record in reversed(round["killfeed"]):
                    if record.player_idx == onevx:
                        onevx_count += 1
                    else:
                        break
                onevx_count += deaths[(1-w)*5:(2-w)*5].count(False)


        for pl in self.__players: ## TODO: may have to change pl.idx to i, pl enumerated
            onevx_pl = 0 if pl.idx != onevx else onevx_count
            rdata.append([pl.ign, self.__ignmat.get_team(pl.idx), kills[pl.idx], deaths[pl.idx], "", "", "", onevx_pl, ""])

        ## Round Info
        if len(round["killfeed"]) == 0: ## highly-unlikely, maybe a issue occurred during recording
            opening_kd = KFRecord("", -1, "", -1, "")
        else:
            opening_kd = round["killfeed"][0]

        rdata.extend([
            [],
            ["Round Info"],
            ["Name", "Value", "Time"],
            ["Site"],
            ["Winning team", f"[{round['winner']}]"],
            ["Win condition", round["win_condition"]],
            ["Opening kill", opening_kd.player, opening_kd.time],
            ["Opening death", opening_kd.target, opening_kd.time],
            ["Planted at", round["bomb_planted_at"]],
            ["Defused at", round["bomb_defused_at"]]
        ])

        ## Kill/death feed
        rdata.extend([
            [],
            ["Kill/death feed"],
            ["Player", "Target", "Time", "Traded"],
        ])
        for i, record in enumerate(round["killfeed"]):
            record: KFRecord
            traded = i+1 < len(round["killfeed"]) \
                and time_delta(record.time, round["killfeed"][i+1].time, round["bomb_planted_at"]) <= 6 \
                and self.__ignmat.get_team(record.player_idx) != self.__ignmat.get_team(record.target_idx)
            rdata.append([record.player, record.target, record.time, traded])
        
        return rdata

    def upload_save(self) -> None:
        ...


class Analyser(ABC):
    """
    Main class `Analyser`
    Operates the main inference loop `run` and records match/round information
    """
    PROB_THRESHOLD = 0.5

    SHARPEN_KERNEL = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]])
    
    def __init__(self, args: Namespace):
        self.config:    dict = args.config
        self.verbose:    int = args.verbose
        self.atk_first: bool = True ## TODO: infer
        self.prog_args = args
        self._debug_print(f"Config Keys -", list(self.config.keys()))

        if args.check:
            self.check()

        self.running = False
        self.tdelta: float = self.config.get("SCREENSHOT_PERIOD", 1.0)
        
        self.ign_matrix = IGNMatrix.new(self.config["IGNS"], self.config["IGN_MODE"])
        self.reader = easyocr.Reader(['en'], gpu=not args.cpu)
        self._verbose_print(0, "EasyOCR Reader model loaded")
        
        self.state = State(False, True, False, False)
        self.history = History(self.verbose)
        self.save_manager = SaveManager(self.prog_args.save, self.history, self.ign_matrix)

        self.current_time: Timestamp = None
        self.defuse_countdown_timer = None


    def check(self) -> None:
        regions = self.__get_ss_regions(self._get_ss_region_keys())
        self.__save_regions(regions)
        sys_exit()

    def __save_regions(self, regions: list[np.ndarray]) -> None:
        if not exists("images"):
            mkdir("images")

        print("Info: Saving check images")
        for region_name, region_image in zip(self._get_ss_region_keys(), regions):
            if region_name == "KILL_FEED_REGION":
                region_image = self._killfeed_preprocess(region_image)
            Image.fromarray(region_image).save(join("images", f"{region_name}.jpg"))


    @abstractmethod
    def _get_ss_region_keys(self) -> list[str]:
        ...

    def __get_ss_regions(self, regions: list[str]) -> list[np.ndarray]:
        """Takes a screenshot of the screen, selects regions, and returns them as numpy.ndarray"""
        screenshot = pyautogui.screenshot(allScreens=True)
        return [np.array(screenshot.crop(Analyser.convert_region(self.config[region])), copy=False) for region in regions]

    @staticmethod
    def convert_region(region: list[int]) -> list[int]:
        """Converts (X,Y,W,H) -> (Left,Top,Right,Bottom)"""
        left, top, width, height = region
        return (left, top, left + width, top + height)


    def run(self):
        self.running = True
        self._verbose_print(0, "Running...")

        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time():
                continue

            __inference_start = time()
            regions = self.__get_ss_regions(self._get_ss_region_keys())

            team1_scoreline, team2_scoreline, timer, feed = regions
            self._handle_scoreline(team1_scoreline, team2_scoreline)
            self._handle_timer(timer)
            self._read_feed(feed)
            # self._test_timer(timer)

            self._debug_print(f"Inference time {time()-__inference_start:.2f}s")
            self.timer = time()
    
    ## ----- OCR -----
    def _killfeed_preprocess(self, image: np.ndarray) -> np.ndarray:
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.fastNlMeansDenoising(image, None, 5, 7, 21)
    
        # image = cv2.filter2D(image, -1, Analyser.SHARPEN_KERNEL)
        sf = self.config.get("SCREENSHOT_RESIZE_FACTOR", 2)
        new_width = int(image.shape[1] * sf * 0.75)
        new_height = int(image.shape[0] * sf)
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return image

    def _screenshot_preprocess(self, image: np.ndarray,
                                to_gray: bool = True,
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
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def _readtext(self, image: np.ndarray, prob: float=PROB_THRESHOLD, allowlist: str | None = None) -> list[str]:
        """Performs the EasyOCR inference and cleans the output based on the model's assigned probabilities and a threshold"""
        results = self.reader.readtext(image, allowlist=allowlist)
        return [out[1] for out in results if out[2] > prob]


    ## ----- IN ROUND OCR FUNCTIONS -----
    @abstractmethod
    def _handle_scoreline(self, team1_scoreline: np.ndarray, team2_scoreline: np.ndarray) -> None:
        ...
    
    @abstractmethod
    def _handle_timer(self, timer: np.ndarray) -> None:
        ...
    
    @abstractmethod
    def _read_feed(self, feed: np.ndarray) -> None:
        ...

    ## ----- GAME STATE FUNCTIONS -----
    @abstractmethod
    def _new_round(self, score1: int, score2: int) -> None:
        ...

    @abstractmethod
    def _end_round(self) -> None:
        ...    

    # ----- ENG OF PROGRAM / SAVING -----
    def _end_game(self) -> None:
        self.history.set("winner", self.__ask_winner())
        self.save_manager.save()
        print(f"Data Saved to {self.prog_args.save}, program terminated.")
        sys_exit()
    
    def __ask_winner(self) -> int:
        while (winner := input("Who won the last round? (0/1) >> ")) not in ["0", "1"]:
            print("Invalid option, pick either 0 or 1")
            continue

        return int(winner)

    ## ----- PRINT FUNCTION -----
    def _verbose_print(self, verbose_value: int, *prompt) -> None:
        if self.verbose > verbose_value:
            print("Info:", *prompt)

    def _debug_print(self, *prompt: str) -> None:
        if self.verbose == 3:
            print("Debug:", *prompt)


class InPersonAnalyser(Analyser):
    NUM_LAST_SECONDS = 4   ## number of seconds to continue reading killfeed after round end (reliability reasons)
    END_ROUND_SECONDS = 12 ## number of seconds to check no timer to determine round end
    
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TIMER_REGION", "KILL_FEED_REGION"]

    RED_THRESHOLD = 0.73
    RED_RGB_SPACE = np.array([ ## Defines the range for red color in HSV space
        [240, 10, 10],
        [255, 35, 35]])

    def __init__(self, args) -> None:
        super(InPersonAnalyser, self).__init__(args)

        self.last_kf_seconds = None
        self.end_round_seconds = None

    def _get_ss_region_keys(self) -> list[str]:
        return InPersonAnalyser.SCREENSHOT_REGIONS

    ## ----- SCORELINE -----
    def _handle_scoreline(self, team1_scoreline: np.ndarray, team2_scoreline: np.ndarray) -> None:
        """Extracts the current scoreline visible and determines when a new rounds starts"""
        if not self.state.end_round: return

        img1 = self._screenshot_preprocess(team1_scoreline, squeeze_width=0.5)
        img2 = self._screenshot_preprocess(team2_scoreline, squeeze_width=0.5)

        allowlist = "0123456789"
        results = self._readtext(img1, allowlist=allowlist) + self._readtext(img2, allowlist=allowlist)
        if len(results) != 2:
            return
        if not fullmatch(r"^\d+$", results[0]) or not fullmatch(r"^\d+$", results[1]):
            return
        
        score1, score2 = map(int, results)
        self._new_round(score1, score2)


    ## ----- TIMER FUNCTION -----
    def _handle_timer(self, timer_image: np.ndarray) -> None:
        new_time = self.__read_timer(timer_image)
        
        if new_time is not None: ## timer is showing
            self.current_time = new_time
            self.defuse_countdown_timer = None

            self.last_kf_seconds = None
            self.end_round_seconds = None
        
        elif self.__is_bomb_countdown(timer_image):
            if self.defuse_countdown_timer is None: ## bomb planted
                self.defuse_countdown_timer = time()
                self.history.set("bomb_planted_at", str(self.current_time))

                self.state.bomb_planted = True
                self._verbose_print(0, f"{self.current_time}: BOMB PLANTED")
            else:
                self.current_time = Timestamp.from_int(self.current_time.to_int() - int(time() - self.defuse_countdown_timer))

        elif self.last_kf_seconds is None and self.end_round_seconds is None and self.state.in_round:
            self.last_kf_seconds = time()
            self.end_round_seconds = time()
        
        if self.last_kf_seconds is not None and self.last_kf_seconds + InPersonAnalyser.NUM_LAST_SECONDS < time():
            self.last_kf_seconds = None

        if self.end_round_seconds is not None and self.end_round_seconds + InPersonAnalyser.END_ROUND_SECONDS < time():
            self._end_round()
            self.end_round_seconds = None

    
    def __read_timer(self, image: np.ndarray) -> Timestamp | None:
        """
        Reads the current time displayed in the region `TIMER_REGION`
        If the timer is not present, None is returned
        """
        image = self._screenshot_preprocess(image)
        results = self._readtext(image, prob=0.4, allowlist="0123456789:")
        if len(results) == 0: return None

        result = results[0] if len(results) == 1 else "".join(results)
        if (time := search(r"([0-2]):?([0-5]\d)", result)) and self.state.last_ten < 4:
            self.state.last_ten = 0
            return Timestamp(int(time.group(1)), int(time.group(2)))
        elif (time := search(r"(\d):?\d\d", result)):
            self.state.last_ten += 1
            return Timestamp(0, int(time.group(1)))

        return None
    
    def __is_bomb_countdown(self, image: np.ndarray) -> bool:
        """
        When a bomb is planted, the timer is replaced with a majority red circular countdown
        This method detects when the bomb defuse countdown is shown using a majority red threshold
        """
        mask = cv2.inRange(image, InPersonAnalyser.RED_RGB_SPACE[0], InPersonAnalyser.RED_RGB_SPACE[1])

        # Calculate the percentage of red in the image
        red_percentage = np.sum(mask > 0) / mask.size
        self._debug_print(f"{red_percentage=}")
        return red_percentage > InPersonAnalyser.RED_THRESHOLD


    ## ----- KILL FEED -----
    def _read_feed(self, image: np.ndarray) -> None:
        if not self.state.in_round and self.last_kf_seconds is None: return

        image = self._killfeed_preprocess(image) ## remember, image has been resized
        ocr_results = self.reader.readtext(image)
        cleaned_results = self.__clean_kf_results(ocr_results)
        pairs = self.__get_kf_pairs(cleaned_results)

        for left, right in pairs:
            player = self.ign_matrix.get(left, 0.75)
            target = self.ign_matrix.get(right, 0.75)
            
            if player is None or target is None:
                continue ## invalid igns

            record = KFRecord(player.ign, player.idx, target.ign, target.idx, self.__get_time())
            if record not in self.history.get("killfeed"):
                self.history.set("killfeed", record)
                ##self.ign_matrix.update_team_table(player.idx, target.idx)

    def __clean_kf_results(self, results: list[tuple]) -> list[tuple]:
        results = [res for res in results if res[2] > 0.15]
        new_results = []
        for res1 in results:
            for res2 in results:
                if res1 is not res2 and box_collision(res1[0], res2[0]):
                    new_results.append(join_results(res1, res2))
                    break
            else:
                new_results.append(res1)

        return new_results

    def __get_kf_pairs(self, results: list[tuple]) -> list[tuple[str, str]]:
        """
        This helper method attempts to pair up player and target from the ocr results
        The method checks:
        1. if the player bbox is completely left of target bbox
        2. if the midline of the player and target bboxes are less than 20% max_width apart
        If the conditions are met, the player and target are paired
        """
        n = len(results)
        if n <= 1: return [] ## TODO: account for un-alives

        pairs = []
        for i, res1 in enumerate(results[:-1]):
            box1 = bbox_to_rect(res1[0])
            mid1 = box1[1] + box1[3]/2 
            for res2 in results[i+1:]:
                box2 = bbox_to_rect(res2[0])
                mid2 = box2[1] + box2[3]/2
                if box1[0] + box1[2] < box2[0] and abs(mid1-mid2) < 0.2 * max(box1[3], box2[3]):
                    # area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
                    # print(f"{res1[1]}: {area1} | {res2[1]}: {area2}")

                    pairs.append((res1[1], res2[1]))

        return pairs

    def __get_time(self) -> str:
        if self.defuse_countdown_timer is None:
            return str(self.current_time)
        else:
            time_past = time() - self.defuse_countdown_timer
            return f"!0:{int(45-time_past)}"

    def _new_round(self, score1: int, score2: int) -> None:
        """
        When a new round starts, this method is called, initialising a new round history
        The parameters `score1` and `score2` are the current scores displayed at the start of a new round
        """
        ## infer winner of previous round based on new scoreline
        if self.history.get("winner") is None and self.history.get("scoreline") is not None:
            _, _score2 = self.history.get("scoreline")
            self.history.set("winner", int(_score2 < score2)) ## if _score1+1 == score1, return 0

        self.state = State(True, False, False, 0)

        new_round = score1 + score2 + 1
        self.history.new_round(new_round)
        self.history.set("scoreline", [score1, score2])
    
    def _end_round(self) -> None:
        # self.history.set("win_condition", self._get_wincon())
        self.history.set("round_end_at", str(self.current_time))
        self.state = State(False, True, False, 0)

        if self.history.roundn >= self.config["MAX_ROUNDS"]:
            self._end_game()
        elif self.prog_args.upload_save:
            self.save_manager.upload_save()
        else:
            self.save_manager.save(append=self.prog_args.append_save)
    
    def _get_wincon(self) -> WinCondition:
        # if self.state.bomb_planted:
        #     win_con = WinCondition.DISABLED_DEFUSER if self.state.disabled_defuser else WinCondition.DEFUSED_BOMB
        # elif self.current_time == "0:00": ## this could be doubious
        #     win_con = WinCondition.TIME
        #     ## self.__history_set("winner", )
        # else: ## killed opps
        #     ...

        # self.__history_set("win_condition", win_con)
        return WinCondition.KILLED_OPPONENTS


class SpectatorAnalyser(Analyser):
    NUM_LAST_SECONDS = 4
    END_ROUND_SECONDS = 12
    
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TIMER_REGION", "KILL_FEED_REGION"]

    def __init__(self, args) -> None:
        super(SpectatorAnalyser, self).__init__(args)

    def _get_ss_region_keys(self) -> list[str]:
        return SpectatorAnalyser.SCREENSHOT_REGIONS

    ## ----- IN ROUND OCR FUNCTIONS -----
    def _handle_scoreline(self, team1_scoreline: np.ndarray, team2_scoreline: np.ndarray) -> None:
        ...
    
    def _handle_timer(self, timer: np.ndarray) -> None:
        ...
    
    def _read_feed(self, feed: np.ndarray) -> None:
        ...

    ## ----- GAME STATE FUNCTIONS -----
    def _new_round(self, score1: int, score2: int) -> None:
        ...

    def _end_round(self) -> None:
        ...    


def bbox_to_rect(bbox: list[list[int]]) -> list[int]:
    tl, tr, _, bl = bbox
    return [tl[0], tl[1], tr[0]-tl[0], bl[1]-tl[1]]


def box_collision(b1: list[int], b2: list[int]) -> bool:
    return b1[0] <= b2[0]+b2[2] and b1[1] <= b2[1]+b2[3] \
            and b2[0] <= b1[0]+b1[2] and b2[1] <= b1[1]+b1[3]


Result_t: TypeAlias = tuple[list[list[int]],str,float]
def join_results(res1: Result_t, res2: Result_t, sep: str = "_"):
    if res2[0][0][0] < res1[0][0][0]: res1, res2 = res2, res1 ## make res1 the left most result

    box1, box2 = bbox_to_rect(res1[0]), bbox_to_rect(res2[0])
    x, y = min(box1[0], box2[0]), min(box1[1], box2[1])
    x2, y2 = max(box1[0]+box1[2], box2[0]+box2[2]), max(box1[1]+box1[3], box2[1]+box2[3])
    join_box = [[x, y], [x2, y], [x2, y2], [x, y2]]

    join_str  = res1[1] + sep + res2[1]
    join_prob = min(res1[2], res2[2])
    return (join_box, join_str, join_prob)


def transpose(matrix: list[list]) -> list[list]:
    return [list(row) for row in zip(*matrix)]


def time_to_seconds(ts: str, bpat: str = None) -> int:
    m,s = map(int, ts.lstrip("!").split(":"))
    if ts[0] == "!": ## bomb was planted
        assert bpat is not None
        bm, bs = map(int, bpat.split(":"))
        return bm*60 + bs + s -45
    else:
        return m*60 + s

def time_delta(t1: str, t2: str, bpat: str = None) -> int:
    return time_to_seconds(t1, bpat) - time_to_seconds(t2, bpat)