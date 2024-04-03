import json
import pyautogui
import cv2
import numpy as np
import easyocr
from string import ascii_letters
from argparse import Namespace
from re import search, fullmatch
from time import time
from os import mkdir
from os.path import join, exists
from sys import exit as sys_exit
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from Levenshtein import ratio as leven_ratio
from openpyxl import load_workbook, Workbook
from typing import TypeAlias, Optional


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
    - fixed: will return None for all non-fixed IGNs
    - infer: will infer the non-fixed IGNs from the OCR's output
    The matrix will return an index/ID and the true/most-seen IGN when requested using the `get` method
    The matrix uses Levenshtein distance to determine whether two IGNs are the same (could be improved?)
    """

    @abstractmethod
    def get(self, pseudoIGN: str, threshold: float) -> Optional[Player]:
        ...
    
    @abstractmethod
    def from_idx(self, idx: int) -> Optional[Player]:
        ...
    
    @abstractmethod
    def get_team(self, ign: str | int) -> int:
        ...
    
    def update_team_table(self, player: str | int, target: str | int) -> None:
        ...
    
    @abstractmethod
    def get_players(self, flip: bool = False) -> list[Player]:
        ...
    
    @abstractmethod
    def evaluate(self, pign: str) -> float:
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
    def _check_fixed(ign_list: list[str], pign: str, threshold: float) -> Optional[Player]:
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
    
    def get(self, pign: str, threshold: float = VALID_THRESHOLD) -> Optional[Player]:
        """This method is used to request the index/ID and true/most-seen IGN from the pseudoIGN argument."""
        return IGNMatrix._check_fixed(self.__matrix, pign, threshold)

    def from_idx(self, idx: int) -> Optional[Player]:
        if 0 <= idx <= 10:
            return Player(idx, self.__matrix[idx])

        return None
    
    def get_team(self, ign: str | int) -> Optional[int]:
        if type(ign) == str and ((pl := self.get(ign)) is not None):
            return int(pl.idx >= 5)
        elif type(ign) == int and 0 <= ign < 10:
            return int(ign >= 5)

        return None

    def get_players(self, flip: bool = False) -> list[Player]:
        players = [Player(i, ign) for i, ign in enumerate(self.__matrix)]
        return players  if not flip  else  players[5:] + players[:5]
    
    def evaluate(self, pign: str) -> float:
        if pign in self.__matrix: return 1.0
        return max([IGNMatrix._compare_names(pign, name) for name in self.__matrix])

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
    EVAL_DECAY = 0.95

    def __init__(self, igns: list[str]) -> None:
        self.__fixmat = igns
        self.__fixlen = len(igns)

        self.__semi_fixmat: dict[int,dict[str,int]] = {}
        self.__matrix:      dict[int,dict[str,int]] = {}
        self.__semi_fixlen = 0
        self.__idx_counter = self.__fixlen
    
    def get(self, pign: str, threshold: float = VALID_THRESHOLD) -> Optional[Player]:
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
        idx = self.__idx_counter
        self.__matrix[idx] = { pign: 1 }
        self.__idx_counter += 1
        return Player(idx, pign)

    def __in_names_dict(self, pign: str, idx: int, names_dict: dict, threshold: float) -> Optional[Player]:
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
        if sum(names_dict.values()) >= IGNMatrixInfer.ASSIMILATE_THRESHOLD:
            self.__semi_fixmat[idx] = self.__matrix.pop(idx)
            self.__semi_fixlen += 1
        
    
    def from_idx(self, idx: int) -> Optional[str]:
        """Returns the (most likely) IGN from a given idx"""
        if 0 <= idx < self.__fixlen:
            return self.__fixmat[idx]
        elif idx in self.__semi_fixmat:
            return IGNMatrixInfer._max_dict(self.__semi_fixmat[idx])
        elif idx in self.__matrix:
            return IGNMatrixInfer._max_dict(self.__matrix[idx])

        return None
    
    def get_team(self, ign: str | int) -> Optional[int]:
        ...

    def get_players(self, flip: bool = False) -> list[Player]:
        igns = ... ## TODO
    
    def evaluate(self, pign: str) -> float:
        """
        Evaluates an pseudoIGN, determining how close it is to a real ign
        Eval decay: `(IGNMatrixInfer.EVAL_DECAY ** (max_occr - occr)` factor gives emphasis to pseudoIGNs seen more often
        """
        if pign in self.__fixmat or pign in self.__semi_fixmat:
            return 1.0
        
        max_val = 0.0
        for names_dict in self.__matrix.values():
            max_occr = max(names_dict.values())
            evals = [IGNMatrix._compare_names(pign, name) * (IGNMatrixInfer.EVAL_DECAY ** (max_occr - occr)) for name, occr in names_dict.items()]
            max_val = max(max_val, *evals)

        return max_val

    @staticmethod
    def new(igns: list[str]) -> 'IGNMatrixInfer':
        """Type checking of `igns` is done by the `__cparse_IGNS` function"""
        return IGNMatrixInfer(igns)

    @staticmethod
    def _max_dict(_dict: dict[str: int]) -> str:
        return max(_dict, key=_dict.get)


@dataclass
class State:
    in_round:     bool
    end_round:    bool
    bomb_planted: bool
    last_ten:     int


class WinCondition(StrEnum):
    KILLED_OPPONENTS = "KilledOpponents"
    TIME             = "Time"
    DEFUSED_BOMB     = "DefusedBomb"
    DISABLED_DEFUSER = "DisabledDefuser"
    UNKNOWN          = "Unknown"


@dataclass
class Timestamp:
    minutes: int
    seconds: int

    def __sub__(self, other: 'Timestamp') -> int:
        return self.to_int() - other.to_int()

    def to_int(self) -> int:
        return self.minutes * 60 + self.seconds

    @staticmethod
    def from_int(num: int) -> 'Timestamp':
        m = int(num / 60)
        s = num - 60*m
        return Timestamp(m, s)
    
    def __str__(self) -> str:
        if self.seconds < 0: return f"{self.minutes}:{self.seconds:03}"
        else: return f"{self.minutes}:{self.seconds:02}"

    __repr__ = __str__


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
    time: Timestamp
    
    def __eq__(self, other: 'KFRecord') -> bool:
        return self.player_idx == other.player_idx and self.target_idx == other.target_idx
    
    def to_str(self) -> str:
        return f"{self.time}| {self.player} -> {self.target}"

    def to_json(self, ign_matrix: Optional[IGNMatrix] = None) -> dict:
        if ign_matrix is None:
            player = self.player
            target = self.target
        else:
            player = ign_matrix.from_idx(self.player_idx) or self.player
            target = ign_matrix.from_idx(self.target_idx) or self.target

        return {
            "time": str(self.time),
            "player": player,
            "target": target
        }

    __str__ = to_str
    __repr__ = to_str # to_json


class History(dict):
    def __init__(self, verbose: int) -> None:
        self.__roundn = -1
        self.__verbose = verbose
    
    @property
    def is_ready(self) -> bool:
        return self.__roundn > 0
    
    @property
    def roundn(self) -> int:
        return self.__roundn
    
    def set_roundn(self, roundn: int) -> None:
        """This method should only be used if something has gone wrong, `roundn` should be coupled to `new_round` function """
        self.__roundn = roundn
    
    def get(self, key: str):
        if self.__roundn <= 0: return None
        return self[self.__roundn].get(key, None)
    
    def get_round(self, roundn: Optional[int] = None) -> Optional[dict]:
        if roundn is None: roundn = self.__roundn
        if roundn not in self: return None
        return self[roundn]
    
    def set(self, key: str, value) -> None:
        """Sets the values of elements in the history attributes; appending values to the `killfeed` element"""
        if not self.is_ready: return

        APPEND_KEYS = ["killfeed", "deaths"]
        if key in APPEND_KEYS:
            self[self.__roundn][key].append(value)
        else:
            self[self.__roundn][key] = value

        EXCLUDE_PRINT = ["deaths"]
        if self.__verbose > 1 and key not in EXCLUDE_PRINT:
            print(f"{self.__roundn}|{key}: {self[self.__roundn][key]}")
        elif self.__verbose > 2:
            self.print()

    def new_round(self, round_number: int) -> None:
        print(f"Info: New Round {round_number}")
        self.__roundn = round_number
        self[round_number] = {
            "scoreline": None,
            "atk_side": None,
            "bomb_planted_at": None,
            "disabled_defuser_at": None,
            "round_end_at": None,
            "win_condition": None,
            "winner": None,
            "killfeed": [],
            "deaths": []
        }
        
    def to_json(self) -> dict:
        out = {}
        for ridx, round in self.items():
            out[ridx] = {}
            for key, value in round.items():
                if type(value) == Timestamp:
                    out[ridx][key] = str(value)
                    continue

                match key:
                    case "killfeed":
                        out[ridx][key] = [record.to_json() for record in round[key]]
                    case "win_condition":
                        out[ridx][key] = value.value
                    case "deaths":
                        continue
                    case _:
                        out[ridx][key] = value
        return out

    def print(self) -> None:
        print(self.__roundn, self.get_round())


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
    
    def __make_tempfile(self) -> str:
        temp_savefile = self.__savefile.copy()
        temp_savefile.filename += " - Copy"
        return join("saves", str(temp_savefile))

    def _save_xlsx(self, append: bool = False) -> None:
        # Create a new workbook
        savepath = join("saves", str(self.__savefile))
        if append and exists(savepath):
            try:
                workbook = load_workbook(savepath)
            except PermissionError:
                print(f"ERROR: Permission Denied! Cannot open save file {savepath}, file may already be open. Defaulting to append-save=False")
                workbook = Workbook()
                savepath = self.__make_tempfile()
                append = False
        else:
            workbook = Workbook()

        ## remove default worksheet
        if "Sheet" in workbook.sheetnames:
            del workbook["Sheet"]

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
        try:
            workbook.save(savepath)
        except PermissionError:
            new_savepath = self.__make_tempfile()
            print(f"ERROR: Permission Denied! Cannot save to {savepath}, file may already be open. Saving to {new_savepath}")
            workbook.save(new_savepath)

    
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
            if (round := self.__history.get_round(rn)) is not None:
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

        bpat = round["bomb_planted_at"]
        ddat = round["disabled_defuser_at"]
        rdata.extend([
            [],
            ["Round Info"],
            ["Name", "Value", "Time"],
            ["Site"],
            ["Winning team", f"[{round['winner']}]"],
            ["Win condition", round["win_condition"].value],
            ["Opening kill", opening_kd.player, str(opening_kd.time)],
            ["Opening death", opening_kd.target, str(opening_kd.time)],
            ["Planted at", str(bpat) if bpat is not None else ""],
            ["Defused at", str(ddat) if ddat is not None else ""]
        ])

        ## Kill/death feed
        rdata.extend([
            [],
            ["Kill/death feed"],
            ["Player", "Target", "Time", "Traded", "Refragged Death", "Refragged Kill"],
        ])
        refragged_kills = []
        for i, record in enumerate(round["killfeed"]):
            record: KFRecord
            n = len(round["killfeed"])
            if i+1 == n:
                rdata.append([record.player, record.target, str(record.time), False, False, i in refragged_kills])
                break

            traded = i+1 < n \
                and (record.time - round["killfeed"][i+1].time) <= 6 \
                and self.__ignmat.get_team(record.player_idx) != self.__ignmat.get_team(record.target_idx)
            
            refragged_death = False
            for j, r2 in enumerate(round["killfeed"][i+1:], start=i+1):
                r2: KFRecord
                if record.time - r2.time > 6: break
                if record.player_idx == r2.target_idx:
                    refragged_death = True
                    refragged_kills.append(j)

            rdata.append([record.player, record.target, str(record.time), traded, refragged_death, i in refragged_kills])
        
        return rdata

    def upload_save(self) -> None:
        ...


class Analyser(ABC):
    """
    Main class `Analyser`
    Operates the main inference loop `run` and records match/round information
    """
    PROB_THRESHOLD = 0.5
    KF_ALLOWLIST = ascii_letters + "0123456789.-_"
    
    def __init__(self, args: Namespace):
        self.config: dict = args.config
        self.verbose: int = args.verbose
        self.prog_args = args
        self._debug_print(f"Config Keys -", list(self.config.keys()))

        if args.check:
            self.check()
            if not args.test: sys_exit()

        self.running = False
        self.tdelta: float = self.config.get("SCREENSHOT_PERIOD", 1.0)
        
        self.ign_matrix = IGNMatrix.new(self.config["IGNS"], self.config["IGN_MODE"])
        self.reader = easyocr.Reader(['en'], gpu=not args.cpu)
        self._verbose_print(0, "EasyOCR Reader model loaded")

        self.state = State(False, True, False, False)
        self.history = History(self.verbose)
        self.save_manager = SaveManager(self.prog_args.save, self.history, self.ign_matrix)

        self.current_time: Timestamp = None
        self.defuse_countdown_timer: float = None

    ## ----- CHECK -----
    def check(self) -> None:
        if not exists("images"):
            mkdir("images")

        print("Info: Saving check images")
        region_keys = self._get_ss_region_keys()
        regions = self._get_ss_regions(region_keys)
        for name, img in regions.items():
            cv2.imwrite(join("images", f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # ----- TEST -----
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
            region: np.array(screenshot.crop(Analyser.convert_region(self.config[region])), copy=False)
            for region in region_keys
        }

    @staticmethod
    def convert_region(region: list[int]) -> list[int]:
        """Converts (X,Y,W,H) -> (Left,Top,Right,Bottom)"""
        left, top, width, height = region
        return (left, top, left + width, top + height)

    # ----- MAIN LOOP -----
    def run(self):
        self.running = True
        self._verbose_print(0, "Running...")

        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time():
                continue

            __inference_start = time()

            regions = self._get_ss_regions()
            self._handle_scoreline(regions)
            self._handle_timer(regions)
            self._handle_feed(regions)

            self._debug_print(f"Inference time {time()-__inference_start:.2f}s")
            self.timer = time()
    
    ## ----- OCR -----
    def _screenshot_preprocess(self,
                               image: np.ndarray,
                               to_gray: bool = True,
                               denoise: bool = False,
                               squeeze_width: float = -1) -> np.ndarray:
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

        scale_factor = self.config.get("SCREENSHOT_RESIZE", 4)
        if squeeze_width != -1:
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
    def _fix_state(self, roundn: int) -> None:
        ...    

    # ----- ENG OF PROGRAM / SAVING -----
    @abstractmethod
    def _end_game(self) -> None:
        ...
    
    def _ask_winner(self) -> int:
        while (winner := input("Who won the last round? (0/1) >> ")) not in "01":
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
    
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION",
                          "TEAM1_SIDE_REGION", "TEAM2_SIDE_REGION",
                          "TIMER_REGION",
                          "KILLFEED_REGION"]

    SCORELINE_PROB = 0.25
    TIMER_PROB = 0.35
    KF_PROB = 0.15

    PROXIMITY_DIST = 35

    RED_THRESHOLD = 0.73
    RED_RGB_SPACE = np.array([ ## Defines the range for red color in HSV space
        [240, 10, 10],
        [255, 35, 35]])

    def __init__(self, args) -> None:
        super(InPersonAnalyser, self).__init__(args)

        self.last_kf_seconds = None
        self.end_round_seconds = None

        self.atkside_icon = self.__load_side_icon()
    
    def __load_side_icon(self) -> np.ndarray:
        assert exists("res/swords.jpg"), "res/swords.jpg does not exist!"
        img = cv2.imread("res/swords.jpg", cv2.IMREAD_GRAYSCALE)
        side_region_size = self.config["TEAM1_SIDE_REGION"][2:]
        return cv2.resize(img, side_region_size, interpolation=cv2.INTER_LINEAR)

    def _get_ss_region_keys(self) -> list[str]:
        return InPersonAnalyser.SCREENSHOT_REGIONS
    
    def test(self) -> None:
        regions   = self._get_ss_regions()
        scoreline = self.__read_scoreline(regions["TEAM1_SCORE_REGION"], regions["TEAM2_SCORE_REGION"])
        atkside   = self.__read_atkside(regions["TEAM1_SIDE_REGION"], regions["TEAM2_SIDE_REGION"])
        time_read = self.__read_timer(regions["TIMER_REGION"])
        feed_read = self.__read_feed(regions["KILLFEED_REGION"])

        print(f"Test: {scoreline=} | {atkside=} | {time_read} | ", end="")
        for left, right in feed_read:
            print(f"{left} -> {right}, ", end="")
        print()

    ## ----- SCORELINE -----
    def _handle_scoreline(self, regions: Analyser.Regions_t) -> None:
        """Extracts the current scoreline visible and determines when a new rounds starts"""
        if not self.state.end_round: return

        scores = self.__read_scoreline(regions["TEAM1_SCORE_REGION"], regions["TEAM2_SCORE_REGION"])
        if not scores: return

        new_roundn = sum(scores)+1
        if new_roundn in self.history:
            self._fix_state(new_roundn)

        self._new_round(*scores)

        atkside = self.__read_atkside(regions["TEAM1_SIDE_REGION"], regions["TEAM2_SIDE_REGION"])
        self.history.set("atk_side", atkside)
    
    def __read_scoreline(self, scoreline1: np.ndarray, scoreline2: np.ndarray) -> Optional[tuple[int, int]]:
        img1 = self._screenshot_preprocess(scoreline1, to_gray=True, squeeze_width=0.65)
        img2 = self._screenshot_preprocess(scoreline2, to_gray=True, squeeze_width=0.65)

        allowlist = "0123456789"
        results = self._readtext(img1, InPersonAnalyser.SCORELINE_PROB, allowlist=allowlist) + self._readtext(img2, InPersonAnalyser.SCORELINE_PROB, allowlist=allowlist)
        if len(results) != 2:
            return None
        if not fullmatch(r"^\d+$", results[0]) or not fullmatch(r"^\d+$", results[1]):
            return None
        
        return tuple(map(int, results))
    
    def __read_atkside(self, side1: np.ndarray, side2: np.ndarray) -> Optional[int]:
        """
        Matches the side icon next to the scoreline with `res/swords.jpg` to determine which side is attack.
        """
        icon1 = self._screenshot_preprocess(side1, to_gray=True, denoise=True)
        icon2 = self._screenshot_preprocess(side2, to_gray=True, denoise=True)

        res_icon1 = cv2.matchTemplate(self.atkside_icon, icon1, cv2.TM_CCOEFF_NORMED)
        res_icon2 = cv2.matchTemplate(self.atkside_icon, icon2, cv2.TM_CCOEFF_NORMED)

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
        """
        """
        if not self.history.is_ready: return

        timer_image = regions["TIMER_REGION"]
        new_time = self.__read_timer(timer_image)

        if new_time is not None: ## timer is showing
            self.current_time = new_time
            self.defuse_countdown_timer = None

            self.last_kf_seconds = None
            self.end_round_seconds = None
        
        elif self.__is_bomb_countdown(timer_image):
            if self.defuse_countdown_timer is None: ## bomb planted
                self.defuse_countdown_timer = time()
                self.history.set("bomb_planted_at", self.current_time)

                self.state.bomb_planted = True
            else:
                bomb_planted_at: Timestamp = self.history.get("bomb_planted_at")
                self.current_time = Timestamp.from_int(bomb_planted_at.to_int() - int(time() - self.defuse_countdown_timer))

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
        if (time := search(r"([0-2]).?([0-5]\d)", result)) and self.state.last_ten < 4:
            self.state.last_ten = 0
            return Timestamp(int(time.group(1)), int(time.group(2)))
        elif (time := search(r"(\d).?\d\d", result)):
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
    def _handle_feed(self, regions: Analyser.Regions_t) -> None:
        """
        """
        if not self.history.is_ready: return
        if not self.state.in_round and self.last_kf_seconds is None: return

        image = regions["KILLFEED_REGION"]
        for left, right in self.__read_feed(image):
            player = self.ign_matrix.get(left, 0.75)
            target = self.ign_matrix.get(right, 0.75)
            
            if player is None or target is None:
                continue ## invalid igns

            record = KFRecord(player.ign, player.idx, target.ign, target.idx, self.current_time)
            if record not in self.history.get("killfeed"):
                self.history.set("killfeed", record)
                self.history.set("deaths", target.idx)
                ##self.ign_matrix.update_team_table(player.idx, target.idx)
    
    def __read_feed(self, image: np.ndarray) -> list[tuple[str, str]]:
        image = self._screenshot_preprocess(image, to_gray=True, denoise=True, squeeze_width=0.75)

        ocr_results = self.reader.readtext(image, allowlist=Analyser.KF_ALLOWLIST)
        for i, res in enumerate(ocr_results):
            ocr_results[i] += (self.ign_matrix.evaluate(res[1]), )

        cleaned_results = self.__clean_kf_results(ocr_results)
        pairs, boxes = self.__get_kf_pairs(cleaned_results)
        self.__test_pairs(image, boxes)
        return pairs

    def __clean_kf_results(self, results: list[tuple]) -> list[tuple]:
        """
        To clean the IGNMatrix evaluated easyocr results, the method does:
        1. filters out any result with an easyocr probability score < KF_PROB  (~0.15)
        2. Attempt to pair up boxes which have been split for a single IGN (easyocr struggles with underscores and many dots)
          a. if the boxes are colliding or if their closest vertical edges are <= PROXIMITY_DIST, and inline with one-another
          b. join the boxes and re-evaluate the concatenated ign, if the eval score > than either parent box
          c. replace the joined box with res1
        """
        results = [res for res in results if res[2] > InPersonAnalyser.KF_PROB]
        new_results = []
        for res1 in results:
            for res2 in results:
                r1, r2 = bbox_to_rect(res1[0]), bbox_to_rect(res2[0]) ## convert from easyocr bounding-box to rect (x,y,w,h)
                prox_dist = InPersonAnalyser.PROXIMITY_DIST * (self.config.get("SCREENSHOT_RESIZE", 4) / 4)
                if res1 is not res2 and (rect_collision(r1, r2) or rect_proximity(r1, r2) <= prox_dist):
                    jr = join_results(res1, res2)
                    eval_score = self.ign_matrix.evaluate(jr[1]) ## re-evaluate joined text
                    if eval_score > max(res1[3], res2[3]):
                        jr += (eval_score, ) ## add eval to joined-result
                        new_results.append(jr)
                        break
            else:
                new_results.append(res1)

        return new_results

    def __get_kf_pairs(self, results: list[tuple]) -> tuple[list[tuple[str, str]], list[tuple[list[int], list[int]]]]:
        """
        This helper method attempts to pair up player and target from the ocr results
        The method checks:
        1. if the player bbox is completely left of target bbox
        2. if the midline of the player and target bboxes are less than 20% max_width apart
        If the conditions are met, the player and target are paired
        """
        if len(results) <= 1: return [], [] ## TODO: account for un-alives

        pairs = []
        boxes = []
        for i, res1 in enumerate(results[:-1]):
            box1 = bbox_to_rect(res1[0])
            mid1 = box1[1] + box1[3]/2 
            for res2 in results[i+1:]:
                box2 = bbox_to_rect(res2[0])
                mid2 = box2[1] + box2[3]/2
                if box1[0] + box1[2] < box2[0] and abs(mid1-mid2) < 0.2 * max(box1[3], box2[3]):
                    pairs.append((res1[1], res2[1]))
                    boxes.append((box1, box2))

        return pairs, boxes

    def __test_pairs(self, image: np.ndarray, boxes: list[tuple[str, str]]) -> None:
        if self.prog_args.test:
            rect_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
            for left, right in boxes:
                cv2.rectangle(rect_img, left[:2],  [left[0]+left[2], left[1]+left[3]], (0, 0, 255), 3)
                cv2.rectangle(rect_img, right[:2], [right[0]+right[2], right[1]+right[3]], (0, 255, 0), 3)
            cv2.imwrite("images/TEST_KF_BOXES.jpg", rect_img)

    # ----- GAME STATE -----
    def __pre_new_round(self, score2: int, save: bool = True) -> None:
        if len(self.history) == 0: return

        ## infer winner of previous round based on new scoreline
        if self.history.get("winner") is None and self.history.get("scoreline") is not None:
            _, _score2 = self.history.get("scoreline")
            self.history.set("winner", int(_score2 < score2)) ## if _score1+1 == score1, return 0

        win_con = self._get_wincon()
        self.history.set("win_condition", win_con)

        if win_con == WinCondition.DISABLED_DEFUSER:
            self.history.set("disabled_defuser_at", self.history.get("round_end_at"))

        if not save: return
        ## Save round data once all data can be extracted
        if self.prog_args.upload_save:
            self.save_manager.upload_save()
        else:
            self.save_manager.save(append=self.prog_args.append_save)

    def _new_round(self, score1: int, score2: int) -> None:
        """
        When a new round starts, this method is called, initialising a new round history
        The parameters `score1` and `score2` are the current scores displayed at the start of a new round
        """
        new_round = score1 + score2 + 1
        if new_round in self.history: return

        self.__pre_new_round(score2, save=True)
        self.state = State(True, False, False, 0)

        self.history.new_round(new_round)
        self.history.set("scoreline", [score1, score2])
    
    def _end_round(self) -> None:
        self.history.set("round_end_at", self.current_time)
        self.state = State(False, True, False, 0)

        mxrnd = self.config["MAX_ROUNDS"]
        if self.history.roundn >= mxrnd or (not self.config["SCRIM"] and max(self.history.get("scoreline", 0)) == (mxrnd+1)/2):
            self._end_game()
    
    def _get_wincon(self) -> WinCondition:
        bpat    =   self.history.get("bomb_planted_at")
        winner  =   self.history.get("winner")
        defside = 1-self.history.get("atk_side")

        if bpat is not None and winner == defside:
            return WinCondition.DISABLED_DEFUSER
        elif bpat is not None:
            return WinCondition.DEFUSED_BOMB

        elif 0 <= self.current_time.to_int() <= 1 \
                and winner == defside \
                and self.__wincon_alive_count(defside) > 0:
            return WinCondition.TIME

        elif self.__wincon_alive_count(1-winner) == 0:
            return WinCondition.KILLED_OPPONENTS
        
        return WinCondition.UNKNOWN
    
    def __wincon_alive_count(self, side: int) -> bool:
        alive = 5 ## TODO: history, ign matrix team count
        for d_idx in self.history.get("deaths"):
            if self.ign_matrix.get_team(d_idx) == side:
                alive -= 1
        return alive
    
    def _end_game(self) -> None:
        winner = self._ask_winner()
        self.history.set("winner", winner)
        scoreline = self.history.get("scoreline")[:]
        scoreline[winner] += 1
        self.__pre_new_round(scoreline, save=False)

        self.save_manager.save()
        print(f"Data Saved to {self.prog_args.save}, program terminated.")
        sys_exit()
    
    def _fix_state(self, roundn: int) -> None:
        """
        Hard reset of state in rare occassions when state is wrong,
          should be called when known to be in round
        """
        self.state.in_round = True
        self.state.end_round = False

        timer_region = self._get_ss_regions(["TIMER_REGION"])[0]
        self.state.bomb_planted = self.__is_bomb_countdown(timer_region)

        ct = self.current_time.to_int()
        if ct < 10:
            self.state.last_ten = 10-ct
        self.history.set_roundn(roundn)


class SpectatorAnalyser(Analyser):
    NUM_LAST_SECONDS = 4
    END_ROUND_SECONDS = 12
    
    SCREENSHOT_REGIONS = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TIMER_REGION", "KILLFEED_REGION"]

    def __init__(self, args) -> None:
        super(SpectatorAnalyser, self).__init__(args)

    def _get_ss_region_keys(self) -> list[str]:
        return SpectatorAnalyser.SCREENSHOT_REGIONS
    
    def test(self) -> None:
        ...

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
    
    def _fix_state(self, roundn: int) -> None:
        ...


# ----- HELPER FUNCTIONS -----
def transpose(matrix: list[list]) -> list[list]:
    return [list(row) for row in zip(*matrix)]


def rect_collision(r1: list[int], r2: list[int]) -> bool:
    return r1[0] <= r2[0]+r2[2] and r1[1] <= r2[1]+r2[3] \
            and r2[0] <= r1[0]+r1[2] and r2[1] <= r1[1]+r1[3]


def bbox_to_rect(bbox: list[list[int]]) -> list[int]:
    tl, tr, _, bl = bbox
    return [tl[0], tl[1], tr[0]-tl[0], bl[1]-tl[1]]


def box_collision(b1: list[list[int]], b2: list[list[int]]) -> bool:
    return rect_collision(bbox_to_rect(b1), bbox_to_rect(b2))


def rect_proximity(r1: list[int], r2: list[int]) -> float:
    if not (r1[1] <= r2[1]+r2[3]/2 <= r1[1]+r1[3]): return float("inf")
    return r2[0] - (r1[0]+r1[2])


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
