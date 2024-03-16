import argparse
import json
import pyautogui
import cv2
import numpy as np
import easyocr
from dataclasses import dataclass
from re import search
from time import sleep, time
from os.path import join, exists
from PIL import Image
from Levenshtein import ratio as leven_ratio


class IGNMatrix:
    VALID_THRESHOLD = 0.75
    
    def __init__(self, fixed: int, igns: list[str]) -> None:
        self.__fixed = fixed
        self.__igns = igns
        
    def get(self, ign: str) -> tuple[int, str]:
        ## Check the fixed igns
        fixed_igns = self.__igns[:self.__fixed]
        for i, name in enumerate(fixed_igns):
            if IGNMatrix.__compare_names(ign, name) > IGNMatrix.VALID_THRESHOLD:
                return i, name
        
        if self.__fixed == 10: return None, None

        ## Check the unfixed, infered igns
        unfixed_igns = self.__igns[self.__fixed:]
        for i, names_dict in enumerate(unfixed_igns, start=self.__fixed):
            if ign in names_dict:
                names_dict[ign] += 1
                return i, max(names_dict, key=lambda k: names_dict[k])
            
            scores = [IGNMatrix.__compare_names(ign, name) for name in names_dict.keys()]
            if max(scores) > IGNMatrix.VALID_THRESHOLD:
                names_dict[ign] = 1
                return i, max(names_dict, key=lambda k: names_dict[k])
        
        ## if ign has not been seen, add to matrix
        idx = len(self.__igns)
        self.__igns.append({ign: 1})
        return idx, ign

    @staticmethod
    def new(igns: list[list|str]) -> 'IGNMatrix':
        if len(igns) == 0:
            return IGNMatrix(0, [])
        
        if type(igns[0]) == list:
            if len(igns) == 1:
                if len(igns[0]) == 5 or len(igns[0]) == 10:
                    return IGNMatrix(len(igns[0]), igns[0])
                
                raise ValueError(f"Invalid Config IGN list, only {len(igns[0])} IGNS, must be 5/10")
                
            elif len(igns) == 2:
                if len(igns[0]) != 5: raise ValueError(f"Invalid Config IGN list team 1, only {len(igns[0])} IGNS")
                if len(igns[1]) != 5: raise ValueError(f"Invalid Config IGN list team 2, only {len(igns[1])} IGNS")
                return IGNMatrix(10, igns[0] + igns[1])
        
        elif type(igns[0]) == str:
            if len(igns) == 5 or len(igns) == 10:
                return IGNMatrix(len(igns), igns)
            
            raise ValueError(f"Invalid Config IGN list, only {len(igns)} IGNS, must be 5/10")

        raise ValueError("Invalid Config IGN list")
            
    
    @staticmethod
    def __compare_names(name1: str, name2: str) -> float:
        return leven_ratio(name1.lower(), name2.lower())


@dataclass
class KFRecord:
    player: str
    player_idx: int
    target: str
    target_idx: str
    time: str
    
    def __eq__(self, other: 'KFRecord') -> bool:
        return self.player_idx == other.player_idx and self.target_idx == other.target_idx

@dataclass
class RHRecord:
    bomb_planted_time: str


class Analyser:
    PROB_THRESHOLD = 0.5
    RED_THRESHOLD = 0.85
    
    def __init__(self, args: argparse.Namespace):
        self.config: dict  = args.config
        self.gpu: bool     = not args.cpu
        self.verbose: bool = args.verbose

        self.ign_matrix = IGNMatrix.new(self.config["IGNS"])
        self.kill_feed: list[KFRecord] = []
        self.round_history: list[RHRecord] = []

        self.running = False
        self.tdelta: float = self.config.get("SCREENSHOT_PERIOD", 1.0)
        
        self.reader = easyocr.Reader(['en'], gpu=self.gpu)
        if self.verbose:
            print("Info: EasyOCR Reader model loaded")
        
        self.current_time = 0
        self.no_timer = 0
        self.defuse_countdown_timer = None
                
        self.__red_hsv_space = np.array([  # Define the range for red color in HSV space
            [0, 70, 50],
            [10, 255, 255],
            [170, 70, 50],
            [180, 255, 255]])


    def run(self):
        self.running = True
        # if self.verbose:
        print("Info: Running...")

        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time():
                continue
            
            timer_image = pyautogui.screenshot(region=self.config["TIMER_REGION"])
            feed_image = pyautogui.screenshot(region=self.config["KILL_FEED_REGION"])
            
            self.__handle_timer(timer_image)
            self.__read_feed(feed_image)

            if self.verbose and False:
                print(f"Info: Inference time {time()-self.timer:.2f}s")
            
            self.timer = time()
    
    def __handle_timer(self, timer_image: Image.Image) -> None:
        timer_image_np = np.array(timer_image)
        if self.__is_timer(timer_image_np):
            new_time = self.__read_timer(timer_image_np)
            if new_time is not None:
                self.current_time = new_time
                self.no_timer = 0
                self.defuse_countdown_timer = None
            else:
                self.no_timer += 1

        elif self.defuse_countdown_timer is None: ## bomb planted
            self.defuse_countdown_timer = time()
            self.round_history.append(RHRecord(self.current_time))
            print(f"{self.current_time}: BOMB PLANTED")
    
    def __is_timer(self, image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create masks for red color
        mask1 = cv2.inRange(hsv, self.__red_hsv_space[0], self.__red_hsv_space[1])
        mask2 = cv2.inRange(hsv, self.__red_hsv_space[2], self.__red_hsv_space[3])
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate the percentage of red in the image
        red_percentage = np.sum(red_mask > 0) / red_mask.size
        return red_percentage < Analyser.RED_THRESHOLD


    def __screenshot_process(self, image: Image.Image | np.ndarray, to_gray: bool=True) -> np.ndarray:
        # Convert the PIL Image to a NumPy array (RGB)
        if type(image) != np.ndarray:
            image = np.array(image)

        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        scale_factor = self.config.get("SCREENSHOT_RESIZE_X", 2)
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def __read_timer(self, screenshot: Image.Image | np.ndarray) -> None:
        image = self.__screenshot_process(screenshot)
        results = self.reader.readtext(image)

        cleaned_results = [out[1] for out in results if out[2] > Analyser.PROB_THRESHOLD]
        for read_time in cleaned_results:
            if (time := search(r"(\d?\d)[ \.:](\d\d)", read_time)):
                return f"{time.group(1)}:{time.group(2)}"
        
        return None
    
    def __get_time(self) -> str:
        if self.defuse_countdown_timer is None:
            return self.current_time
        else:
            time_past = time() - self.defuse_countdown_timer
            return f"!0:{int(45-time_past)}"

    def __read_feed(self, screenshot: Image.Image) -> None:
        image = self.__screenshot_process(screenshot)
        results = self.reader.readtext(image)
        
        cleaned_results = [text for (_, text, prob) in results if prob > Analyser.PROB_THRESHOLD]
        if len(cleaned_results) % 2 != 0: return None  ## TODO: could cause an issue when someone c4's themselves
        
        for i in range(0, len(cleaned_results), 2):
            player_ign_raw = cleaned_results[i]
            target_ign_raw = cleaned_results[i+1]
            p_idx, p_name = self.ign_matrix.get(player_ign_raw)
            t_idx, t_name = self.ign_matrix.get(target_ign_raw)
            
            if p_idx is None or t_idx is None: continue
            
            record = KFRecord(p_name, p_idx, t_name, t_idx, self.__get_time())
            if record not in self.kill_feed:
                self.kill_feed.append(record)
                ## print(f"{record.time}: {record.player}/{record.player_idx} -> {record.target}/{record.target_idx}")
                print(f"{record.time}: {record.player} -> {record.target}")


def __parse_config(arg: str) -> dict:
    if not (exists(arg) or exists(join("configs", arg))):
        raise argparse.ArgumentError(f"Config file '{arg}' cannot be found!")
    
    if not exists(arg):
        arg = join("configs", arg)
    
    REQUIRED_CONFIG_KEYS = ["TIMER_REGION", "KILL_FEED_REGION"]
    OPTIONAL_CONFIG_KEYS = ["SCREENSHOT_RESIZE_X", "SCREENSHOT_PERIOD", "IGNS"]
    DEFAULT_CONFIG_FILENAME = "defaults.json"
    with open(arg, "r", encoding="utf-8") as f_in:
        config = json.load(f_in)

    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise argparse.ArgumentError(f"Config file does not contain key '{key}'!")
    
    print(f"Info: Loaded configuration file '{arg}'")
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
        
    return config


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
                        action="store_true",
                        help="Determines how detailed the console output is",
                        dest="verbose")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Flag for only cpu execution, if your machine does not support gpu acceleration")

    args = parser.parse_args()
    if args.delay > 0:
        sleep(args.delay)
    
    Analyser(args).run()
