import argparse
import json
import pyautogui
import cv2
import numpy as np
import easyocr
from re import search
from time import sleep, time
from os.path import join, exists
from PIL import Image
from Levenshtein import ratio as leven_ratio


class Analyser:
    PROB_THRESHOLD = 0.5
    
    def __init__(self, args: argparse.Namespace):
        self.config = args.config
        self.gpu = not args.cpu
        self.verbose = args.verbose

        self.ign_fix, self.ign_matrix = Analyser.__parse_igns(self.config["IGNS"])
        self.kill_feed = []
        
        self.running = False
        self.tdelta = self.config.get("SCREENSHOT_PERIOD", 1.0)
        
        self.reader = easyocr.Reader(['en'], gpu=self.gpu)
        if self.verbose:
            print("Info: EasyOCR Reader model loaded")
        
        self.current_time = 0
        self.no_timer = 0
    
    @staticmethod
    def __parse_igns(igns: list[list[str]]) -> tuple[int, list]:
        if len(igns) == 0:
            return 0, []
        
        elif len(igns) == 1:
            if len(igns[0]) != 5: raise ValueError(f"Invalid Config IGN list, only {len(igns[0])} IGNS")
            return 5, igns[0]

        elif len(igns) == 2:
            if len(igns[0]) != 5: raise ValueError(f"Invalid Config IGN list, only {len(igns[0])} IGNS")
            if len(igns[1]) != 5: raise ValueError(f"Invalid Config IGN list, only {len(igns[1])} IGNS")
            return 10, igns[0] + igns[1]

        else:
            raise ValueError(f"Invalid Config IGN list, too many teams")


    def run(self):
        self.running = True
        if self.verbose:
            print("Info: Running...")

        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time(): continue
            
            ## READ TIMER
            timer_image = pyautogui.screenshot(region=self.config["TIMER_REGION"])
            new_time = self.__read_timer(timer_image)
            if new_time is not None:
                self.current_time = new_time
                self.no_timer = 0
            else:
                self.no_timer += 1
            
            ## READ KILL FEED
            feed_image = pyautogui.screenshot(region=self.config["KILL_FEED_REGION"])
            self.__read_feed(feed_image)

            if self.verbose and False:
                print(f"Info: Inference time {time()-self.timer:.2f}s")
            
            self.timer = time()
            

    def __screenshot_process(self, screenshot: Image.Image) -> np.ndarray:
        # Convert the PIL Image to a NumPy array (RGB)
        img_np = np.array(screenshot)
        image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        scale_factor = self.config.get("SCREENSHOT_RESIZE_X", 2)
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def __read_timer(self, screenshot: Image.Image) -> None:
        image = self.__screenshot_process(screenshot)
        results = self.reader.readtext(image)

        cleaned_results = [out[1] for out in results if out[2] > Analyser.PROB_THRESHOLD]
        for read_time in cleaned_results:
            if (time := search(r"(\d?\d)[ \.:](\d\d)", read_time)):
                return f"{time.group(1)}:{time.group(2)}"
        
        return None

    def __read_feed(self, screenshot: Image.Image) -> None:
        image = self.__screenshot_process(screenshot)
        results = self.reader.readtext(image)
        
        cleaned_results = [text for (_, text, prob) in results if prob > Analyser.PROB_THRESHOLD]
        if len(cleaned_results) % 2 != 0: return None
        
        for i in range(0, len(cleaned_results), 2):
            name1 = cleaned_results[i]
            name2 = cleaned_results[i+1]
    
    
    def __compare_names(name1: str, name2: str) -> float:
        return leven_ratio(name1.lower(), name2.lower())
        
        


def get_screen_bb() -> list[int]:
    res = pyautogui.resolution()
    return [0, 0, res.width, res.height]


def __parse_config(arg: str) -> dict:
    if not (exists(arg) or exists(join("configs", arg))):
        raise argparse.ArgumentError(f"Config file '{arg}' cannot be found!")
    
    if not exists(arg):
        arg = join("configs", arg)
    
    REQUIRED_CONFIG_KEYS = ["TIMER_REGION", "KILL_FEED_REGION", "IGNS"]
    OPTIONAL_CONFIG_KEYS = ["SCREENSHOT_RESIZE_X", "SCREENSHOT_PERIOD"]
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
