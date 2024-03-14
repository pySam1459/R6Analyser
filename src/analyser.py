import argparse
import json
import cv2
import pyautogui
import numpy as np
import easyocr
from PIL import Image
from time import sleep, time
from re import fullmatch
from os.path import join, exists


class Analyser:
    def __init__(self, args: argparse.Namespace):
        self.config = args.config
        self.gpu = not args.cpu
        
        self.running = False
        self.tdelta = 1.0
        
        # Initialize the EasyOCR Reader
        self.reader = easyocr.Reader(['en'], gpu=self.gpu)
    
    def run(self):
        self.running = True
        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time(): continue
            self.timer = time()
            
            self.__read_timer()
            self.__read_feed()

    
    def __read_timer(self) -> None:
        screenshot = pyautogui.screenshot(region=self.config["TIMER"])
        img = self.__ss_process(screenshot)
        results = self.reader.readtext(img)

        for (bbox, text, prob) in results:
            print(text)
    
    def __screenshot_process(self, screenshot: Image) -> np.ndarray:
        # Convert the PIL Image to a NumPy array (RGB)
        img_np = np.array(screenshot)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        


def get_screen_bb() -> list[int]:
    res = pyautogui.resolution()
    return [0, 0, res.width, res.height]


def __parse_config(arg: str) -> dict:
    if not (exists(arg) or exists(join("configs", arg))):
        raise argparse.ArgumentError(f"Config file '{arg}' cannot be found")
    
    if not exists(arg):
        arg = join("configs", arg)
    
    CONFIG_KEYS = ["TIMER", "KILL_FEED"]
    with open(arg, "r", encoding="utf-8") as f_in:
        config = json.load(f_in)

    for key in CONFIG_KEYS:
        if key not in config:
            raise argparse.ArgumentError(f"Config file does not contain key '{key}'")

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
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Flag for only cpu execution, if your machine does not support gpu acceleration")

    args = parser.parse_args()
    if args.delay > 0:
        sleep(args.delay)
    
    Analyser(args).run()
