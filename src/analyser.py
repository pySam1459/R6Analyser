import argparse
import json
import cv2
import pyautogui
import numpy as np
import easyocr
from PIL import Image
from time import sleep, time
from os.path import join, exists


class Analyser:
    def __init__(self, args: argparse.Namespace):
        self.config = args.config
        self.gpu = not args.cpu
        
        self.running = False
        self.tdelta = 1.0
        
        self.reader = easyocr.Reader(['en'], gpu=self.gpu)
        self.count = 0
    
    def run(self):
        self.running = True
        self.timer = time()
        while self.running:
            if self.timer + self.tdelta > time(): continue
            self.timer = time()
            
            self.__read_timer()
            self.__read_feed()


    def __screenshot_process(self, screenshot: Image.Image) -> np.ndarray:
        # Convert the PIL Image to a NumPy array (RGB)
        img_np = np.array(screenshot)
        image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        new_width = int(image.shape[1] * 2)
        new_height = int(image.shape[0] * 2)
        new_dim = (new_width, new_height)

        # Resize the image
        resized_image = cv2.resize(image, new_dim, interpolation = cv2.INTER_LINEAR)
        return resized_image
    
    def __read_timer(self) -> None:
        screenshot = pyautogui.screenshot(region=self.config["TIMER"])
        img = self.__screenshot_process(screenshot)

        results = self.reader.readtext(img)
        for (bbox, text, prob) in results:
            print(text)

    def __read_feed(self) -> None:
        image = pyautogui.screenshot(region=self.config["KILL_FEED"])
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Find contours
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Assume that the largest contour is the gun symbol and remove it (if consistently in the same area)
        # # You can also filter by aspect ratio or area to refine what contours to remove
        # for cnt in contours:
        #     if cv2.contourArea(cnt) < 1000:  # Threshold area to distinguish between text and symbols
        #         continue
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

        # OCR with PyTesseract
        results = self.reader.readtext(gray)
        for (bbox, text, prob) in results:
            if prob > 0.5:
                print(f"{text}/{prob:.4f}", end="\t")
        print()
        


def get_screen_bb() -> list[int]:
    res = pyautogui.resolution()
    return [0, 0, res.width, res.height]


def __parse_config(arg: str) -> dict:
    if not (exists(arg) or exists(join("configs", arg))):
        raise argparse.ArgumentError(f"Config file '{arg}' cannot be found!")
    
    if not exists(arg):
        arg = join("configs", arg)
    
    REQUIRED_CONFIG_KEYS = ["TIMER", "KILL_FEED", "IGNS"]
    OPTIONAL_CONFIG_KEYS = ["SCREENSHOT_RESIZE_X", "SCREENSHOT_PERIOD"]
    DEFAULT_CONFIG_FILENAME = "defaults.json"
    with open(arg, "r", encoding="utf-8") as f_in:
        config = json.load(f_in)

    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise argparse.ArgumentError(f"Config file does not contain key '{key}'!")
    
    to_add = [key for key in OPTIONAL_CONFIG_KEYS if key not in config]
    if len(to_add) > 0:
        if not exists(DEFAULT_CONFIG_FILENAME):
            raise argparse.ArgumentError("'defaults.json' does not exist!")

        with open(DEFAULT_CONFIG_FILENAME, "r", encoding="utf-8") as f_in:
            default_config = json.load(f_in)
        
        for key in to_add:
            if key not in default_config:
                raise Exception("defaults.json has been modified, key '{key}' has been removed!")
            
            config[key] = default_config[key]
        

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
                        help="Determines how detailed the console output is")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Flag for only cpu execution, if your machine does not support gpu acceleration")

    args = parser.parse_args()
    if args.delay > 0:
        sleep(args.delay)
    
    Analyser(args).run()
