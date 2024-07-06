import argparse
import cv2
import tkinter as tk
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from PIL.ImageTk import PhotoImage
from pyautogui import screenshot, size as screen_size
from screeninfo import get_monitors, Monitor
from typing import Optional

from capture import CaptureMode
from utils import Config


__all__ = ["RegionTool"]


REGION_TOOL_INSTRUCTIONS = """
--- REGION TOOL --- PRESS ESCAPE TO QUIT ---
CONTROLS:
  Select Region\t\t\tClick, Drag, Un-Click
  Assign Region\t\t\tPress keys 1,2
    1. Timer (select entire region)
    2. First kill feed line (bottom line, account for large names)
  Save Region to Config\t\tPress Enter
    If multiple configs, Enter will switch to next config
    To go back\t\t\tPress Backspace

  For Video Files
    Change Frame\t\tLeft/Right arrow keys
"""


class RegionTool(ABC):
    REGIONS = {49: "TIMER", 50: "KF_LINE"}  ## keycode: region
    COLOURS = {"TIMER": "green", "KF_LINE": "blue"}
    ARROW_CODES = [37, 38, 39, 40] ## left,up,right,down

    def __init__(self) -> None:
        self.size = screen_size()

        self.start_x = self.start_y = self.end_x = self.end_y = 0
        self.sf = 1

        self.selection = None
    
    @staticmethod
    def new(args: argparse.Namespace) -> "RegionTool":
        cfg = args.config if type(args.config) == Config else args.config[0]
        mode = cfg.capture.mode
        match mode:
            case CaptureMode.SCREENSHOT:
                return RTScreenShot(args)
            case CaptureMode.VIDEOFILE:
                return RTVideoFile(args)
            case _:
                raise NotImplementedError(f"RegionTool does not support {mode} yet")

    def _set_config(self, config: Config) -> None:
        self.config = config
        self.sels = {v: self.config.capture.regions[v]
                     for v in RegionTool.REGIONS.values()
                     if v in self.config.capture.regions}

    def _init(self, display: list[int]) -> None:
        self.display = display

        self.root = tk.Tk()
        self.root.geometry(f"{self.size[0]}x{self.size[1]}+0+0")
        self.root.wm_title("Region Tool")

        self.canvas = tk.Canvas(self.root, width=self.size[0], height=self.size[1])
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind the mouse events
        self.canvas.bind("<Button-1>", self.__on_click)  # Left mouse button click
        self.canvas.bind("<B1-Motion>", self.__on_drag)  # Dragging with left mouse button held down
        self.canvas.bind("<ButtonRelease-1>", self.__on_release)  # Left mouse button release
        self.root.bind("<Key>", self.__on_keypress)
        self.root.bind("<Escape>", self.stop)

    def _set_image(self, image: np.ndarray) -> None:
        h,w,_ = image.shape
        sw,sh = self.size
        ar, sar = w/h, sw/sh
        
        img = Image.fromarray(image)
        if sw == w and sh == h:
            self.sf = 1
        elif ar == sar:
            self.sf = sw/w
            img = img.resize((sw, sh))
        else:
            ## to keep aspect ratio of original image, find scale-factor which fits to screensize
            self.sf = min(sw/w, sh/h)
            img = img.resize((int(w*self.sf), int(h*self.sf)))

        self.__photoimage = PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.__photoimage)
        self._draw_boxes()

    def _draw_boxes(self) -> None:
        if self.selection is not None:
            self.__draw_rect(self.selection, "orange", "TEMP")

        for reg_key, rect in self.sels.items():
            self.__draw_rect(rect, RegionTool.COLOURS[reg_key], reg_key)
    
    def __draw_rect(self, abs_rect: list[int], colour: str, tags: str) -> None:
        self.canvas.delete(tags)
        x, y = abs_rect[0]-self.display[0], abs_rect[1]-self.display[1]
        self.canvas.create_rectangle(
            x, y, x + abs_rect[2], y + abs_rect[3],
            outline=colour, width=2, tags=tags
        )

    def __on_click(self, event: tk.Event) -> None:
        """Handle the initial click by saving the start coordinates."""
        self.start_x, self.start_y = event.x, event.y

    def __on_drag(self, event: tk.Event) -> None:
        """Handle the drag operation by updating the overlay with the region"""
        self.end_x, self.end_y = event.x, event.y
        self.canvas.delete("DRAG")  # Remove the old region
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y,
                                     outline="red", width=2, tags="DRAG")

    def __on_release(self, event: tk.Event) -> None:
        """Handle the release of the mouse button, close the program."""
        self.end_x, self.end_y = event.x, event.y
        tlx, tly = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
        brx, bry = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
        self.selection = [tlx+self.display[0], tly+self.display[1], brx-tlx, bry-tly]
        self._draw_boxes()

    def __on_keypress(self, event: tk.Event) -> None:
        if event.keycode in RegionTool.REGIONS:
            self.sels[RegionTool.REGIONS[event.keycode]] = self.selection
            self.selection = None
        elif event.keycode == 13: ## RETURN
            self._on_return()
        elif event.keycode in [37, 38, 39, 40]:
            self._on_arrows(event)

        self._draw_boxes()
    
    @abstractmethod
    def _on_return(self) -> None:
        ...
    
    @abstractmethod
    def _on_arrows(self, event: tk.Event) -> None:
        ...
    
    def _save_config(self) -> None:
        self.config.save(self.config.cfg_file_path)

    def run(self) -> None:
        print(REGION_TOOL_INSTRUCTIONS)
        self.root.mainloop()

    def stop(self, *_) -> None:
        self.root.destroy()
        exit()


class RTScreenShot(RegionTool):
    def __init__(self, args: argparse.Namespace) -> None:
        super(RTScreenShot, self).__init__()
        monitors = get_monitors()
        self.monitor: Monitor = monitors[args.display]

        display = [self.monitor.x, self.monitor.y, self.monitor.width, self.monitor.height]
        self.image = np.array(screenshot(region=display, allScreens=True))

        self._set_config(args.config)
        self._init(display)
        self._set_image(self.image)
    
    def _on_return(self) -> None:
        for region, rect in self.sels.items():
            self.config.capture.regions[region] = rect ## might have to scale here

        print(f"SAVED UPDATED CONFIG TO {self.config.cfg_file_path}")
        self._save_config()
        self.stop()

    def _on_arrows(self, _: tk.Event) -> None:
        ...


class RTVideoFile(RegionTool):
    FRAME_SKIP = 15

    def __init__(self, args: argparse.Namespace) -> None:
        super(RTVideoFile, self).__init__()

        self.args = args
        display = [0, 0, self.size.width, self.size.height]
        self._init(display)

        self.__video = None
        self.is_multi = type(args.config) == list and len(args.config) > 1
        self.frame_idx, self.max_frame = 0, 0
        if self.is_multi:
            self.new_config(0)
        else:
            self._set_config(args.config)
            self.new_video(self.config)
    
    def new_config(self, idx: int) -> None:
        self.cfg_idx = idx
        self._set_config(self.args.config[self.cfg_idx])
        self.new_video(self.config)

    def new_video(self, config: Config) -> None:
        if self.__video:
            self.__video.release()

        self.__video = cv2.VideoCapture(config.capture.file)
        self.max_frame = int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.image = self.__get_frame(self.__video, self.frame_idx)
        self._set_image(self.image)
    
    def __get_frame(self, video: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    def _on_return(self) -> None:
        for region, rect in self.sels.items():
            self.config.capture.regions[region] = rect ## might have to scale here

        self._save_config()
        if (self.is_multi and self.cfg_idx+1 >= len(self.config)) or not self.is_multi:
            self.stop()

        self.new_config(self.cfg_idx + 1)
    
    def _on_arrows(self, event: tk.Event) -> None:
        if event.keycode == 37: ## left
            self.frame_idx = max(0, self.frame_idx-RTVideoFile.FRAME_SKIP)
        elif event.keycode == 38: ## up
            ...
        elif event.keycode == 39: ## right ## TODO: can't hold down left/right, selection scaling 
            self.frame_idx = min(self.max_frame, self.frame_idx+RTVideoFile.FRAME_SKIP)
        elif event.keycode == 40: ## down
            ...
        
        self._set_image(self.__get_frame(self.__video, self.frame_idx))
    
    def stop(self, *_) -> None:
        if self.__video:
            self.__video.release()
        
        super().stop()

if __name__ == "__main__":
    print("Please use the region tool from the cli\nExample: python ./run.py <Config> --region-tool")