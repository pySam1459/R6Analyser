import cv2
import tkinter as tk
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
from PIL import Image
from PIL.ImageTk import PhotoImage
from pyautogui import screenshot, size
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from screeninfo import get_monitors, Monitor
from typing import Optional, TypeVar, Sequence

from config import RTConfig, RTRegionsCFG
from config.region_models import TimerRegion, KFLineRegion
from utils import BBox_t
from utils.cli import AnalyserArgs
from utils.enums import CaptureMode


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


class Keys(IntEnum):
    RETURN = 13
    DELETE = 46

    LEFT = 37
    RIGHT = 39
    UP = 38
    DOWN = 40

    W = 87
    A = 65
    S = 83
    D = 68
    Q = 81
    E = 69

    ONE = 49
    TWO = 50
    THREE = 51
    FOUR = 52


T = TypeVar('T', bound=BaseModel)
class RegionTool(ABC):
    REGIONS = {Keys.ONE: "timer", Keys.TWO: "kf_line"}
    COLOURS = {
        "active_drag": "yellow",
        "new_region": "orange",
        "selected": "red",
        "timer": "green",
        "kf_line": "blue"
    }
    REGION_MAP: dict[str, T] = {"timer": TimerRegion, "kf_line": KFLineRegion}

    def __init__(self, config: RTConfig) -> None:
        self.config = config

        self.size = size()
        self.start_x = self.start_y = self.end_x = self.end_y = 0
        self.sf = 1

        self.active_drag: Optional[BBox_t] = None
        self.selected: Optional[str] = None
        self.sels: dict[str, T] = self.__get_sels(config)

        self.__photoimage = None
    
    def __get_sels(self, config: RTConfig) -> dict[str, T]:
        if config.capture.regions is not None:
            regions = config.capture.regions.model_dump(exclude_none=True)
            return {reg: RegionTool.REGION_MAP[reg].model_validate({reg: rect})
                    for reg, rect in regions.items()}
        return {}

    @staticmethod
    def new(args: AnalyserArgs, config: RTConfig) -> "RegionTool":
        mode = config.capture.mode
        match mode:
            case CaptureMode.SCREENSHOT:
                return RTScreenShot(args, config)
            case CaptureMode.VIDEOFILE:
                return RTVideoFile(args, config)
            case _:
                raise NotImplementedError(f"RegionTool does not support {mode} yet")

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
    
    def run(self) -> None:
        print(REGION_TOOL_INSTRUCTIONS)
        self.root.mainloop()

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
        for reg_key, region_model in self.sels.items():
            self.canvas.delete(reg_key)
            for model_attr, model_value in region_model.model_dump().items():
                if isinstance(model_value, tuple):
                    self.__draw_selection(model_value, reg_key, reg_key != model_attr)
                elif isinstance(model_value, list):
                    for el in model_value:
                        self.__draw_selection(el, reg_key, reg_key != model_attr)

    def __draw_selection(self, abs_rect: BBox_t, tag: str, dashed: bool) -> None:
        x, y = abs_rect[0]-self.display[0], abs_rect[1]-self.display[1]
        rect = (x, y, x + abs_rect[2], y + abs_rect[3])
        self.__draw_rect(rect, tag, dashed)

    def __draw_rect(self, rect: BBox_t, tag: str, dashed = True) -> None:
        if tag != self.selected:
            colour = RegionTool.COLOURS.get(tag, "pink")
        else:
            colour = RegionTool.COLOURS["selected"]

        if not dashed:
            self.canvas.create_rectangle(rect, outline=colour, width=2, tags=tag)
        else:
            x1,y1,x2,y2 = rect
            lines = [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]
            for line in lines:
                self.canvas.create_line(*line, dash=(5, 2), width=4, fill=colour, tags=tag)

    def __on_click(self, event: tk.Event) -> None:
        """Handle the initial click by saving the start coordinates."""
        self.start_x, self.start_y = event.x, event.y

    def __on_drag(self, event: tk.Event) -> None:
        """Handle the drag operation by updating the overlay with the region"""
        self.end_x, self.end_y = event.x, event.y
        rect = [self.start_x, self.start_y, self.end_x, self.end_y]
        self.canvas.delete("active_drag")
        self.__draw_rect(rect, "active_drag")

    def __on_release(self, event: tk.Event) -> None:
        """Handle the release of the mouse button, close the program."""
        self.end_x, self.end_y = event.x, event.y
        tlx, tly = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
        brx, bry = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
        self.sels["new_region"] = (tlx+self.display[0], tly+self.display[1], brx-tlx, bry-tly)
        self.selected = "new_region"
        self._draw_boxes()

    def __on_keypress(self, event: tk.Event) -> None:
        if event.keycode in RegionTool.REGIONS:
            reg = RegionTool.REGIONS[event.keycode]
            if self.selected == "new_region":
                self.sels[reg] = self.sels.pop("new_region")
                self.selected = None
            elif self.selected is None and reg in self.sels:
                self.selected = reg
        
        elif event.keycode == Keys.DELETE:
            if self.selected is not None:
                self.sels.pop(self.selected)
                self.selected = None

        elif event.keycode == Keys.RETURN:
            self._on_return()
        elif event.keycode in Keys:
            self._on_arrows(event)

        self._draw_boxes()

    @abstractmethod
    def _on_arrows(self, event: tk.Event) -> None:
        ...

    def _on_return(self) -> None:
        print(f"Saved to: {self.config.config_path}")
        self._save_config()
        self.stop()

    def _save_config(self) -> None:
        self.config.capture.regions = RTRegionsCFG.model_validate(self.sels)
        with open(self.config.config_path, "w") as f_out:
            f_out.write(self.config.model_dump_json(indent=4, exclude_none=True))

    def stop(self, *_) -> None:
        self.root.destroy()


class RTScreenShot(RegionTool):
    def __init__(self, args: AnalyserArgs, config: RTConfig) -> None:
        super(RTScreenShot, self).__init__(config)
        monitors = get_monitors()
        self.monitor: Monitor = monitors[args.display-1]

        display = [self.monitor.x, self.monitor.y, self.monitor.width, self.monitor.height]
        self.image = np.array(screenshot(region=display, allScreens=True)) # type: ignore

        self._init(display)
        self._set_image(self.image)

    def _on_arrows(self, _: tk.Event) -> None:
        ...


@dataclass
class VideoDetails:
    max_frame: int
    frame_width: int
    frame_height: int
    fps: int


class RTVideoFile(RegionTool):
    FRAME_SKIP = 0.1
    SMALL_SKIP = 1
    BIG_SKIP = 10

    VIDEO_PROPS = {
        "max_frame": cv2.CAP_PROP_FRAME_COUNT,
        "frame_width": cv2.CAP_PROP_FRAME_WIDTH,
        "frame_height": cv2.CAP_PROP_FRAME_HEIGHT,
        "fps": cv2.CAP_PROP_FPS
    }

    def __init__(self, args: AnalyserArgs, config: RTConfig) -> None:
        super(RTVideoFile, self).__init__(config)    
        self.args = args

        display = [0, 0, self.size.width, self.size.height]
        self._init(display)

        self.frame_idx = 0
        self.__video = self.new_video(self.config)
        self.image = self.__set_frame(self.frame_idx)

    def new_video(self, config: RTConfig) -> cv2.VideoCapture:
        if config.capture.file is None:
            raise ValueError(f"Configuration capture file is None!")

        try:
            video = cv2.VideoCapture(str(config.capture.file))
            self.vdets = VideoDetails(**{k: int(video.get(pid))
                                        for k,pid in RTVideoFile.VIDEO_PROPS.items()})
        except Exception as e:
            raise ValueError(f"An Error occurred when attempting to read from video file\n{e}")

        return video
        
    def __get_frame(self, video: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __set_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        image = self.__get_frame(self.__video, frame_idx)
        if image is not None:
            self._set_image(image)
        return image

    def _on_arrows(self, event: tk.Event) -> None:
        lr_frame_skip = max(int(round(RTVideoFile.FRAME_SKIP * self.vdets.fps)), 1)
        lr_small_skip = RTVideoFile.SMALL_SKIP * self.vdets.fps
        lr_big_skip = RTVideoFile.BIG_SKIP * self.vdets.fps
         ## TODO: can't hold down left/right, selection scaling 
        if event.keycode == Keys.W and self.frame_idx - lr_frame_skip >= 0: ## up
            self.frame_idx -= lr_frame_skip
        if event.keycode == Keys.A and self.frame_idx - lr_small_skip >= 0: ## left
            self.frame_idx -= lr_small_skip
        if event.keycode == Keys.Q and self.frame_idx - lr_big_skip >= 0: ## up
            self.frame_idx -= lr_big_skip
        
        if event.keycode == Keys.S and self.frame_idx + lr_frame_skip < self.vdets.max_frame: ## down
            self.frame_idx += lr_frame_skip
        if event.keycode == Keys.D and self.frame_idx + lr_small_skip < self.vdets.max_frame: ## right
            self.frame_idx += lr_small_skip
        if event.keycode == Keys.E and self.frame_idx + lr_frame_skip < self.vdets.max_frame: ## up
            self.frame_idx += lr_big_skip
        
        self.image = self.__set_frame(self.frame_idx)
    
    def stop(self, *_) -> None:
        if self.__video:
            self.__video.release()
        
        super().stop()
