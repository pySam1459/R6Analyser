import tkinter as tk
import cv2
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum
from decord import VideoReader
from PIL import Image
from PIL.ImageTk import PhotoImage
from pyautogui import screenshot
from pydantic.dataclasses import dataclass
from screeninfo import get_monitors
from typing import Any, Optional, TypeVar, Generic, cast

from config import RTConfig, RTRegionsCFG
from config.region_models import TimerRegion, KFLineRegion
from utils import BBox_t
from utils.cli import AnalyserArgs

from .utils import *


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


class FrameSkips:
    FRAME_SKIP = 0.1
    SMALL_SKIP = 1
    BIG_SKIP   = 10


T = TypeVar('T', TimerRegion, KFLineRegion)
class RegionTool(Generic[T], ABC):
    REGION_LIST = ["timer", "kf_line"]
    REGIONS     = {Keys.ONE: "timer", Keys.TWO: "kf_line"}
    REGION_MAP  = {"timer": TimerRegion, "kf_line": KFLineRegion}
    COLOURS     = {
        "active_drag": "yellow",
        "new_region": "orange",
        "selected": "red",
        "timer": "green",
        "kf_line": "blue"
    }

    def __init__(self, config: RTConfig) -> None:
        self.config = config
        self.running = True

        self.start = None
        self.active_drag: Optional[BBox_t] = None
        self.selected:    Optional[str]    = None

        regions, params   = self.__load_config(config)
        self.region_sels:   dict[str, T]      = regions
        self.region_params: dict[str, Any]    = params
        self.bbox_sels:     dict[str, BBox_t] = {}

        self.__photo_ref = None

    def __create_region_model(self, reg: str, rect: BBox_t, params: dict[str, Any]) -> T:
        model = cast(T, RegionTool.REGION_MAP[reg])
        return model.model_validate({reg: rect} | params)

    def __load_config(self, config: RTConfig) -> tuple[dict[str, T], dict[str, Any]]:
        if config.capture.regions is None:
            return ({}, {})

        region_config = config.capture.regions.model_dump(exclude_none=True)
        params = {reg: value
                     for reg, value in region_config.items()
                     if reg not in RegionTool.REGION_LIST}
        models = {reg: self.__create_region_model(reg, rect, params)
                     for reg, rect in region_config.items()
                     if reg in RegionTool.REGION_LIST}

        return models, params
    
    @property
    @abstractmethod
    def capture_rect(self) -> Rect_t:
        ...

    def __set_model(self, reg: str, rect: BBox_t) -> T:
        rm = self.__create_region_model(reg, rect, self.region_params)
        self.region_sels[reg] = rm
        return rm

    def init_window(self, width: int, height: int) -> None:
        self.root = tk.Tk()
        self.root.geometry(f"{width}x{height}+0+0")
        self.root.wm_title("Region Tool")

        self.canvas = tk.Canvas(self.root,
                                width=width,
                                height=height)
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

    def set_background(self, image: np.ndarray) -> None:
        img_ = Image.fromarray(image)

        self.__photo_ref = PhotoImage(img_)
        self.canvas.delete("background")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.__photo_ref, tags="background")
        self.render()

    def render(self) -> None:
        if not self.running:
            return

        for reg_key, bbox in self.bbox_sels.items():
            self.__draw_selection(cast(BBox_t, bbox), reg_key, False)

        for reg_key, region in self.region_sels.items():
            self.canvas.delete(reg_key)
            for model_attr, model_value in region.model_dump().items():
                dashed = model_attr not in RegionTool.COLOURS
                if isinstance(model_value, tuple):
                    self.__draw_selection(model_value, reg_key, dashed)
                elif isinstance(model_value, list):
                    for el in model_value:
                        self.__draw_selection(el, reg_key, dashed)

    def __draw_selection(self, abs_rect: BBox_t, tag: str, dashed: bool) -> None:
        left = abs_rect[0]-self.capture_rect[0]
        top  = abs_rect[1]-self.capture_rect[1]
        right = left + abs_rect[2]
        bottom = top + abs_rect[3]

        self.__draw_rect((left, top, right, bottom), tag, dashed)

    def __draw_rect(self, rect: BBox_t, tag: str, dashed = False) -> None:
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
                self.canvas.create_line(*line, dash=(5, 2), width=2, fill=colour, tags=tag)

    def __on_click(self, event: tk.Event) -> None:
        """Handle the initial click by saving the start coordinates."""
        self.start = Point(event.x, event.y)

    def __on_drag(self, event: tk.Event) -> None:
        """Handle the drag operation by updating the overlay with the region"""
        if self.start is None:
            return

        rect = (self.start.x, self.start.y, event.x, event.y)
        self.canvas.delete("active_drag")
        self.__draw_rect(rect, "active_drag")

    def __on_release(self, event: tk.Event) -> None:
        """Handle the release of the mouse button, close the program."""
        if self.start is None:
            return

        offset = Point(self.capture_rect[0], self.capture_rect[1])
        start  = self.start + offset
        end    = Point(event.x, event.y) + offset

        new_region_rect = start.rect(end)
        self.bbox_sels["new_region"] = new_region_rect
        self.selected = "new_region"

        self.start = None
        self.canvas.delete("active_drag")
        self.render()

    def __on_keypress(self, event: tk.Event) -> None:
        if event.keycode in RegionTool.REGIONS:
            self.__key_regions(event)
        elif event.keycode == Keys.DELETE:
            self.__key_delete()
        elif event.keycode == Keys.RETURN:
            self._on_return()
        elif event.keycode in Keys:
            self._on_arrows(event)

        self.render()

    def __key_regions(self, event: tk.Event) -> None:
        assert event.keycode in RegionTool.REGIONS

        reg = RegionTool.REGIONS[event.keycode]
        if self.selected == "new_region":
            self.__set_model(reg, self.bbox_sels.pop("new_region"))
            self.canvas.delete("new_region")
            self.selected = None
        elif reg in self.bbox_sels or reg in self.region_sels:
            self.selected = reg    

    def __key_delete(self) -> None:
        if self.selected is not None:
            if self.selected in self.bbox_sels:
                self.bbox_sels.pop(self.selected)
            elif self.selected in self.region_sels:
                self.region_sels.pop(self.selected)

            self.canvas.delete(self.selected)
            self.selected = None

    @abstractmethod
    def _on_arrows(self, event: tk.Event) -> None:
        ...

    def _on_return(self) -> None:
        print(f"Saved to: {self.config.config_path}")
        self._save_config()
        self.stop()

    def _save_config(self) -> None:
        region_data = {reg_key: getattr(self.region_sels[reg_key], reg_key)
                    for reg_key in self.region_sels}

        self.config.capture.regions = RTRegionsCFG.model_validate(region_data)
        with open(self.config.config_path, "w") as f_out:
            config_data = self.config.model_dump_json(indent=4, exclude_none=True)
            f_out.write(config_data)

    def stop(self, *_) -> None:
        self.running = False
        self.root.destroy()


class RTScreenShot(RegionTool):
    def __init__(self, args: AnalyserArgs, config: RTConfig) -> None:
        super(RTScreenShot, self).__init__(config)
        
        monitors = get_monitors()
        display = monitors[args.display-1]
        self._capture_rect = monitor_rect(display)
        ss_img = screenshot(region=self._capture_rect, allScreens=True)

        self.init_window(display.width, display.height)
        self.set_background(np.array(ss_img))
    
    @property
    def capture_rect(self) -> Rect_t:
        return self._capture_rect

    def _on_arrows(self, _: tk.Event) -> None:
        ...


@dataclass
class VideoDetails:
    max_frame: int
    frame_width: int
    frame_height: int
    fps: int


def on_arrows(frame_idx: int, vdets: VideoDetails, event: tk.Event) -> int:
    lr_frame_skip = max(int(round(FrameSkips.FRAME_SKIP * vdets.fps)), 1)
    lr_small_skip = FrameSkips.SMALL_SKIP * vdets.fps
    lr_big_skip   = FrameSkips.BIG_SKIP * vdets.fps
        ## TODO: can't hold down left/right, selection scaling

    if event.keycode == Keys.W and frame_idx - lr_frame_skip >= 0: ## up
        frame_idx -= lr_frame_skip
    if event.keycode == Keys.A and frame_idx - lr_small_skip >= 0: ## left
        frame_idx -= lr_small_skip
    if event.keycode == Keys.Q and frame_idx - lr_big_skip >= 0: ## up
        frame_idx -= lr_big_skip
    
    if event.keycode == Keys.S and frame_idx + lr_frame_skip < vdets.max_frame: ## down
        frame_idx += lr_frame_skip
    if event.keycode == Keys.D and frame_idx + lr_small_skip < vdets.max_frame: ## right
        frame_idx += lr_small_skip
    if event.keycode == Keys.E and frame_idx + lr_frame_skip < vdets.max_frame: ## up
        frame_idx += lr_big_skip

    return frame_idx


class RTVideoFile(RegionTool):
    def __init__(self, config: RTConfig) -> None:
        super(RTVideoFile, self).__init__(config)

        self.frame_idx = 0
        self.vr = self.new_video(self.config)
        self.image = self.__get_frame(self.frame_idx)

        assert self.image is not None, "Cannot get video frame"
        self._capture_rect = (0, 0, self.image.shape[1], self.image.shape[0])
        self.init_window(self.image.shape[1], self.image.shape[0])
        self.set_background(self.image)
    
    @property
    def capture_rect(self) -> Rect_t:
        return self._capture_rect

    def new_video(self, config: RTConfig) -> VideoReader:
        if config.capture.file is None:
            raise ValueError(f"Configuration capture file is None!")

        try:
            vr = VideoReader(str(config.capture.file), num_threads=1)
            h,w,*_ = vr[0].asnumpy().shape
            self.vdets = VideoDetails(
                max_frame=len(vr),
                frame_width=w,
                frame_height=h,
                fps=vr.get_avg_fps(),
            )
        except Exception as e:
            raise ValueError(f"An Error occurred when attempting to read from video file\n{e}")

        return vr
        
    def __get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx >= self.vdets.max_frame:
            return None

        return self.vr[frame_idx].asnumpy()

    def __set_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        image = self.__get_frame(frame_idx)
        if image is not None:
            self.set_background(image)
        return image

    def _on_arrows(self, event: tk.Event) -> None:
        self.frame_idx = on_arrows(self.frame_idx, self.vdets, event)
        self.image = self.__set_frame(self.frame_idx)


class RTYoutubeVideo(RegionTool):
    def __init__(self, config: RTConfig, cap: cv2.VideoCapture) -> None:
        super(RTYoutubeVideo, self).__init__(config)

        self.frame_idx = 0
        self.cap = cap
        self.vdets = self.get_details()
        self.image = self.__get_frame(self.frame_idx)

        assert self.image is not None, "Cannot get video frame"
        self._capture_rect = (0, 0, self.image.shape[1], self.image.shape[0])
        self.init_window(self.image.shape[1], self.image.shape[0])
        self.set_background(self.image)
    
    @property
    def capture_rect(self) -> Rect_t:
        return self._capture_rect

    def get_details(self) -> VideoDetails:
        ret, frame = self.cap.read()
        if ret is None:
            raise ValueError("Cannot read from youtube video capture")

        h,w,*_ = frame.shape
        return VideoDetails(
            max_frame=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            frame_width=w,
            frame_height=h,
            fps=int(round(self.cap.get(cv2.CAP_PROP_FPS))),
        )
        
    def __get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx >= self.vdets.max_frame:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __set_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        image = self.__get_frame(frame_idx)
        if image is not None:
            self.set_background(image)
        return image

    def _on_arrows(self, event: tk.Event) -> None:
        self.frame_idx = on_arrows(self.frame_idx, self.vdets, event)
        self.image = self.__set_frame(self.frame_idx)
