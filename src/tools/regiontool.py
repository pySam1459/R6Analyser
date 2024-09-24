import cv2
import tkinter as tk
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from PIL.ImageTk import PhotoImage
from pyautogui import screenshot, size as screen_size
from pydantic import BaseModel
from screeninfo import get_monitors, Monitor
from typing import Optional

from config import RTConfig
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


class RegionTool(ABC):
    REGIONS = {49: "timer", 50: "kf_line"}  ## keycode: region
    COLOURS = {"timer": "green", "kf_line": "blue"}
    ARROW_CODES = [37, 38, 39, 40] ## left,up,right,down

    def __init__(self, config: RTConfig) -> None:
        self.config = config

        self.size = screen_size()
        self.start_x = self.start_y = self.end_x = self.end_y = 0
        self.sf = 1

        self.selection: Optional[BBox_t] = None
        self.sels: dict[str, BBox_t] = config.capture.regions.__dict__

        self.__photoimage = None

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
        if self.selection is not None:
            self.__draw_rect(self.selection, "orange", "selection")

        for reg_key, rect in self.sels.items():
            self.__draw_rect(rect, RegionTool.COLOURS[reg_key], reg_key)
    
    def __draw_rect(self, abs_rect: BBox_t, colour: str, tags: str) -> None:
        ## TODO: dashed lines for projected regions - https://chatgpt.com/share/66f21307-551c-8004-a05a-1ddae93cd2ea
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
        self.selection = (tlx+self.display[0], tly+self.display[1], brx-tlx, bry-tly)
        self._draw_boxes()

    def __on_keypress(self, event: tk.Event) -> None:
        if event.keycode in RegionTool.REGIONS and self.selection is not None:
            self.sels[RegionTool.REGIONS[event.keycode]] = self.selection
            self.selection = None
            self._draw_boxes()
        
        elif event.keycode in [37, 38, 39, 40]:
            self._on_arrows(event)
            self._draw_boxes()

        elif event.keycode == 13: ## Return/Enter
            self._on_return()

    @abstractmethod
    def _on_arrows(self, event: tk.Event) -> None:
        ...

    def _on_return(self) -> None:
        print(f"Saved to: {self.config.config_path}")
        self._save_config()
        self.stop()

    def _save_config(self) -> None:
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


class VideoDetails(BaseModel):
    max_frame: int
    frame_width: int
    frame_height: int
    fps: int


class RTVideoFile(RegionTool):
    TIME_SKIP = 15    # seconds
    FRAME_SCROLL = 15 # frames

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

        video = cv2.VideoCapture(str(config.capture.file))
        self.vdets = VideoDetails(**{k: int(video.get(pid))
                                     for k,pid in RTVideoFile.VIDEO_PROPS.items()})
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
        lr_skip = RTVideoFile.TIME_SKIP * self.vdets.fps
        if event.keycode == 37 and self.frame_idx - lr_skip >= 0: ## left
            self.frame_idx -= lr_skip
        if event.keycode == 38: ## up
            ...
        if event.keycode == 39 and self.frame_idx + lr_skip: ## right ## TODO: can't hold down left/right, selection scaling 
            self.frame_idx += lr_skip
        if event.keycode == 40: ## down
            ...
        
        self.image = self.__set_frame(self.frame_idx)
    
    def stop(self, *_) -> None:
        if self.__video:
            self.__video.release()
        
        super().stop()
