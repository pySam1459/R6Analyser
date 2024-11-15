import cv2
import numpy as np
from dataclasses import dataclass
from yt_dlp import YoutubeDL
from typing import Optional, Any

import settings
from args import AnalyserArgs
from config import Config
from utils import Timestamp
from utils.cv import crop2bbox

from .regions import Regions
from .base import *


class YoutubeVideoCapture(FpsCapture):
    def __init__(self, args: AnalyserArgs, config: Config, cap: cv2.VideoCapture):
        super(YoutubeVideoCapture, self).__init__(args, config)
        self.cap = cap

        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.__frame_idx = self._get_start()
        if self.__frame_idx >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError("Start time > video duration")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.__frame_idx)
    
    @property
    def frame_idx(self) -> int:
        return self.__frame_idx

    @property
    def fps(self) -> float:
        return self.__fps
    
    def __offset(self, offset: Optional[Timestamp]) -> int:
        if offset is None:
            return 0

        frame_idx = int(round(self.fps * offset.to_int()))
        
        return frame_idx

    def __next_frame(self, frame_interval: int, jump: bool) -> np.ndarray | None:
        self.__frame_idx += frame_interval

        if jump:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.__frame_idx)
            frame_interval = 1

        for _ in range(frame_interval):
            ret, frame = self.cap.read()
            if not ret:
                return None

        return frame

    def next(self, dt: float, jump = False) -> Optional[Regions]:
        frame_interval = max(int(round(dt * self.__fps)), 1)
        frame = self.__next_frame(frame_interval, jump)
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._set_regions(frame)

    def stop(self) -> None:
        self.cap.release()


class YoutubeStreamCapture(TimeCapture):
    def __init__(self, config: Config, cap: cv2.VideoCapture):
        super(YoutubeStreamCapture, self).__init__(config)
        self.cap = cap

    def __next_frame(self) -> Optional[np.ndarray]:
        latest_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, latest_frame - 1)
        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def next(self, *_, **__) -> Optional[Regions]:
        frame = self.__next_frame()
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._set_regions(frame)

    def stop(self) -> None:
        self.cap.release()


@dataclass
class VideoDetails:
    url: str
    is_live: bool


def _get_live_details(info_dict: dict[str, Any]) -> VideoDetails:
    video_url = info_dict.get("url")
    if video_url is None or not isinstance(video_url, str):
        raise ValueError("Error extracting information from YouTube URL: Cannot fetch stream url")

    return VideoDetails(url=video_url, is_live=True)



def _format_filter(vformat: dict[str,str]) -> bool:
    height = int(vformat.get("height", 0))
    return (
        vformat.get("ext") == "mp4" and
        vformat.get("protocol", "").startswith("http") and
        height >= settings.SETTINGS.min_video_resolution
    )


def _get_video_details(info_dict: dict[str, Any]) -> VideoDetails:
    # Find the best available format
    formats: list[dict[str,str]] = info_dict.get("formats", [])
    for f in filter(_format_filter, formats):
        video_url = f.get("url")
        if video_url is None:
            continue

        return VideoDetails(url=video_url, is_live=False)
    else:
        raise ValueError("Could not retrieve a valid video URL")


def get_video_stream_details(youtube_url: str) -> VideoDetails:
    ydl_opts = {
        "format": "bestvideo[ext=mp4]/mp4",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(youtube_url, download=False)
        except Exception as e:
            raise ValueError(f"Error extracting information from YouTube URL: {e}")

        if not info_dict:
            raise ValueError(f"Error extracting information from YouTube URL: No info fetched")

        # For live streams, the "url" field contains the stream URL
        if info_dict.get("is_live"):
            return _get_live_details(info_dict)
        else:
            return _get_video_details(info_dict)


def create_youtube_capture(args: AnalyserArgs, config: Config) -> YoutubeVideoCapture | YoutubeStreamCapture:
    assert config.capture.url is not None, "Invalid Config! YouTube URL is not provided! Please add YouTube URL to capture.url"

    video_stream_url = get_video_stream_details(config.capture.url)

    cap = cv2.VideoCapture(video_stream_url.url)
    if not cap.isOpened():
        raise ValueError(f"Video Capture Error: Unable to open video stream from {config.capture.url}")
    
    if video_stream_url.is_live:
        return YoutubeStreamCapture(config, cap)
    else:
        return YoutubeVideoCapture(args, config, cap)
