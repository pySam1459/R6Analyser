import cv2
import numpy as np
from dataclasses import dataclass
from yt_dlp import YoutubeDL
from typing import Type, TypeVar, Optional, cast

from config import Config
from utils import Timestamp, ndefault

from .utils import InPersonRegions, SpectatorRegions, crop2bbox
from .base import *


YDL_OPTIONS = {
    'format': 'bestvideo[ext=mp4]/mp4',
    'quiet': True,
    'no_warnings': True,
    'skip_download': True,
}


T = TypeVar('T', InPersonRegions, SpectatorRegions)

class YoutubeVideoCapture(FpsCapture[T]):
    def __init__(self, config: Config, region_model: Type[T], cap: cv2.VideoCapture):
        super(YoutubeVideoCapture, self).__init__(config, region_model)
        self.cap = cap

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_idx = self.__offset(config.capture.start)
    
    def __offset(self, offset: Optional[Timestamp]) -> int:
        if offset is None:
            return 0

        frame_idx = int(round(self.fps * offset.to_int()))
        if frame_idx >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            raise ValueError("Capture offset is >= total frames")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return frame_idx

    def __next_frame(self, frame_interval: int, jump: bool) -> np.ndarray | None:
        self.frame_idx += frame_interval

        if jump:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            frame_interval = 1

        for _ in range(frame_interval):
            ret, frame = self.cap.read()
            if not ret:
                return None

        return frame

    def next(self, dt: float, jump = False) -> Optional[T]:
        frame_interval = max(int(round(dt * self.fps)), 1)
        frame = self.__next_frame(frame_interval, jump)
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._set_regions(frame)

    def stop(self) -> None:
        self.cap.release()


class YoutubeStreamCapture(TimeCapture[T]):
    def __init__(self, config: Config, region_model: Type[T], cap: cv2.VideoCapture):
        super(YoutubeStreamCapture, self).__init__(config, region_model)
        self.cap = cap

    def __next_frame(self) -> Optional[np.ndarray]:
        latest_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, latest_frame - 1)
        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def next(self, *_, **__) -> Optional[T]:
        frame = self.__next_frame()
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._set_regions(frame)

    def stop(self) -> None:
        self.cap.release()


@dataclass
class _VideoStream:
    url: str
    is_live: bool


def get_video_stream_details(youtube_url: str) -> _VideoStream:
    with YoutubeDL(YDL_OPTIONS) as ydl:
        try:
            info_dict = ydl.extract_info(youtube_url, download=False)
        except Exception as e:
            raise ValueError(f"Error extracting information from YouTube URL: {e}")

        if not info_dict:
            raise ValueError(f"Error extracting information from YouTube URL: No info fetched")

        # For live streams, the 'url' field contains the stream URL
        if info_dict.get('is_live'):
            video_url = cast(str, info_dict.get('url'))
            return _VideoStream(url=video_url, is_live=True)
        else:
            # Find the best available format
            formats: list[dict[str,str]] = info_dict.get('formats', [])
            for f in formats:
                height = ndefault(f.get("height", 0), 0)
                if (f.get('ext') == 'mp4' and
                    f.get('protocol', "").startswith('http') and
                    height >= 1080):
                    video_url = cast(str, f.get('url'))
                    return _VideoStream(url=video_url, is_live=False)
            else:
                raise ValueError("Could not retrieve a valid video URL")


def create_youtube_capture(config: Config, region_model: Type[T]) -> YoutubeVideoCapture | YoutubeStreamCapture:
    assert config.capture.url is not None, "Invalid Config! YouTube URL is not provided! Please add YouTube URL to capture.url"

    video_stream_url = get_video_stream_details(config.capture.url)

    cap = cv2.VideoCapture(video_stream_url.url)
    if not cap.isOpened():
        raise ValueError(f"Video Capture Error: Unable to open video stream from {config.capture.url}")
    
    if video_stream_url.is_live:
        return YoutubeStreamCapture(config, region_model, cap)
    else:
        return YoutubeVideoCapture(config, region_model, cap)
