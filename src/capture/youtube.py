import cv2
import numpy as np
from dataclasses import dataclass
from shutil import rmtree
from decord import VideoReader
from yt_dlp import YoutubeDL
from typing import Optional, cast

from config import Config
from utils import Timestamp, ndefault, TIMESTAMP_ZERO
from utils.constants import MIN_YT_VIDEO_HEIGHT, DEFAULT_CHUNKS_DIR
from utils.cv import crop2bbox

from .chunk_loader import ChunkFetcher
from .regions import Regions
from .base import *


@dataclass
class _VideoStream:
    url:      str
    is_live:  bool
    duration: Optional[int]
    fps:      Optional[int]


class YoutubeLiveStreamCapture(TimeCapture):
    def __init__(self, config: Config, vs: _VideoStream):
        super(YoutubeLiveStreamCapture, self).__init__(config)

        self.cap = cv2.VideoCapture(vs.url)
        if not self.cap.isOpened():
            raise ValueError(f"Video Capture Error: Unable to open video stream from {config.capture.url}")

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


class YoutubeVideoCaptureStreamOnly(FpsCapture):
    def __init__(self, config: Config, vs: _VideoStream):
        super(YoutubeVideoCaptureStreamOnly, self).__init__(config)

        self.cap = cv2.VideoCapture(vs.url)
        if not self.cap.isOpened():
            raise ValueError(f"Video Capture Error: Unable to open video stream from {config.capture.url}")

        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.__frame_idx = self.__offset(config.capture.start)
    
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
        if frame_idx >= self.duration:
            raise ValueError("Capture start is >= video duration")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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


class YoutubeVideoCaptureChunkReader(FpsCapture):
    def __init__(self, config: Config, vs: _VideoStream):
        super(YoutubeVideoCaptureChunkReader, self).__init__(config)
        assert vs.fps is not None, "Cannot fetch video fps, try `capture.stream_only: true`"
        assert vs.duration, "Cannot fetch video duration, try `capture.stream_only: true`"

        self.__fps = vs.fps
        self.duration = vs.duration
        self.chunk_offset = self.__chunk_offset(config.capture.start)
        self.chunk_length = 0

        self.vr = None
        self.fetcher = ChunkFetcher(vs.url,
                                    vs.duration,
                                    config.capture.video_chunk_size,
                                    self.chunk_offset)
        
        self.__frame_idx = self.chunk_offset
        self.fetcher.prepare_first()
    
    @property
    def frame_idx(self) -> int:
        return self.__frame_idx

    @property
    def fps(self) -> float:
        return self.__fps
    
    def __chunk_offset(self, offset: Optional[Timestamp]) -> int:
        if offset is None:
            return 0

        coff = offset.to_int()
        if coff >= self.duration:
            raise ValueError("Capture start is >= total frames")

        return coff
 
    def __next_frame(self, frame_interval: int) -> np.ndarray | None:
        self.__frame_idx += frame_interval

        ## chunk index
        chidx = self.__frame_idx - self.chunk_offset
        if self.vr is None or chidx >= self.chunk_length:
            if self.vr is not None:
                ## update chunk_offset to latest chunk start time
                self.chunk_offset += self.chunk_length
                self.__frame_idx = (chidx - self.chunk_length) + self.chunk_offset  ## for synchronicity
            
            ## get next videochunk file path and re-init VR with new chunk
            next_path = self.fetcher.get_next_path()
            if next_path is None:
                return None

            self.vr = VideoReader(str(next_path), num_threads=1)
            self.chunk_length = len(self.vr)
            self.fetcher.prepare_next()

        chidx = self.__frame_idx - self.chunk_offset
        return self.vr[chidx].asnumpy()

    def next(self, dt: float, *_) -> Optional[Regions]:
        frame_interval = max(int(round(dt * self.__fps)), 1)
        frame = self.__next_frame(frame_interval)
        if frame is None:
            return None

        frame = crop2bbox(frame, self._bboxes.max_bounds)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._set_regions(frame)

    def stop(self) -> None:
        """cleanup chunks from file-tree"""
        self.fetcher.stop()
        rmtree(DEFAULT_CHUNKS_DIR)


def format_filter(format: dict[str, str]) -> bool:
    height = ndefault(cast(int, format.get("height", 0)), 0)
    return (
        format.get("ext") == "mp4" and
        format.get("protocol", "").startswith("http") and
        format.get("url") is not None and
        height >= MIN_YT_VIDEO_HEIGHT
    )


def get_video_stream_details(youtube_url: str) -> _VideoStream:
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
            video_url = cast(str, info_dict.get("url"))
            return _VideoStream(url=video_url, is_live=True, duration=None, fps=None)

        # Find the best available format
        formats: list[dict[str,str]] = info_dict.get("formats", [])
        vformat = next(filter(format_filter, formats), None)
        if vformat is None:
            raise ValueError("Could not retrieve a valid video URL")

        video_url = vformat["url"]
        duration = info_dict.get("duration")
        fps = info_dict.get("fps")
        return _VideoStream(url=video_url, is_live=False, duration=duration, fps=fps)


def create_youtube_capture(config: Config) -> Capture:
    assert config.capture.url is not None, "Invalid Config! YouTube URL is not provided! Please add YouTube URL to capture.url"

    vs = get_video_stream_details(config.capture.url)
    
    if vs.is_live:
        return YoutubeLiveStreamCapture(config, vs)
    elif config.capture.stream_only:
        return YoutubeVideoCaptureStreamOnly(config, vs)
    else:
        return YoutubeVideoCaptureChunkReader(config, vs)
