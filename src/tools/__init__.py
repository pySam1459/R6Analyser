import cv2

from args import AnalyserArgs
from capture.youtube import get_video_stream_details
from config import RTConfig
from utils.enums import CaptureMode

from .regiontool import RegionTool, RTScreenShot, RTVideoFile, RTYoutubeVideo
from .deps_check import check_dependencies


__all__ = [
    "create_regiontool",
    "check_dependencies"
]


def create_youtube_rt(config: RTConfig) -> RegionTool:
    assert config.capture.url is not None, "Invalid Config! YouTube URL is not provided! Please add YouTube URL to capture.url"

    video_stream_url = get_video_stream_details(config.capture.url)

    cap = cv2.VideoCapture(video_stream_url.url)
    if not cap.isOpened():
        raise ValueError(f"Video Capture Error: Unable to open video stream from {config.capture.url}")

    if video_stream_url.is_live:
        raise NotImplementedError(f"RegionTool does not support Youtube livestream yet")
    else:
        return RTYoutubeVideo(config, cap)


def create_regiontool(args: AnalyserArgs, config: RTConfig) -> RegionTool:
    mode = config.capture.mode
    match mode:
        case CaptureMode.SCREENSHOT:
            return RTScreenShot(args, config)
        case CaptureMode.VIDEOFILE:
            return RTVideoFile(config)
        case CaptureMode.YOUTUBE:
            return create_youtube_rt(config)
        case _:
            raise NotImplementedError(f"RegionTool does not support {mode} yet")
