from args import AnalyserArgs
from config import Config
from utils.enums import CaptureMode

from .base import Capture
from .screenshot import ScreenshotCapture
from .videofile import VideoFileCapture
from .youtube import create_youtube_capture
from .twitch import TwitchStreamCapture


__all__ = [
    "Capture",
    "create_capture"
]


def create_capture(args: AnalyserArgs, config: Config) -> Capture:
    match config.capture.mode:
        case CaptureMode.SCREENSHOT:
            return ScreenshotCapture(config)
        case CaptureMode.VIDEOFILE:
            return VideoFileCapture(args, config)
        case CaptureMode.YOUTUBE:
            return create_youtube_capture(args, config)
        case CaptureMode.TWITCH:
            return TwitchStreamCapture(config)
