from config import Config
from typing import Type, TypeVar

from utils.enums import CaptureMode

from .base import Capture
from .screenshot import ScreenshotCapture
from .videofile import VideoFileCapture
from .youtube import create_youtube_capture
from .twitch import TwitchStreamCapture
from .utils import RegionBBoxes, InPersonRegions, SpectatorRegions


__all__ = [
    "Capture",
    "create_capture",
    "RegionBBoxes",
    "InPersonRegions",
    "SpectatorRegions",
]


T = TypeVar('T', InPersonRegions, SpectatorRegions)
def create_capture(config: Config, region_model: Type[T]) -> Capture[T]:
    match config.capture.mode:
        case CaptureMode.SCREENSHOT:
            return ScreenshotCapture[T](config, region_model)
        case CaptureMode.VIDEOFILE:
            return VideoFileCapture[T](config, region_model)
        case CaptureMode.YOUTUBE:
            return create_youtube_capture(config, region_model)
        case CaptureMode.TWITCH:
            return TwitchStreamCapture[T](config, region_model)
