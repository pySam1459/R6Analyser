from config import RTConfig
from utils.cli import AnalyserArgs
from utils.enums import CaptureMode

from .regiontool import RegionTool, RTScreenShot, RTVideoFile
from .deps_check import check_dependencies


__all__ = [
    "create_regiontool",
    "check_dependencies"
]


def create_regiontool(args: AnalyserArgs, config: RTConfig) -> RegionTool:
    mode = config.capture.mode
    match mode:
        case CaptureMode.SCREENSHOT:
            return RTScreenShot(args, config)
        case CaptureMode.VIDEOFILE:
            return RTVideoFile(config)
        case _:
            raise NotImplementedError(f"RegionTool does not support {mode} yet")
