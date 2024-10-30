from config import Config

from .base import TimeCapture
from .regions import Regions


class TwitchStreamCapture(TimeCapture):
    def __init__(self, config: Config) -> None:
        super(TwitchStreamCapture, self).__init__(config)
        
        raise NotImplementedError("Twitch Steam Capturing is not implemented yet!")

    def next(self) -> Regions:
        ...

    def stop(self) -> None:
        ...
