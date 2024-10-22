from typing import TypeVar, Type

from config import Config

from .base import TimeCapture
from .utils import InPersonRegions, SpectatorRegions


T = TypeVar('T', InPersonRegions, SpectatorRegions)
class TwitchStreamCapture(TimeCapture[T]):
    def __init__(self, config: Config, region_model: Type[T]) -> None:
        super(TwitchStreamCapture, self).__init__(config, region_model)
        
        raise NotImplementedError("Twitch Steam Capturing is not implemented yet!")

    def next(self) -> T:
        ...

    def stop(self) -> None:
        ...
