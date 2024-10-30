from abc import ABC, abstractmethod
from time import perf_counter
from typing import Callable, TypeVar, TypeAlias, Generic

from capture import Capture, Regions
from config import Config
from utils.enums import CaptureTimeType
from utils.timer import AtomicTimer


__all__ = [
    "Scheduler",
    "create_scheduler"
]


Handlers_t: TypeAlias = dict[str, Callable[[Regions], None]]

class Scheduler(ABC):
    def __init__(self, capture: Capture, handlers: Handlers_t) -> None:
        self._capture = capture
        self._handlers = handlers

    @abstractmethod
    def tick(self) -> bool:
        ...


class TimeScheduler(Scheduler):
    def __init__(self, config: Config,
                 capture: Capture,
                 handlers: Handlers_t) -> None:
        super(TimeScheduler, self).__init__(capture, handlers)

        self.__timer = AtomicTimer(capture.get_time)

        self.__periods = config.scheduler.model_dump()
        self.__handler_timers = { key: 0 for key in self.__periods }
    
    def tick(self) -> bool:
        regions = None

        dt = int(self.__timer.update() * 1000)
        for handle in self._handlers:
            self.__handler_timers[handle] += dt
            htimer = self.__handler_timers[handle]
            hperiod = self.__periods[handle]

            if htimer >= hperiod:
                if regions is None and (regions := self._capture.next()) is None:
                    return True

                self._handlers[handle](regions)
                self.__handler_timers[handle] -= hperiod

        return False


class FpsScheduler(Scheduler):
    def __init__(self, config: Config,
                 capture: Capture,
                 handlers: Handlers_t) -> None:
        super(FpsScheduler, self).__init__(capture, handlers)

        self.__periods = config.scheduler.model_dump()
        self.__handler_timers = self.__periods.copy()
    
    def tick(self) -> bool:
        regions = None

        dt = min(self.__handler_timers.values())
        for handle in self._handlers:
            self.__handler_timers[handle] -= dt
            htimer = self.__handler_timers[handle]
            
            if htimer <= 0:
                dt_ms = dt / 1000.0
                if regions is None and (regions := self._capture.next(dt_ms)) is None:
                    return True
                
                self._handlers[handle](regions)
                self.__handler_timers[handle] = self.__periods[handle]

        return False


def create_scheduler(config: Config, capture: Capture, handlers: Handlers_t) -> Scheduler:
    scheduler_map = {
        CaptureTimeType.TIME: TimeScheduler,
        CaptureTimeType.FPS: FpsScheduler,
    }

    return scheduler_map[capture.time_type](config, capture, handlers)
