from typing import Callable, Optional


__all__ = [
    "AtomicTimer"
]


class AtomicTimer:
    def __init__(self, time_callback: Callable[[], float], start: Optional[float] = None) -> None:
        self.__callback = time_callback

        if start is not None:
            self.__timer = start
        else:
            self.__timer = time_callback()

    @property
    def time(self) -> float:
        return self.__timer

    def update(self) -> float:
        old_time = self.__timer
        self.__timer = self.__callback()
        return self.__timer - old_time
