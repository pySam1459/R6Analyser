from typing import Callable


__all__ = ["Timer"]


class Timer:
    def __init__(self, get_time_func: Callable[[], float], period: float) -> None:
        self._get_time = get_time_func
        self._time = 0
        self._period = period
    
    def update(self) -> float:
        old_time = self._time
        self._time = self._get_time()
        return self._time - old_time
    
    def step(self) -> bool:
        return self._period == 0 or self._time + self._period <= self._get_time()
