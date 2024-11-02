import sys
from io import StringIO
from pydantic.dataclasses import dataclass
from re import match
from tqdm import tqdm
from typing import Optional, Self

from .constants import TIMESTAMP_PATTERN, NUMBER_PATTERN
from .enums import Team


@dataclass
class Scoreline:
    left: int
    right: int

    @property
    def scores(self) -> tuple[int, int]:
        return (self.left, self.right)

    @property
    def total(self) -> int:
        return self.left + self.right
    
    @property
    def max(self) -> int:
        if self.left < self.right:
            return self.right
        return self.left
    
    @property
    def diff(self) -> int:
        return abs(self.left-self.right)
    
    def inc(self, side: Team) -> Self:
        if side == Team.TEAM0:
            self.left += 1
        elif side == Team.TEAM1:
            self.right += 1
        return self


@dataclass
class Timestamp:
    """Helper dataclass to deal with the timer"""
    minutes: int
    seconds: int

    def __sub__(self, other: 'Timestamp') -> int:
        return self.to_int() - other.to_int()
    
    def __lt__(self, other: 'Timestamp') -> bool:
        return self.to_int() < other.to_int()

    def to_int(self) -> int:
        return self.minutes * 60 + self.seconds

    @staticmethod
    def from_int(num: int) -> 'Timestamp':
        m = int(num / 60)
        s = num - 60*m
        return Timestamp(minutes=m, seconds=s)

    @staticmethod
    def from_str(string: str) -> Optional['Timestamp']:
        if match(TIMESTAMP_PATTERN, string):
            colons = string.count(":")
            if colons == 0:
                return Timestamp(minutes=0, seconds=int(string))
            elif colons == 1:
                m,s = string.split(":")
                return Timestamp(minutes=int(m), seconds=int(s))
            elif colons == 2:
                h,m,s = string.split(":")
                return Timestamp(minutes=int(h)*60+int(m), seconds=int(s))
        elif match(NUMBER_PATTERN, string):
            return Timestamp.from_int(int(string))

        return None

    def __str__(self) -> str:
        is_neg = self.seconds < 0
        sec_str = str(self.seconds).zfill(is_neg + 2)
        return f"{self.minutes}:{sec_str}"

    __repr__ = __str__


class ProgressBar:
    def __init__(self, add_postfix = False) -> None:
        bar_format = "{desc}|{bar}"
        if add_postfix:
            bar_format += "|{postfix}"

        self.__tqdmbar = tqdm(total=180, bar_format=bar_format)

        self.__header  = ""
        self.__time    = ""
        self.__value   = "-"
    
    def set_prefix(self, refresh = False) -> None:
        self.__tqdmbar.set_description_str(f"{self.__header} | {self.__time} | {self.__value} ", refresh=refresh)

    def set_postfix(self, value: str) -> None:
        self.__tqdmbar.set_postfix_str(value, refresh=False)


    def set_total(self, value: int) -> None:
        self.__tqdmbar.total = value

    def set_header(self, nround: int, sl: Scoreline) -> None:
        self.__header = f"{nround}/{sl.left}:{sl.right}"
        self.set_prefix()

    def set_time(self, value: Timestamp | int) -> None:
        assert type(value) in [Timestamp, int], f"Invalid value type: {type(value)}"
        if isinstance(value, Timestamp):
            self.__time = str(value)
            value = value.to_int()
        elif isinstance(value, int):
            self.__time = str(Timestamp.from_int(value))

        self.__tqdmbar.n = value
        self.set_prefix()

    def set_desc(self, value: str) -> None:
        self.__value = value
        self.set_prefix()


    def reset(self) -> None:
        self.__tqdmbar.n = 180
        self.__tqdmbar.total = 180
        self.__value = "-"
        self.__time = "3:00"
        self.set_prefix(refresh=True)

    def refresh(self) -> None:
        self.__tqdmbar.refresh()

    def close(self) -> None:
        self.__tqdmbar.close()

    def new_round(self, sl: Scoreline) -> None:
        self.reset()
        self.set_header(sl.total + 1, sl)
        self.refresh()

    def bomb(self) -> None:
        self.set_time(45)
        self.set_total(45)
        self.refresh()


    @staticmethod
    def print(*prompt: object, sep: Optional[str] = " ", end: Optional[str] = "\n", flush: bool = False) -> None:
        """Replaces the builtin `print` function with one which works with the tqdm progress bar"""
        temp_out = StringIO()
        sys.stdout = temp_out
        print(*prompt, sep=sep, end=end, flush=flush)
        sys.stdout = sys.__stdout__
        tqdm.write(temp_out.getvalue(), end='')
