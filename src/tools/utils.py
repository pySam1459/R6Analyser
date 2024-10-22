from dataclasses import dataclass
from PIL import Image
from screeninfo import Monitor
from typing import TypeAlias


Rect_t: TypeAlias = tuple[int,int,int,int]

@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x+other.x, self.y+other.y)
    
    def rect(self, other: 'Point') -> Rect_t:
        minx   = min(self.x, other.x)
        miny   = min(self.y, other.y)
        width  = abs(self.x -other.x)
        height = abs(self.y -other.y)
        return (minx, miny, width, height)


def resize(image: Image.Image, dim: tuple[int,int]) -> tuple[Image.Image, float]:
    w,   h = image.size
    sw, sh = dim

    if sw == w and sh == h:
        return image, 1.0
    elif w/h == sw/sh:
        sf = w/h
    else:
        sf = min(sw/w, sh/h)
        dim = (int(w*sf), int(h*sf))
    
    return image.resize(dim), sf


def monitor_rect(monitor: Monitor) -> Rect_t:
    return (monitor.x, monitor.y, monitor.width, monitor.height)
