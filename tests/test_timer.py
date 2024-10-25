import cv2
import numpy as np
import pytest
from pathlib import Path
from os import listdir

from ocr.utils import get_timer_redperc
from utils.constants import BOMB_COUNTDOWN_RT, TIMER_LAST_SECONDS_RT


base = Path(__file__).parent / "resources" / "timer"

def get_all_prefixed(prefix: str) -> list[str]:
    return [file for file in listdir(base) if file.startswith(prefix)]


def load_image(file_path: Path) -> np.ndarray:
    img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("image_filename", get_all_prefixed("red"))
def test_get_timer_redperc(image_filename: str) -> None:
    img = load_image(base / image_filename)

    red_perc = get_timer_redperc(img)
    assert red_perc >= TIMER_LAST_SECONDS_RT

@pytest.mark.parametrize("image_filename", get_all_prefixed("bomb"))
def test_get_bomb_redperc(image_filename: str) -> None:
    img = load_image(base / image_filename)

    red_perc = get_timer_redperc(img)
    assert red_perc >= BOMB_COUNTDOWN_RT
