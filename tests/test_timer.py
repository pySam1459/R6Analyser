import cv2
import numpy as np
import pytest
from pathlib import Path

from analyser.utils import get_timer_redperc
from utils.constants import BOMB_COUNTDOWN_RT, TIMER_LAST_SECONDS_RT


@pytest.fixture
def resource_path() -> Path:
    return Path(__file__).parent / "resources" / "timer"


def load_image(file_path: Path) -> np.ndarray:
    img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("image_filename", ["red_2-22.png"])
def test_get_timer_redperc(resource_path: Path, image_filename: str) -> None:
    img = load_image(resource_path / image_filename)

    red_perc = get_timer_redperc(img)
    assert red_perc >= TIMER_LAST_SECONDS_RT


@pytest.mark.parametrize("image_filename", ["bomb_1.png"])
def test_get_bomb_redperc(resource_path: Path, image_filename: str) -> None:
    img = load_image(resource_path / image_filename)

    red_perc = get_timer_redperc(img)
    assert red_perc >= BOMB_COUNTDOWN_RT
