import cv2
import pytest
from os import listdir
from pathlib import Path
from Levenshtein import ratio

from assets import Assets
from ocr import OCREngine
from settings import Settings, create_settings


res_path = Path(__file__).parent / "resources"
test_files_parent = res_path / "kflines"
test_files = [test_files_parent / file
              for file in listdir(test_files_parent)
              if file.endswith((".png", ".jpg"))]

@pytest.fixture
def settings() -> Settings:
    settings_path = res_path / "settings" / "ocr_engine_settings.json"
    return create_settings(settings_path)

@pytest.fixture
def assets() -> Assets:
    return Assets()


@pytest.mark.parametrize("test_file", test_files, ids=str)
def test_ocr_engine(test_file: Path, settings: Settings, assets: Assets) -> None:
    image = cv2.imread(str(test_file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    assets.resize_height("headshot", image.shape[0])
    engine = OCREngine(settings, assets)

    ocr_line_res = engine.read_kfline(image)
    assert ocr_line_res is not None, "OCRLineResult is None"

    parts = test_file.stem.split(" ")
    if len(parts) == 2:
        headshot = False
        left, right = parts
    elif len(parts) == 3 and parts[1] == "X":
        headshot = True
        left, right = parts[0], parts[2]
    else:
        raise ValueError(f"Invalid pytest configuration | file {test_file} is incorrectly named")

    if left == "":
        assert ocr_line_res.left is None, f"if left='', ocr_left is {ocr_line_res.left}"
    else:
        assert ocr_line_res.left is not None, f"ocr left should be defined"
        rat_left = ratio(ocr_line_res.left, left)
        assert rat_left > 0.8, f"ratio: {rat_left}, left value: {ocr_line_res.left}"
    
    rat_right = ratio(ocr_line_res.right, right)
    assert rat_right > 0.8, f"ratio: {rat_right}, right value: {ocr_line_res.right}"

    assert ocr_line_res.headshot == headshot, f"headshot: {ocr_line_res.headshot}"