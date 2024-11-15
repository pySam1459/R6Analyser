import cv2
import numpy as np
import pytest
from os import listdir, makedirs
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from Levenshtein import ratio
from PIL import Image
from typing import Sequence

import settings
from assets import Assets
from ocr import OCREngine, OCRLineResult
from ocr.utils import HSVColourRange, get_hsv_range
from params import OCRParams
from utils import load_json



def load_test_files(parent: Path) -> list[Path]:
    return [parent / file
            for file in listdir(parent)
            if file.endswith((".png", ".jpg"))]


def load_rgb(file: Path) -> np.ndarray:
    bgr_img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


project_base = Path(__file__).parent.parent
res_path = project_base / "tests" / "resources"
out_path = project_base / "tests" / "out" / "ocrengine"

sl_test_files_parent = res_path / "scoreline"
sl_test_files = load_test_files(sl_test_files_parent)

kflines_test_files_parent = res_path / "kfline"
kflines_test_files = load_test_files(kflines_test_files_parent)
kfline_colours = load_json(kflines_test_files_parent / "colours.json")

timer_test_files_parent = res_path / "timer"
timer_test_files = load_test_files(timer_test_files_parent)


@pytest.fixture(scope="session", autouse=True)
def setup_test_dir():
    ## prep out files
    if out_path.exists():
        rmtree(out_path)
    makedirs(out_path, exist_ok=True)


settings.SETTINGS = settings.create_settings(res_path / "settings" / "ocr_engine_settings.json")


@pytest.fixture
def assets() -> Assets:
    assets_path = Path("assets")
    return Assets(assets_path)

@pytest.fixture
def ocr_params() -> OCRParams:
    defaults = load_json(project_base / "settings" / "defaults.json")
    assert "ocr" in defaults

    ocr_params = defaults["ocr"]
    return OCRParams(**ocr_params)


def get_ids(test_file: Path) -> str:
    return test_file.stem


def get_out_dir(out_path: Path, test_file: Path) -> Path:
    test_case_out_path = out_path / test_file.stem.replace(".", "")
    makedirs(test_case_out_path, exist_ok=True)
    return test_case_out_path


# --- SCORELINE ---
def write_score_out_image(test_file: Path, image: Image.Image) -> None:
    score_out_path = get_out_dir(out_path / "scoreline", test_file)
    image.save(score_out_path / "threshold.jpg")


@pytest.mark.parametrize("test_file", sl_test_files, ids=get_ids)
def test_ocr_engine_scoreline(test_file: Path,
                              ocr_params: OCRParams,
                              assets: Assets) -> None:
    image = load_rgb(test_file)

    engine = OCREngine(ocr_params, settings, assets)  # type: ignore
    engine._debug_vars = {"th_path": out_path / "scoreline" / test_file.name}
    makedirs(out_path / "scoreline", exist_ok=True)

    score = engine.read_score(image)

    expected_text, _ = test_file.stem.split("_", maxsplit=1)

    if expected_text.lower() == "none":
        assert score is None, "Score is not None"
    else:
        assert score is not None, f"None was returned instead of: {expected_text}"
        assert score == expected_text, f"Score: {score} does not match expected text: {expected_text}"
    
    engine.stop()


# --- KFLINE ---
def write_kflines_out_images(test_file: Path, olr: OCRLineResult):
    kfline_out_path = get_out_dir(out_path / "kfline", test_file)
    if olr.left_image is not None:
        li = cv2.cvtColor(olr.left_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(kfline_out_path / "left.jpg"), li)
    if olr.middle_image is not None:
        mi = cv2.cvtColor(olr.middle_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(kfline_out_path / "middle.jpg"), mi)
    if olr.right_image is not None:
        ri = cv2.cvtColor(olr.right_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(kfline_out_path / "right.jpg"), ri)


def get_colours(test_file: Path, ocr_params: OCRParams) -> Sequence[HSVColourRange]:
    stds = (ocr_params.hue_std, ocr_params.sat_std)
    if test_file.stem.startswith("none"):
        colours = ((127, 127), (127, 127))
        return (get_hsv_range(colours[0], stds), get_hsv_range(colours[1], stds))

    else:
        assert test_file.stem in kfline_colours, f"You haven't added the colours of test {test_file.stem} to colours.json"
        colours = kfline_colours[test_file.stem]
        return (get_hsv_range(colours[0], stds), get_hsv_range(colours[1], stds))


@pytest.mark.parametrize("test_file", kflines_test_files, ids=get_ids)
def test_ocr_engine_kfline(test_file: Path,
                           ocr_params: OCRParams,
                           assets: Assets) -> None:
    image = load_rgb(test_file)
    colours = get_colours(test_file, ocr_params)

    engine = OCREngine(ocr_params, assets)
    engine._set_colours(colours[0], colours[1])

    result = engine.read_kfline(image)

    if result is not None:
        write_kflines_out_images(test_file, result)

    stem = test_file.stem
    if stem.startswith("none"):
        assert result is None, "OCRLineResult must be None"
        return

    assert result is not None, "OCRLineResult is None"

    parts = stem.strip("# ").split(" ")

    if stem.startswith("#"): ## suicide
        stem = stem[1:]
        parts.insert(0, "")
    
    if len(parts) == 2:
        headshot = False
        left, right = parts
    elif len(parts) == 3 and parts[1] == "X":
        headshot = True
        left, right = parts[0], parts[2]
    else:
        raise ValueError(f"Invalid pytest configuration | file {test_file} is incorrectly named")

    MIN_RATIO = 0.7

    if left == "":
        assert result.left is None, f"if left='', ocr_left is {result.left}"
    else:
        assert result.left is not None, f"ocr left should be defined"
        rat_left = ratio(result.left, left)
        assert rat_left >= MIN_RATIO, f"ratio: {rat_left}, left value: {result.left}"
    
    rat_right = ratio(result.right, right)
    assert rat_right >= MIN_RATIO, f"ratio: {rat_right}, right value: {result.right}"

    assert result.headshot == headshot, f"headshot: {result.headshot}"
    engine.stop()


# --- TIMER ---
@pytest.mark.parametrize("test_file", timer_test_files, ids=get_ids)
def test_ocr_engine_timer(test_file: Path,
                          ocr_params: OCRParams,
                          assets: Assets) -> None:
    image = load_rgb(test_file)

    engine = OCREngine(ocr_params, assets) # type: ignore

    timer, is_bc = engine.read_timer(image)
    if timer is not None:
        timer = str(timer)

    stem = test_file.stem

    if stem.startswith("bomb"):
        expected_result = (None, True)
    elif stem.startswith("inf"):
        expected_result = ("0:00", False)
    else:
        if "_" in stem:
            _, stem = stem.split("_", maxsplit=1)

        if " " in stem:
            expected_result = (stem.replace(" ", ":"), False)
        elif "." in stem:
            sec = stem[0]
            expected_result = (f"0:0{sec}", False)

    assert expected_result == (timer, is_bc)

    engine.stop()


@pytest.mark.skip(reason="Used to measure OCREngine.read_kfline performance")
def test_ocr_engine_kfline_performance(ocr_params: OCRParams,
                                       assets: Assets) -> None:
    images = [cv2.imread(str(file), cv2.IMREAD_COLOR)
              for file in kflines_test_files]
    n = len(images)

    engine = OCREngine(ocr_params, assets)

    n_iter = 500
    start = perf_counter()
    for i in range(n_iter):
        engine.read_kfline(images[i%n])

    end = perf_counter()
    total = end-start
    print(f"Total: {total}s\t\tAverage: {total/n_iter}s")
    
    engine.stop()
