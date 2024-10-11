import cv2
import pytest
from os import listdir, makedirs
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from Levenshtein import ratio

from assets import Assets
from ocr import OCREngine, OCRLineResult
from settings import Settings, create_settings


def load_test_files(parent: Path) -> list[Path]:
    return [parent / file for file in listdir(parent) if file.endswith((".png", ".jpg"))]


res_path = Path(__file__).parent / "resources"
out_path = Path(__file__).parent / "out" / "ocrengine"

kflines_test_files_parent = res_path / "kflines"
kflines_test_files = load_test_files(kflines_test_files_parent)


@pytest.fixture(scope="session", autouse=True)
def setup_test_dir():
    ## prep out files
    if out_path.exists():
        rmtree(out_path)
    makedirs(out_path, exist_ok=True)

@pytest.fixture
def settings() -> Settings:
    settings_path = res_path / "settings" / "ocr_engine_settings.json"
    return create_settings(settings_path)

@pytest.fixture
def assets() -> Assets:
    assets_path = Path("assets")
    return Assets(assets_path)

def get_ids(test_file: Path) -> str:
    return test_file.stem


def write_kflines_out_images(test_file: Path, olr: OCRLineResult):
    test_case_out_path = out_path / test_file.stem.replace(".", "")
    test_case_out_path.mkdir(exist_ok=True)
    if olr.left_image is not None:
        li = cv2.cvtColor(olr.left_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(test_case_out_path / "left.jpg"), li)
    if olr.middle_image is not None:
        mi = cv2.cvtColor(olr.middle_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(test_case_out_path / "middle.jpg"), mi)
    if olr.right_image is not None:
        ri = cv2.cvtColor(olr.right_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(test_case_out_path / "right.jpg"), ri)


@pytest.mark.parametrize("test_file", kflines_test_files, ids=get_ids)
def test_ocr_engine_kfline(test_file: Path, settings: Settings, assets: Assets) -> None:
    image = cv2.imread(str(test_file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    engine = OCREngine(settings, assets)

    ocr_line_res = engine.read_kfline(image)

    if ocr_line_res is not None:
        write_kflines_out_images(test_file, ocr_line_res)

    if test_file.stem.startswith("none"):
        assert ocr_line_res is None, "OCRLineResult must be None"
        return

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
        assert rat_left >= 0.8, f"ratio: {rat_left}, left value: {ocr_line_res.left}"
    
    rat_right = ratio(ocr_line_res.right, right)
    assert rat_right >= 0.8, f"ratio: {rat_right}, right value: {ocr_line_res.right}"

    # assert ocr_line_res.headshot == headshot, f"headshot: {ocr_line_res.headshot}"
    engine.stop()


timer_test_files_parent = res_path / "timer"
timer_test_files = load_test_files(timer_test_files_parent)

@pytest.mark.parametrize("test_file", timer_test_files, ids=get_ids)
def test_ocr_engine_timer(test_file: Path, settings: Settings) -> None:
    image = cv2.imread(str(test_file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    engine = OCREngine(settings, None) # type: ignore

    timer, is_bc = engine.read_timer(image)
    if timer is not None:
        timer = str(timer)

    if test_file.stem.startswith("bomb"):
        expected_result = (None, True)
    if " " in test_file.stem:
        expected_result = (test_file.stem.replace(" ", ":"), False)
    elif "." in test_file.stem:
        sec = test_file.stem[0]
        expected_result = (f"0:0{sec}", False)

    assert expected_result == (timer, is_bc)

    engine.stop()


@pytest.mark.skip(reason="Used to measure OCREngine.read_kfline performance")
def test_ocr_engine_kfline_performance(settings: Settings, assets: Assets) -> None:
    images = [cv2.imread(str(file), cv2.IMREAD_COLOR)
              for file in kflines_test_files]
    n = len(images)

    engine = OCREngine(settings, assets)

    n_iter = 500
    start = perf_counter()
    for i in range(n_iter):
        engine.read_kfline(images[i%n])

    end = perf_counter()
    total = end-start
    print(f"Total: {total}s\t\tAverage: {total/n_iter}s")
    