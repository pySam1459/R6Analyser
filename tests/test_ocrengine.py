import cv2
import pytest
from os import listdir, makedirs
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from Levenshtein import ratio
from PIL import Image

from assets import Assets
from ocr import OCREngine, OCRLineResult, OCRParams
from settings import Settings, create_settings
from utils import load_json, squeeze_image, clip_around


def load_test_files(parent: Path) -> list[Path]:
    return [parent / file for file in listdir(parent) if file.endswith((".png", ".jpg"))]

project_base = Path(__file__).parent.parent
res_path = project_base / "tests" / "resources"
out_path = project_base / "tests" / "out" / "ocrengine"

sl_test_files_parent = res_path / "scoreline"
sl_test_files = load_test_files(sl_test_files_parent)

kflines_test_files_parent = res_path / "kfline"
kflines_test_files = load_test_files(kflines_test_files_parent)

timer_test_files_parent = res_path / "timer"
timer_test_files = load_test_files(timer_test_files_parent)


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

@pytest.fixture
def ocr_params() -> OCRParams:
    defaults = load_json(project_base / "settings" / "defaults.json")
    assert "ocr_params" in defaults

    ocr_params = defaults["ocr_params"]
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
                              settings: Settings) -> None:
    image = cv2.imread(str(test_file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    engine = OCREngine(ocr_params, settings, None)  # type: ignore
    score = engine.read_score(image)

    # image_ = clip_around(image, ocr_params.sl_clip_around)
    # image_ = cv2.resize(image_, None, fx=ocr_params.sl_scalex, fy=ocr_params.sl_scaley, interpolation=cv2.INTER_CUBIC)
    # th_img = engine._debug_threshold(image_)
    # write_score_out_image(test_file, th_img)

    expected_text, _ = test_file.stem.split("_", maxsplit=1)

    if expected_text.lower() == "none":
        assert score is None, "Score is not None"
    else:
        assert score is not None, f"None was returned instead of: {expected_text}"
        assert score == expected_text, f"Score: {score} does not match expected text: {expected_text}"


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


@pytest.mark.parametrize("test_file", kflines_test_files, ids=get_ids)
def test_ocr_engine_kfline(test_file: Path,
                           ocr_params: OCRParams,
                           settings: Settings,
                           assets: Assets) -> None:
    image = cv2.imread(str(test_file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    engine = OCREngine(ocr_params, settings, assets)

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

    MIN_RATIO = 0.7

    if left == "":
        assert ocr_line_res.left is None, f"if left='', ocr_left is {ocr_line_res.left}"
    else:
        assert ocr_line_res.left is not None, f"ocr left should be defined"
        rat_left = ratio(ocr_line_res.left, left)
        assert rat_left >= MIN_RATIO, f"ratio: {rat_left}, left value: {ocr_line_res.left}"
    
    rat_right = ratio(ocr_line_res.right, right)
    assert rat_right >= MIN_RATIO, f"ratio: {rat_right}, right value: {ocr_line_res.right}"

    assert ocr_line_res.headshot == headshot, f"headshot: {ocr_line_res.headshot}"
    engine.stop()


# --- TIMER ---
@pytest.mark.parametrize("test_file", timer_test_files, ids=get_ids)
def test_ocr_engine_timer(test_file: Path,
                          ocr_params: OCRParams,
                          settings: Settings) -> None:
    image = cv2.imread(str(test_file), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    engine = OCREngine(ocr_params, settings, None) # type: ignore

    timer, is_bc = engine.read_timer(image)
    if timer is not None:
        timer = str(timer)

    stem = test_file.stem

    if stem.startswith("bomb"):
        expected_result = (None, True)
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
                                       settings: Settings,
                                       assets: Assets) -> None:
    images = [cv2.imread(str(file), cv2.IMREAD_COLOR)
              for file in kflines_test_files]
    n = len(images)

    engine = OCREngine(ocr_params, settings, assets)

    n_iter = 500
    start = perf_counter()
    for i in range(n_iter):
        engine.read_kfline(images[i%n])

    end = perf_counter()
    total = end-start
    print(f"Total: {total}s\t\tAverage: {total/n_iter}s")
    