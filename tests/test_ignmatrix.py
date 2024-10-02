import pytest
from pathlib import Path
from Levenshtein import ratio

from ignmatrix.fixed import IGNMatrixFixed
from utils import load_json
from utils.constants import IM_LEVEN_THRESHOLD
from utils.enums import IGNMatrixMode


path = Path(__file__).parent / "resources" / "ignmatrix" / "fixed.json"
testdata = load_json(path)


@pytest.fixture
def teams() -> tuple[list[str], list[str]]:
    return (testdata["team0"], testdata["team1"])

@pytest.fixture
def fixed_mat(teams) -> IGNMatrixFixed:
    return IGNMatrixFixed(*teams)


@pytest.mark.parametrize("case",
                         testdata["good_cases"],
                         ids=lambda case: case["id"])
def test_fixed_good_name(fixed_mat: IGNMatrixFixed, case: dict[str, str]) -> None:
    pign = case["pign"]
    pl = fixed_mat.get(pign)
    score = fixed_mat.evaluate(pign)

    assert pl is not None
    assert pl.ign == case["rign"], f"{pl.ign=}"
    assert score >= IM_LEVEN_THRESHOLD, f"{score=:.4f}"

@pytest.mark.parametrize("case",
                         testdata["bad_cases"],
                         ids=lambda case: case["id"])
def test_fixed_bad_names(fixed_mat: IGNMatrixFixed, case) -> None:
    pign = case["pign"]
    pl = fixed_mat.get(pign)
    score = fixed_mat.evaluate(pign)

    assert pl is None, f"{ratio(pign, pl.ign)=:.4f}"
    assert score < IM_LEVEN_THRESHOLD, f"{score=:.4f}"


def test_fixed_mode(fixed_mat: IGNMatrixFixed) -> None:
    assert fixed_mat.mode == IGNMatrixMode.FIXED


def test_fixed_get_teams(teams) -> None:
    mat = IGNMatrixFixed(*teams)
    mat_teams = mat.get_teams()
    out = (mat_teams.team0, mat_teams.team1)

    for i, (ot, kt) in enumerate(zip(out, teams)):
        for opl, kign in zip(ot, kt):
            assert opl.ign == kign
            assert opl.team == i
