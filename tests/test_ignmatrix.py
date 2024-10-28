import pytest
from pathlib import Path
from Levenshtein import ratio
from typing import Any

from ignmatrix.fixed import IGNMatrixFixed
from utils import load_json
from utils.enums import IGNMatrixMode, Team


path = Path(__file__).parent / "resources" / "ignmatrix" / "fixed.json"
testdata = load_json(path)

IM_LEVEN_THRESHOLD = 0.7

@pytest.fixture
def teams() -> tuple[list[str], list[str]]:
    return (testdata["team0"], testdata["team1"])

@pytest.fixture
def fixed_mat(teams) -> IGNMatrixFixed:
    return IGNMatrixFixed(teams[0], teams[1], IM_LEVEN_THRESHOLD)


@pytest.mark.parametrize("case",
                         testdata["good_cases"],
                         ids=lambda case: case["id"])
def test_fixed_good_name(fixed_mat: IGNMatrixFixed, case: dict[str, Any]) -> None:
    pign, pteam = case["pign"], Team(case["pteam"])
    pl = fixed_mat.get(pign, pteam)
    score = fixed_mat.evaluate(pign)

    assert pl is not None
    assert pl.ign == case["rign"], f"{pl.ign=}"
    assert score >= IM_LEVEN_THRESHOLD, f"{score=:.4f}"

@pytest.mark.parametrize("case",
                         testdata["bad_cases"],
                         ids=lambda case: case["id"])
def test_fixed_bad_names(fixed_mat: IGNMatrixFixed, case: dict[str, Any]) -> None:
    pign, pteam = case["pign"], Team(case["pteam"])
    pl = fixed_mat.get(pign, pteam)
    score = fixed_mat.evaluate(pign)

    assert pl is None, f"Pl is not None {pl}, ratio: {ratio(pign, pl.ign)=:.4f}"

    if pl is not None and pl.team == pteam:
        assert score < IM_LEVEN_THRESHOLD, f"Score >= threshold, eval: {score=:.4f}"


def test_fixed_mode(fixed_mat: IGNMatrixFixed) -> None:
    assert fixed_mat.mode == IGNMatrixMode.FIXED


def test_fixed_get_teams(teams) -> None:
    mat = IGNMatrixFixed(teams[0], teams[1], IM_LEVEN_THRESHOLD)
    mat_teams = mat.get_teams()
    out = (mat_teams.team0, mat_teams.team1)

    for i, (ot, kt) in enumerate(zip(out, teams)):
        for opl, kign in zip(ot, kt):
            assert opl.ign == kign
            assert opl.team == i
