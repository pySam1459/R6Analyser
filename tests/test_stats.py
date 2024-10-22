import pytest
from itertools import repeat
from pathlib import Path
from unittest.mock import patch
from typing import Any, TypeAlias

from utils import Timestamp, load_json
from utils.enums import Team
from history import HistoryRound, KFRecord
from ignmatrix import Player
from ignmatrix.player import FixedPlayer
from stats import RoundStats_t, compile_round_stats


TestCase_t:   TypeAlias = dict[str,Any]
Assertions_t: TypeAlias = dict[str,dict[str,Any]]
PlayerMap_t:  TypeAlias = dict[str,Player]
# hround, player_map, assertions, ids
TestData_t:   TypeAlias = tuple[HistoryRound, PlayerMap_t, Assertions_t, list[str]]

base = Path(__file__).parent / "resources" / "stats"


def parse_kfrecord(el: dict[str,Any], player_map: PlayerMap_t) -> KFRecord:
    assert "player" in el and "target" in el and "time" in el
    return KFRecord(player_map[el["player"]],
                    player_map[el["target"]],
                    Timestamp(**el["time"]),
                    el.get("headshot", False))


EXCLUDE_PROPS = ["id", "assertions"]
def parse_testcase(case: TestCase_t, player_map: PlayerMap_t) -> TestCase_t:
    case = {k:v for k,v in case.items() if k not in EXCLUDE_PROPS}
    if "killfeed" in case:
        case["killfeed"] = [parse_kfrecord(el, player_map) for el in case["killfeed"]]
    return case


def load_testdata(filename: str) -> list[TestData_t]:
    raw_testdata = load_json(base / filename)

    player_igns: list[str] = raw_testdata["players"]
    player_map: dict[str,Player] = {ign: FixedPlayer(ign, team=Team(i>=5)) for i,ign in enumerate(player_igns)}

    raw_testcases: list[TestCase_t] = raw_testdata["test_cases"]
    clean_testcases = [parse_testcase(case, player_map) for case in raw_testcases]
    hrounds = [HistoryRound.model_validate(ccase) for ccase in clean_testcases]

    assertions = [case.get("assertions", {}) for case in raw_testcases]
    ids = [case.get("id", f"test-case-{i+1}") for i, case in enumerate(raw_testcases)]
    return list(zip(hrounds, repeat(player_map), assertions, ids))


def check_assertions(rs: RoundStats_t, player_map: PlayerMap_t, assertions: Assertions_t) -> None:
    for pign, asrts in assertions.items():
        if pign not in player_map:
            raise ValueError(f"Assertion player IGN {pign} is not in players list")

        pl = player_map[pign]
        prs = rs[pl.uid]
        for assert_key, assert_value in asrts.items():
            attr_value = getattr(prs, assert_key)
            assert attr_value == assert_value, f"{pign}: {assert_key} | found {attr_value}, expected {assert_value}"


@pytest.mark.parametrize("testdata",
                         load_testdata("single_rounds.json"),
                         ids=lambda td: td[3])
def test_single_rounds(testdata: TestData_t) -> None:
    hround, player_map, assertions, _ = testdata

    rs = compile_round_stats(hround, list(player_map.values()))
    check_assertions(rs, player_map, assertions)
