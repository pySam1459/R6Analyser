import os
import pytest

from src.config import create_config, Config


@pytest.fixture
def resource_path() -> str:
    return os.path.join(os.path.dirname("__file__"), "resources")


def assert_config_requireds(config: Config) -> None:
    assert hasattr(config, "scrim")
    assert hasattr(config, "spectator")
    assert hasattr(config, "capture")
    assert hasattr(config.capture, "mode")
    assert hasattr(config.capture, "regions")
    assert hasattr(config.capture.regions, "timer")
    assert hasattr(config.capture.regions, "kf_line")
    assert hasattr(config, "igns")


def assert_config_defaults(config: Config) -> None:
    assert hasattr(config, "screenshot_resize")
    assert hasattr(config, "screenshot_period")
    assert hasattr(config, "last_winner")


def assert_config_inferreds(config: Config) -> None:
    assert hasattr(config, "ign_mode")
    assert hasattr(config, "max_rounds")
    assert hasattr(config, "rounds_per_side")
    assert hasattr(config.capture.regions, "team1_score")
    assert hasattr(config.capture.regions, "team2_score")
    assert hasattr(config.capture.regions, "team1_side")
    assert hasattr(config.capture.regions, "team2_side")


@pytest.mark.parametrize("good_config", [
    "good_config1.json"
])


def test_create_config(resource_path: str, good_config: str) -> None:
    config_path = os.path.join(resource_path, good_config)
    config = create_config(config_path, args=mock_config_args)

    assert_config_requireds(config)
    assert_config_defaults(config)
    assert_config_inferreds(config)
