from pathlib import Path
from re import compile
from string import ascii_lowercase, ascii_uppercase


IGN_REGEX            = compile(r"^[0-9a-zA-Z\-_\.]{2,18}$")
SOFTWARE_KEY_PATTERN = r"[a-f0-9]{64}"
SCORELINE_PATTERN    = r"^[0-9O]{1,2}$"
TIMESTAMP_PATTERN    = r"^(\d+:)?(\d+:)?\d{1,2}$"
NUMBER_PATTERN       = r"^\d+$"

__SETTINGS_PATH       = Path("settings")
DEBUG_PATH            = __SETTINGS_PATH / "debug.json"
DEFAULTS_PATH         = __SETTINGS_PATH / "defaults.json"
SETTINGS_PATH         = __SETTINGS_PATH / "settings.json"
GAME_SETTINGS_PATH    = __SETTINGS_PATH / "game-settings.json"
SOFTWARE_KEY_FILE     = Path("SOFTWARE_KEY")
ASSETS_PATH           = Path("assets")
DEFAULT_SAVE_DIR      = Path("saves")
DEFAULT_IMAGE_DIR     = Path("images")
DEFAULT_TESSDATA_PATH = Path("./tessdata")

DEFAULT_LANGUAGE = "en"

KC_ENDPOINT = "kcehcyek"[::-1]

## Console text colour codes
RED   = '\033[91m'
YLW   = '\u001b[33m'
WHITE = '\033[97m'

## Red Thresholds (RT)
BOMB_COUNTDOWN_RT = 0.70
TIMER_LAST_SECONDS_RT = 0.65

OCR_TIMER_THRESHOLD = 20

DIGITS         = "0123456789"
TIMER_CHARLIST = DIGITS + ":."
IGN_EXTRA_CHARS = ".-_"
IGN_CHARLIST = ascii_lowercase + ascii_uppercase + DIGITS + IGN_EXTRA_CHARS
SCORE_CHARLIST = DIGITS + "O"


KILLFEED_PHRASES = [
    "has found the bomb",
    "Friendly Fire has been activated for",
    "Friendly Fire turned off until Action Phase"
]
