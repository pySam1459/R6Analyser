from pathlib import Path
from re import compile


IGN_REGEX = compile(r"^[0-9a-zA-Z\-_\.]{2,18}$")
SOFTWARE_KEY_PATTERN = r"^[a-f0-9]{64}$"
SCORELINE_PATTERN = r"^\d+$"

__SETTINGS_PATH       = Path("settings")
DEBUG_PATH            = __SETTINGS_PATH / "debug.json"
DEFAULTS_PATH         = __SETTINGS_PATH / "defaults.json"
SETTINGS_PATH         = __SETTINGS_PATH / "settings.json"
SOFTWARE_KEY_FILE     = Path("SOFTWARE_KEY")
ASSETS_PATH           = Path("assets")
DEFAULT_SAVE_DIR      = Path("saves")
DEFAULT_IMAGE_DIR     = Path("images")
DEFAULT_TESSDATA_PATH = Path("./tessdata")

DEFAULT_LANGUAGE = "en"

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
IGN_CHARLIST   = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"

## IGNMAT
IM_LEVEN_THRESHOLD = 0.65 ## threshold for Levenshtein distance to determine equality
IM_TEAM_DET_THRESHOLD = 3


LANGUAGES = [
    "en"
]

KILLFEED_PHRASES = [
    "has found the bomb",
    "Friendly Fire has been activated for",
    "Friendly Fire turned off until Action Phase"
]
