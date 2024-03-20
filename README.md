# R6Analyser

This program performs real-time data extraction from a [Rainbow Six Siege](https://www.ubisoft.com/en-gb/game/rainbow-six/siege) game for statistical analysis.</br>
The following information gathered per round includes:
- **Kill Feed**
- **Score Line**
- **Bomb Plant Condition**

## Config
To use R6Analyser, a configuration JSON is required to specify regions of the game window and other program parameters. The following outlines the required and optional paremeters of this configuration file:

### Required Parameters
These parameters must be specified for the program to function correctly; `REGION` parameters can be found using `src\region-tool.py`:
- `TIMER_REGION`: A list of 4 integers specifying the region of the game window where the timer is located.
- `KILL_FEED_REGION`: A list of 4 integers specifying the region of the game window where the kill feed is located.

### Optional Parameters
These parameters are optional and will default to values in `default.json` if not specified:

- `IGNS`: A list of strings specifying the in-game names (IGNs) of the players in the game. The first 5 IGNs will be considered team 1. If this is not specified, the program will infer the names from the kill feed.

- `IGN_MODE`: Specifies how the IGNs are processed. There are three modes available:
  - `fixed`: Will return `None` for all non-fixed IGNs. This mode is used when you have a predefined list of IGNs, and any IGN not in this list will not be considered.
  - `infer`: Will infer the non-fixed IGNs from the OCR's output. Use this mode if you want the program to automatically identify and use IGNs from the game feed.
  - `opposition`: Will return the initial parameter `opp_value` (defaulting to `'OPPOSITION'`) for non-fixed IGNs. This mode is useful when you are only interested in statistics for one team and want to group all opposing team members under a single identifier.

- `SCREENSHOT_RESIZE_FACTOR`: A number specifying the factor by which the screenshot is resized before processing. This can help in optimizing the OCR performance by adjusting the image size.

- `SCREENSHOT_PERIOD`: A number specifying the period in seconds between screenshots. This determines how frequently the program captures the game feed for analysis.

### Inferred Parameters
These parameters are optional and will be inferred by the program if not explicitly specified:
- `TEAM1_SCORE_REGION`: Specifies the region of the game window where team 1's score is displayed.
- `TEAM2_SCORE_REGION`: Specifies the region of the game window where team 2's score is displayed.

### Config File Example
Below is an example configuration file that specifies these parameters:
```json
{
  "TIMER_REGION": [1210, 110, 140, 65],
  "KILL_FEED_REGION": [1640, 310, 565, 140],
  "IGNS": [
    "Player1",
    "Player2",
    ...
    "Player10"
  ],
  "IGN_MODE": "fixed"
}
```

## Requirements
This program uses:
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) to perform the necessary OCR for information gathering.
- [OpenCV](https://opencv.org/) for other computer vision tasks.
- [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) to take screenshots of the game window.
- [Numpy](https://numpy.org/) for matrix operations.
