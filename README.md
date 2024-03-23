# R6Analyser

This program performs real-time data extraction from a [Rainbow Six Siege](https://www.ubisoft.com/en-gb/game/rainbow-six/siege) game for statistical analysis.</br>
The following information gathered per round includes:
- **Score Line**
- **Kill Feed**
- **Bomb Plant At**
- **Round End At**

## Config
To use R6Analyser, a configuration JSON is required to specify regions of the game window and other program parameters. The following outlines the required and optional paremeters of this configuration file:

### Required Parameters
These parameters must be specified for the program to function correctly; `REGION` parameters can be found using `src\region-tool.py`:
- `SCRIM`: A boolean (true/false) specifying whether the game is a scrim or not.
- `TIMER_REGION`: A list of 4 integers specifying the region of the game window where the timer is located.
- `KILL_FEED_REGION`: A list of 4 integers specifying the region of the game window where the kill feed is located.

### Optional Parameters
These parameters are optional and will default to values in `default.json` if not specified:

- `IGNS`: (RECOMMENDED) A list of strings specifying the in-game names (IGNs) of the players in the game. The first 5 IGNs will be considered team 1. If this is not specified, the program will infer the names from the kill feed.

- `IGN_MODE`: Specifies how the IGNs are processed. There are three modes available:
  - `fixed`: Will return `None` for all non-fixed IGNs. This mode is used when you have a predefined list of IGNs, and any IGN not in this list will not be considered.
  - `infer`: Will infer the non-fixed IGNs from the OCR's output. Use this mode if you want the program to automatically identify and use IGNs from the game feed.
  - `opposition`: Will return the initial parameter `opp_value` (defaulting to `'OPPOSITION'`) for non-fixed IGNs. This mode is useful when you are only interested in statistics for one team and want to group all opposing team members under a single identifier.

- `SCREENSHOT_RESIZE_FACTOR`: A number specifying the factor by which the screenshot is resized before processing. This can help in optimizing the OCR performance by adjusting the image size.
- `SCREENSHOT_PERIOD`: A number specifying the period in seconds between screenshots. This determines how frequently the program captures the game feed for analysis.
- `SPECTATOR`: (true/false) specifying if the game perspective is in spectator mode, compared to in-person (default).

### Inferred Parameters
These parameters are optional and will be inferred by the program if not explicitly specified:
- `MAX_ROUNDS`: The maximum number of rounds in the game. If scrim is set to true, this will default to 12; otherwise, it will default to 15.
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

## Process
The program works by taking screenshots of the game window at regular intervals `SCREENSHOT_PERIOD` and using OCR to extract the necessary information. Currently, 4 regions of the screenshot are used for OCR:
- **Timer Region**: Used to extract the round timer.
- **Kill Feed Region**: Used to extract the kill feed.
- **Team 1 Score Region**: Used to extract team 1's score.
- **Team 2 Score Region**: Used to extract team 2's score.

### New Round
A new round is detected when the score line changes, determines using the score regions. When this occurs, the program will call `Analyser.__new_round` which will:
- Set the winner of the previous round.
- Update the score line.
- Create a new record for the new round, including new kill feed, win conditions, and timer information.

### Mid Round
During a round, the program will continuously screenshot the game window, reading the timer and kill feed. When a new kill is detected, the program will record the `player` (killer), `target` (dead) and `time` at which the kill occurred.</br>
If the defuser is planted, the program will record the time and will update how the time is displayed: `!0:[45-0]`, using a `!` to indicate the defuser is planted.

### End Round
A round ends when the program cannot read the timer, the red bomb countdown indicator is not showing, and `END_ROUND_SECONDS = 12` have passed. When this occurs, the program will call `Analyser.__end_round` which will:
- Set the win condition for the current round.
- Set the round end at time.
- Check if the game has ended (currently detected by the number of rounds passed), and if so, call `Analyser.__end_game`.
- Otherwise, reset the state of the program for the next round.

### End Game
When the `Analyser.__end_game` method is called, the program will save the recorded game data to a JSON or XLSX file, which can be specified using the `-s` / `--save` argument. (Currently only JSON is supported)

## Requirements
This program uses:
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) to perform the necessary OCR for information gathering.
- [OpenCV](https://opencv.org/) for other computer vision tasks.
- [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) to take screenshots of the game window.
- [Numpy](https://numpy.org/) for matrix operations.
