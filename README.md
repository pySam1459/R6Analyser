# R6Analyser

This program performs real-time data extraction from a [Rainbow Six Siege](https://www.ubisoft.com/en-gb/game/rainbow-six/siege) game for statistical analysis.</br>
The following information gathered per round includes:
- **kill feed**
- **bomb plant condition**

## Config
To use R6Analyser, a configuration JSON is required to specify regions of the game window and other program parameters. The following outlines the required structure of this configuration file:

### Required Parameters
These parameters must be specified for the program to function correctly:
- `TIMER_REGION`: A list of 4 integers specifying the region of the game window where the timer is located.
- `KILL_FEED_REGION`: A list of 4 integers specifying the region of the game window where the kill feed is located.

### Optional Parameters
These parameters are optional and will default to values in `default.json` if not specified:
- `IGNS`: A list of strings specifying the in-game names (IGNs) of the players in the game. If this is not specified, the program will infer the names from the kill feed.
- `IGN_MODE`: A string specifying how the IGNs are processed; ["fixed" | "infer" | "opposition"].
- `SCREENSHOT_RESIZE_FACTOR`: A number specifying the factor by which the screenshot is resized before processing.
- `SCREENSHOT_PERIOD`: A number specifying the period in seconds between screenshots.

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

## Tools
This program uses:
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) to perform the necessary OCR for information gathering.
- [OpenCV](https://opencv.org/) for other computer vision tasks.
- [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) to take screenshots of the game window.
- [Numpy](https://numpy.org/) for matrix operations.
