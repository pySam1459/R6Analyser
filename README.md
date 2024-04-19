# R6Analyser

This program performs real-time data extraction from a [Rainbow Six Siege](https://www.ubisoft.com/en-gb/game/rainbow-six/siege) VOD or live game for statistical analysis.</br>
The following information gathered per round includes:
- **Score Line**
- **Kill Feed**, **Trades**, **Refragged Kills**, **Refragged Deaths**
- **Bomb Plant At**, **Defused At**
- **Round End At**
- **Win Condition**
- **Opening Kill**, **Opening Death**
- **Atk/Def Side**

## How to run
To run the program, you will need to have Python (3.10>=) installed on your system. You can download Python from the [official website](https://www.python.org/downloads/). A Nvidia GPU and [CUDA](https://developer.nvidia.com/cuda-toolkit) is also recommended for real-time data extraction, although this program does provide a CPU-only option (NOT RECOMMENDED).</br>
Before running the program, you will need to install the required packages using the following command:
```bash
pip install -r requirements.txt
```
You must also have made a configuration file (see below) to specify certain regions of the game window and other parameters. 
To check the regions defined in the configuration file, you can run the following command:
```bash
python src\run.py -c <config file> --check
```
which will save the screenshots of the regions defined into a new `images` directory. If these screenshot's do not contain the relevant information, they should be re-defined.</br>
To run the program, use the following command:
```bash
python src\run.py -c <config file>
```
and it is recommended to add a save path, where to save the game data to a JSON or XLSX file:
```bash
python src\run.py -c <config file> --save <save file>
```
Other optional arguments include:
- `-v` / `--verbose`: Print additional information to the console (0-3).
- `-d` / `--delay`: The delay in seconds before the program starts capturing the game window.
- `-s` / `--save`: File to save the game data, either a JSON or XLSX file.
- `--append-save`: Whether to append the game data to an existing save file or overwrite it.
- `--test`: (Debugging) Runs the OCR engine for a single instance of each region and prints the output to the console.
- `--cpu`: (NOT recommended) Use the CPU for OCR instead of the GPU.
- `--region-tool`: Runs the region tool used to find config `_REGION` parameters.
- `--display`: The display number to take the screenshot from (default=0).

## How to use
This program uses OCR to extract text information from the game window, so it is important to have a good quality video and high resolution. If you cannot read the text on the screen, the program will not be able to either. That being said, the program can handle relatively low quality (720p).</br>
tl;dr The better the quality of the video/game, the more accurate the program will be.</br>

## Config
To use R6Analyser, a configuration JSON is required to specify regions of the game window and other program parameters. An example config file can be found at `configs/example.json` and below. `_REGION` parameters can be found using the region tool, which can be run.
```bash
python src\run.py --region-tool --display <display number>
```
This tool takes a screenshot of the window and allows you to select regions using the mouse. The region parameter values are then printed to the console.

The following outlines the required and optional paremeters of a config file:

### Required Parameters
These parameters must be specified:
- `SCRIM`: `true`/`false` specifying whether the game is a scrim or not.
- `SPECTATOR`: `true`/`false` specifying if the game perspective is in spectator mode or in-person.
- `TIMER_REGION`: A list of 4 integers specifying the region of the game window where the timer is located.
- `KF_LINE_REGION`: A list of 4 integers specifying the region of the first line in the kill feed. The region should be filled by the IGN boxes, but should extend leftwards to be able to fit in longer names.
- `IGNS`: A list of 0-10 strings specifying the in-game names (IGNs) of the players in the game. IGNs 1-5 will be considered as one team, with IGNs 6-10 as the other team. It is recommended to specify all 10 IGNs to maximise the accuracy of the program, with a minimum recommendation of 5 IGNs (your team).

### Inferred Parameters
These parameters are optional and will be inferred by the program if not explicitly specified:
- `MAX_ROUNDS`: The maximum number of rounds in the game. If scrim is set to true, this will default to 12; otherwise, it will default to 15. (Infinite OT is not supported yet)
- `ROUNDS_PER_SIDE`: The number of rounds per atk/def side. Inferred to be (MAX_ROUNDS-3) / 2
- `IGN_MODE`: Specifies how the IGNs are processed. There are two modes available:
  - `fixed`: This mode is used when you have a predefined list of IGNs, and any IGN not in this list will not be considered, returned as `None`.
  - `infer`: Use this mode if you want the program to automatically identify and use IGNs from the game feed. If 10 IGNs are already provided, the mode will default to `fixed`. 
- `TEAM1_SCORE_REGION`: The region where team 1's score is displayed.
- `TEAM2_SCORE_REGION`: The region where team 2's score is displayed.
- `TEAM1_SIDE_REGION`: The region where team 1's side icon is displayed.
- `TEAM2_SIDE_REGION`: The region where team 2's side icon is displayed.
- `KILLFEED_REGION`: The region where the kill feed is displayed. This region is inferred from the `KF_LINE_REGION` as approximately 4x the height of the `KF_LINE_REGION`.

### Optional Parameters
These parameters are optional and will default to values in `default.json` if not specified:
- `SCREENSHOT_RESIZE`: (default=4) A number specifying the factor by which the screenshot is resized before OCR-processing.
- `SCREENSHOT_PERIOD`: (default=0.5) This number determines how frequently the program captures the game feed for analysis, the period in seconds between screenshots


### Config File Example
Below is an example configuration file that specifies a set of possible parameters for a standard game:
```json
{
  "SCRIM": false,
  "SPECTATOR": false,
  "TIMER_REGION": [1210, 110, 140, 65],
  "KF_LINE_REGION": [1705, 413, 605, 31],
  "IGNS": [
    "Samba.",
    "Player2",
    "Player3",
    "Player4",
    "Player5"
  ],
  "MAX_ROUNDS": 7,
  "ROUNDS_PER_SIDE": 3,
}
```
With this config file, the remaining 5 player IGNs will be inferred from the game feed. Since standard only has 1 round of overtime, the `MAX_ROUNDS` and `ROUNDS_PER_SIDE` needed to be specified as the inferred `(MAX_ROUNDS-3) / 2` assumes a normal 3 round overtime.

## Process
The program works by taking screenshots of the game window at regular intervals `SCREENSHOT_PERIOD` and using OCR to extract the necessary information. Currently, 6 regions of the screenshot are used for OCR:
- **Timer Region**: Used to extract the round timer.
- **Kill Feed Region**: Used to extract the kill feed.
- **Team 1 Score Region**: Used to extract team 1's score.
- **Team 2 Score Region**: Used to extract team 2's score.
- **Team 1 Side Region**: Used to extract team 1's side.
- **Team 2 Side Region**: Used to extract team 2's side.
Although, only the Timer and Kill Feed regions are used every inference cycle, the other regions are used to infer the game state and are only used when a new round is detected.

### New Round
A new round is detected when the score line changes, determines using the score regions. When this occurs, the program will call `_new_round` which will:
- Set the winner of the previous round.
- Update the score line.
- Update the attack side.
- Create a new record for the new round, including new kill feed, win conditions, and timer information.

### Mid Round
During a round, the program will continuously screenshot the game window, reading the timer and kill feed. When a new kill is detected, the program will record the `player` (killer), `target` (dead) and `time` at which the kill occurred.</br>
If the defuser is planted, the program will continue to record the time using an internal clock (any video skips will interfere with this internal clock). If the time passes `0:00`, the timer will go into the negative, e.g. `0:-12`.

### End Round
A round ends when the program cannot read the timer, the red bomb countdown indicator is not showing, and `END_ROUND_SECONDS = 12` have passed. When this occurs, the program will call `_end_round` which will:
- Set the win condition for the current round.
- Set the round end at time.
- Check if the game has ended (currently detected by the number of rounds passed), and if so, call `_end_game`.
- Otherwise, reset the state of the program for the next round.

### End Game
When the `_end_game` method is called, the program will save the recorded game data to a JSON or XLSX file, which can be specified using the `-s` / `--save` argument. (Currently only JSON is supported)

## Requirements
This program uses:
- [Python](https://www.python.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) to perform the necessary OCR for information gathering.
- [Cuda](https://developer.nvidia.com/cuda-toolkit) for GPU acceleration.
- [OpenCV](https://opencv.org/) for other computer vision tasks.
- [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) to take screenshots of the game window.
- [Numpy](https://numpy.org/) for matrix operations.
and a few other packages which can be found in the `requirements.txt` file.