# R6Analyser

This program performs real-time data extraction from a [Rainbow Six Siege](https://www.ubisoft.com/en-gb/game/rainbow-six/siege) game for statistical analysis.</br>
The following information gathered per round includes:
- **Score Line**
- **Kill Feed**
- **Bomb Plant At**
- **Round End At**
- **Win Condition**

## How to run
To run the program, you will need to have Python (3.10>=) installed on your system. You can download Python from the [official website](https://www.python.org/downloads/). A Nvidia GPU and [CUDA](https://developer.nvidia.com/cuda-toolkit) is also recommended for real-time data extraction, although this program does provide a CPU-only option.</br>
Before running the program, you will need to install the required packages using the following command:
```bash
pip install -r requirements.txt
```
You must also have made a configuration file (see below) to specify the regions of the game window and other parameters. 
To check the regions defined in the configuration file, you can run the following command:
```bash
python src\run.py <config file> --check
```
which will save the screenshots of the regions defined into a new `images` directory.</br>
To run the program, use the following command:
```bash
python src\run.py <config file>
```
and it is recommended to add a save path, where to save the game data to a JSON or XLSX file:
```bash
python src\run.py <config file> --save <save file>
```
Other optional arguments include:
- `-v` / `--verbose`: Print additional information to the console (0-3).
- `-d` / `--delay`: The delay in seconds before the program starts capturing the game window.
- `-s` / `--save`: File to save the game data, either a JSON or XLSX file.
- `--append-save`: Whether to append the game data to an existing save file or overwrite it.
- `--upload-save`: (WIP) Uploads the round data to a Google Sheet.
- `--test`: (Debugging) Runs the OCR engine for a single instance of each region and prints the output to the console.
- `--cpu`: (NOT recommended) Use the CPU for OCR instead of the GPU.


## Config
To use R6Analyser, a configuration JSON is required to specify regions of the game window and other program parameters. The following outlines the required and optional paremeters of this configuration file:

### Required Parameters
These parameters must be specified for the program to function correctly; `REGION` parameters can be found using `src\region-tool.py`:
- `SCRIM`: A boolean (true/false) specifying whether the game is a scrim or not.
- `SPECTATOR`: (`true`/`false`) specifying if the game perspective is in spectator mode, compared to in-person (default).
- `TIMER_REGION`: A list of 4 integers specifying the region of the game window where the timer is located.
- `KILLFEED_REGION`: A list of 4 integers specifying the region of the game window where the kill feed is located.
- `IGNS`: A list of 0-10 strings specifying the in-game names (IGNs) of the players in the game. It is recommended to specify all 10 IGNs to maximise the accuracy of the program. First 5 IGNs are 1 team, last 5 are the other team.

### Inferred Parameters
These parameters are optional and will be inferred by the program if not explicitly specified:
- `MAX_ROUNDS`: The maximum number of rounds in the game. If scrim is set to true, this will default to 12; otherwise, it will default to 15.
- `ROUNDS_PER_SIDE`: The number of rounds per atk/def side. Inferred to be (MAX_ROUNDS-3) / 2
- `IGN_MODE`: Specifies how the IGNs are processed. There are two modes available:
  - `fixed`: This mode is used when you have a predefined list of IGNs, and any IGN not in this list will not be considered, returned as `None`.
  - `infer`: Use this mode if you want the program to automatically identify and use IGNs from the game feed. If 10 IGNs are already provided, the mode will default to `fixed`. 
- `TEAM1_SCORE_REGION`: Specifies the region of the game window where team 1's score is displayed.
- `TEAM2_SCORE_REGION`: Specifies the region of the game window where team 2's score is displayed.
- `TEAM1_SIDE_REGION`: Specifies the region of the game window where team 1's side icon is displayed.
- `TEAM2_SIDE_REGION`: Specifies the region of the game window where team 2's side icon is displayed.

### Optional Parameters
These parameters are optional and will default to values in `default.json` if not specified:
- `SCREENSHOT_RESIZE`: (default=4) A number specifying the factor by which the screenshot is resized before OCR-processing.
- `SCREENSHOT_PERIOD`: (default=0.5) This number determines how frequently the program captures the game feed for analysis, the period in seconds between screenshots


### Config File Example
Below is an example configuration file that specifies these parameters:
```json
{
  "SCRIM": true,
  "SPECTATOR": false,
  "TIMER_REGION": [1210, 110, 140, 65],
  "KILLFEED_REGION": [1640, 310, 565, 140],
  "IGNS": [
    "Player1",
    "Player2",
    "...",
    "Player10"
  ],
  "IGN_MODE": "fixed",
  "MAX_ROUNDS": 12
}
```

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