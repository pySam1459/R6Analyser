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
python src\run.py <config file> --check
```
which will save the screenshots of the regions defined into a new `images` directory. If these screenshot's do not contain the relevant information, they should be re-defined.</br>
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
- `--test`: (Debugging) Runs the OCR engine for a single instance of each region and prints the output to the console.
- `--region-tool`: Runs the region tool used to find config region parameters.
- `--display`: The display number to take the screenshot from (default=0).

## How to use
This program uses OCR to extract text information from the game window, so it is important to have a good quality video and high resolution. If you cannot read the text on the screen, the program will not be able to either. That being said, the program can handle relatively low quality (720p).</br>
tl;dr The better the quality of the video/game, the more accurate the program will be.</br>

## Config
To use R6Analyser, a configuration JSON file is required to specify regions of the game window and other program parameters. An example config file can be found at `configs/example.json` and below. `capture.regions` parameters can be written using the region tool:
```bash
python src\run.py <config file> --region-tool --display <display number>
```
This tool takes a screenshot of the window and allows you to select regions using the mouse. The region parameter values are then printed to the console.

The following outlines the required and optional paremeters of a config file:

### Required Parameters
These parameters must be specified:
- `game_type: 'comp' | 'scrim' | 'ranked' | 'standard' | 'custom'`
- `spectator: bool` - specifying if the game perspective is in spectator mode or in-person.
- `team0: list[str]{0-5}` - A list of max 5 in-game names (IGNS) on team 0 (left hand side), igns will be inferred if not specified
- `team1: list[str]{0-5}` - A list of max 5 in-game names (IGNS) on team 1 (right hand side), igns will be inferred if not specified
- `capture.mode: 'screenshot' | 'videofile'` - 
- `capture.regions.timer: list[int,int,int,int]` - A list of 4 integers specifying the region of the game window where the timer is located.
- `capture.regions.kf_line: list[int,int,int,int]` - A list of 4 integers specifying the region of the first line in the kill feed. The region should be filled by the IGN boxes, but should extend leftwards to be able to fit in longer names.

### Inferred Parameters
These parameters are optional and will be inferred by R6Analyser if not explicitly specified:
- `max_rounds: int`
- `rounds_per_side: int`

### Optional Parameters
These parameters are optional and will default to values in `default.json` if not specified:
- `SCREENSHOT_RESIZE`: (default=4) A number specifying the factor by which the screenshot is resized before OCR-processing.
- `SCREENSHOT_PERIOD`: (default=0.5) This number determines how frequently the program captures the game feed for analysis, the period in seconds between screenshots


### Config File Example
Below is an example configuration file that specifies a set of possible parameters for a standard game:
```json
{
    "game_type": "SCRIM",
    "spectator": false,
    "capture": {
        "mode": "screenshot",
        "regions": {
            "timer": [1210, 110, 140, 65],
            "kf_line": [1705, 413, 605, 31]
        }
    },
    "team0": [
        "Samba.",
        "Player2",
        "Player3",
        "Player4",
        "Player5"
    ],
    "team1": [
        "Shaiiko.BDS",
        "BriD.BDS",
        "LikEfac.BDS",
        "Solotov.BDS",
        "Yuzus.BDS"
    ],
    "MAX_ROUNDS": 12,
    "SCREENSHOT_PERIOD": 0.5
}
```
With this config file, the remaining 5 player IGNs will be inferred from the game feed. Since standard only has 1 round of overtime, the `MAX_ROUNDS` and `ROUNDS_PER_SIDE` needed to be specified as the inferred `(MAX_ROUNDS-3) / 2` assumes a normal 3 round overtime.

## Process
### CaptureMode = SCREENSHOT
The program works by taking screenshots of the game window at regular intervals `SCREENSHOT_PERIOD` and using OCR to extract the necessary information. Currently, 6 regions of the screenshot are used for OCR:
- **Timer**: Used to extract the round timer.
- **Kill Feed**: Used to extract the kill feed.
- **Team 1 Score**: Used to extract team 1's score.
- **Team 2 Score**: Used to extract team 2's score.
- **Team 1 Side**: Used to extract team 1's side.
- **Team 2 Side**: Used to extract team 2's side.
Although, only the Timer and Kill Feed regions are used every inference cycle, the other regions are used to infer the game state and are only used when a new round is detected.

## How to build
To build R6Analyser, run:
```ps1
.\scripts\build.ps1
.\scripts\build.sh
```

The build process goes as follows:
1a. Combine all source code files into a single file with some details removed
1b. Convert combined file into a bytecode .pyc file
2. Run PyInstaller on combined.pyc and a run.py file (which calls combined.main())

Steps 1a, 1b is achieved using the `combined-script.py` and providing a `build_config.json`.
Steps 2 needs a `build.spec` file which specifies how to build into a `.exe`

## Requirements
This program uses:
- [Python](https://www.python.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) to perform the necessary OCR for information gathering.
- [Cuda](https://developer.nvidia.com/cuda-toolkit) for GPU acceleration.
- [OpenCV](https://opencv.org/) for other computer vision tasks.
- [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) to take screenshots of the game window.
- [Numpy](https://numpy.org/) for matrix operations.
and a few other packages which can be found in the `requirements.txt` file.