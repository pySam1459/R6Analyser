# R6Analyser

This program performs real-time data extraction from a [Rainbow Six Siege](https://www.ubisoft.com/en-gb/game/rainbow-six/siege) VOD or live game for statistical analysis.</br>
The following information gathered per round includes:
- **Score Line**
- **Kill Feed**, **Traded Kills**, **Traded Deaths**, **Refragged Kills**
- **Bomb Plant At**, **Disabled Defuser At**
- **Round End At**
- **Win Condition**
- **Opening Kill**, **Opening Death**
- **Atk/Def Side**
And a match stats summary page:


## How to run
### Dependencies
You can check if all dependencies have been successfully installed using:
```bash
./R6Analyser --check
```

### Configuration
You must also have made a configuration file (see below) to specify certain regions of the game window and other parameters. 
To check the regions defined in the configuration file, you can run the following command:
```bash
./R6Analyser -c <config file> --check-regions
```
which will save the screenshots of the regions defined into a new `images` directory. If these screenshot's do not contain the relevant information, they should be re-defined.

### Run
To run the program, use the following command:
```bash
./R6Analyser -c <config file>
```

### All CLI args
- `-c` / `--config`: JSON configuration file located in . or ./configs
- `-k` / `--key`: Software Key, 64 hex key
- `-v` / `--verbose`: Print additional information to the console (0-3).
- `--check-regions`: (Debugging) Saves the capture regions as images for quality check
- `--test-regions`: (Debugging) Runs the OCR engine for a single instance of each capture region and prints the output to the console.
- `--region-tool`: Runs the region tool used to find config region parameters.
- `--check`: Runs a script to check program validity and function
- `--region-tool`: Runs the Region tool instead of R6Analyser.
- `-d` / `--delay`: The delay in seconds before the program starts capturing the game window.
- `--display`: The display number to take the screenshot from (default=0).

### Software Key
A software key is required to use R6Analyser. It will be a hex string of length 64.
E.g. `431739c9f591b2630503341e6f84afa7f9f60580e0de4d59abedfeed3262387f`

A key should generated when you initially acquire the program.
A key can be specified by:
- `.env` file, with field `SOFTWARE_KEY="<KEY>"`
- `-k` / `--key` cli argument

## How to use
This program uses Optical Character Recognition (OCR) to extract text information from the game window, so it is important to have a good quality video and high resolution. If you cannot read the text on the screen, the program will not be able to either. That being said, the program can handle relatively low quality (720p).</br>
tl;dr The better the quality of the video/game, the more accurate the program will be.</br>

## Config
To use R6Analyser, a configuration JSON file is required to specify regions of the game window and other program settings. An example config file can be found at `configs/example.json` and below.

### Regions
Regions are areas of the screen which R6Analyser captures and extracts information from. A region parameter contains 4 numbers `[x-coord,y-coord,width,height]` with the origin (x,y = 0,0) in the top left corner of the screen.

These parameters are defined in the `capture.regions` sub-config:
```json
{
    "capture": {
        "regions": {
            "region1": [x,y,w,h],
            "region2": [x,y,w,h],
        }
    }
}
```
R6Analyser provides a useful tool to create and add these regions to a config, 
```bash
./R6Analyser -c <config file> --region-tool
```
This tool takes a screenshot of the window and allows you to select regions using the mouse.


### Parameters

The following outlines the required and optional paremeters of a config file:

### Recommended Parameters
- `name: str` - name identifier of the match being analysed

### Required Parameters
These parameters must be specified:
- `game_type: 'comp' | 'scrim' | 'ranked' | 'standard' | 'custom'`
- `spectator: bool` - specifying if the game perspective is in spectator mode or in-person.
- `team0: list[str]{0-5}` - A list of max 5 in-game names (IGNS) on team 0 (left hand side), igns will be inferred if not specified
- `team1: list[str]{0-5}` - A list of max 5 in-game names (IGNS) on team 1 (right hand side), igns will be inferred if not specified
- `capture.mode: 'screenshot' | 'videofile'` - 
- `capture.regions.timer: list[int,int,int,int]` - A list of 4 integers specifying the region of the game window where the timer is located.
- `capture.regions.kf_line: list[int,int,int,int]` - A list of 4 integers specifying the region of the first line in the kill feed. The region should be filled by the IGN boxes, but should extend leftwards to be able to fit in longer names.

# Depdendent Parameters
- `capture.file` | `capture.mode == 'videofile'` - the file path to the video being analysed

### Custom Game Type Parameters
These parameters are inferred by the game_type, but must be provided if `game_type == "custom"`
- `max_rounds: int`
- `rounds_per_side: int`
- `overtime_rounds: int`

### Optional Parameters
These parameters are optional and will default to values in `default.json` if not specified:
- `capture.scale_by`: (default=2) A number specifying the factor by which each region is resized before OCR-processing.
- `capture.period`: (default=0.5) This number determines how frequently the program captures the game feed for analysis, the period in seconds between region analysis


### Config File Example
Below is an example configuration file that specifies a set of possible parameters for a standard game:
```json
{
    "name": "Example Config",
    "game_type": "scrim",
    "spectator": false,
    "capture": {
        "mode": "screenshot",
        "regions": {
            "timer": [1210, 110, 140, 65],
            "kf_line": [1705, 413, 605, 31]
        },
        "period": 0.25
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
    "save": {
        "path": "/some/path/here.xlsx"
    }
}
```

## Process
### CaptureMode = SCREENSHOT
The program works by taking screenshots of the game window at regular intervals `capture.period` and using OCR to extract the necessary information. Currently, 6 regions of the screenshot are used for OCR:
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
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) to perform the necessary OCR for information gathering. [Windows Download](https://github.com/UB-Mannheim/tesseract/wiki)
- [OpenCV](https://opencv.org/) for other computer vision tasks.
- [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) to take screenshots of the game window.
- [Numpy](https://numpy.org/) for matrix operations.
and a few other packages which can be found in the `requirements.txt` file.