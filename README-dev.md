# R6Analyser
### Dependencies
To run the program, you will need to have Python (3.10>=) installed on your system. You can download Python from the [official website](https://www.python.org/downloads/).

Before running the program, you will need to install the required packages using the following command:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
TesserOCR must be installed separately using the .whl file they provide [here](https://github.com/simonflueckiger/tesserocr-windows_build/releases).
```bash
pip install <package_name>.whl
```

You can check if all dependencies have been successfully installed using:
```bash
python src\run.py --check
```


### Tests
Modify `addopts = -v -s` to `addopts = -v -s --cov=src` to include coverag


## Process
### CaptureMode = SCREENSHOT
The program works by taking screenshots of the game window and using OCR to extract the necessary information. Currently, `5+n` regions of the screenshot are used for OCR:
- **Timer**: Used to extract the round timer.
- **Kill Feed Lines**: Used to extract the kill feed using `n` kill feed lines.
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
- [OpenCV](https://opencv.org/) for other computer vision tasks.
- [Numpy](https://numpy.org/) for matrix operations.
- [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) to perform the necessary OCR for information gathering. [Windows Download](https://github.com/UB-Mannheim/tesseract/wiki)
- [PyAutoGui](https://pyautogui.readthedocs.io/en/latest/) to take screenshots of the game window.
and a few other packages which can be found in the `requirements.txt` file.
