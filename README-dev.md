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
