#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$script_dir/.."

if [ -d "venv/Scripts" ]; then
    VENV_BIN="venv/Scripts"
elif [ -d "venv/bin" ]; then
    VENV_BIN="venv/bin"
else
    echo "Error: Cannot find 'Scripts' or 'bin' directory in the virtual environment."
    exit 1
fi

source "$VENV_BIN/activate"

echo "Running build script..."

"$VENV_BIN/python.exe" scripts/combine-script.py -b build_config.json  -O combined

"$VENV_BIN/pyinstaller.exe" build.spec

"$VENV_BIN\python.exe" scripts/publish.py -b build_config.json -f dist/R6Analyser -O dist/R6Analyser.zip -v1

deactivate