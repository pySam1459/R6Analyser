#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$script_dir/.."

if [ -d "venv/Scripts" ]; then
    VENV_BIN="venv-prod/Scripts"
elif [ -d "venv/bin" ]; then
    VENV_BIN="venv-prod/bin"
else
    echo "Error: Cannot find 'Scripts' or 'bin' directory in the virtual environment."
    exit 1
fi

source "$VENV_BIN/activate"

echo "Running tests..."

"$VENV_BIN/pytest.exe"

echo "Running build script..."

"$VENV_BIN/pyinstaller.exe" --onefile "src/run.py"

"$VENV_BIN\python.exe" scripts/build.py -c build_config.json -b build/bundle -d dist -O dist/R6Analyser.zip

deactivate