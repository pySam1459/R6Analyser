#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$script_dir/.."

if [ -d "build" ]; then
  rm -rf "build"
fi
if [ -d "dist" ]; then
  rm -rf "dist"
fi

if [ -d "venv-prod/Scripts" ] && [ -d "venv/Scripts" ]; then
    VENV_PROD_BIN="venv-prod/Scripts"
    VENV_DEV_BIN="venv/Scripts"
elif [ -d "venv/bin" ]; then
    VENV_PROD_BIN="venv-prod/bin"
    VENV_DEV_BIN="venv/bin"
else
    echo "Error: Cannot find 'Scripts' or 'bin' directory in the virtual environment."
    exit 1
fi

source "$VENV_DEV_BIN/activate"
echo "Running tests..."
"$VENV_DEV_BIN/pytest.exe"

deactivate
source "$VENV_PROD_BIN/activate"

echo "Running build scripts..."
"$VENV_PROD_BIN/pyinstaller.exe" build.spec
"$VENV_PROD_BIN/python.exe" scripts/bundle.py -c bundle.json -b build/bundle -d dist -O dist/R6Analyser.zip

deactivate