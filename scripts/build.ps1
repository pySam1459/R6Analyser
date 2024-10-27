# Stop execution immediately when a command fails
$ErrorActionPreference = "Stop"

# Get the directory of the script
$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $script_dir)

# Determine the virtual environment bin directory
if (Test-Path "venv-prod\Scripts") {
    $VENV_BIN = "venv-prod\Scripts"
}
elseif (Test-Path "venv-prod\bin") {
    $VENV_BIN = "venv-prod\bin"
}
else {
    Write-Error "Error: Cannot find 'Scripts' or 'bin' directory in the virtual environment."
    exit 1
}

# Activate the virtual environment
& "$VENV_BIN\Activate.ps1"

Write-Host "Running tests..."

& "$VENV_BIN\pytest.exe"

Write-Host "Running build script..."

# Run PyInstaller
& "$VENV_BIN\pyinstaller.exe" --onefile "src/run.py"

# Run publish script
& "$VENV_BIN\python.exe" scripts/build.py -c build_config.json -b build/bundle -d dist -O dist/R6Analyser.zip

# Deactivate the virtual environment
deactivate