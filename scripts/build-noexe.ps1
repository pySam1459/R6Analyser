# Stop execution immediately when a command fails
$ErrorActionPreference = "Stop"

# Get the directory of the script
$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $script_dir)

# Determine the virtual environment bin directory
if (Test-Path "venv\Scripts") {
    $VENV_BIN = "venv\Scripts"
}
elseif (Test-Path "venv\bin") {
    $VENV_BIN = "venv\bin"
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

# Run the combine script
& "$VENV_BIN\python.exe" scripts\combine-script.py -b build_config.json -O build\bundle

# Run publish script
& "$VENV_BIN\python.exe" scripts\publish.py -b build_config.json -f build\bundle -O dist\R6Analyser.zip -v1

# Deactivate the virtual environment
deactivate