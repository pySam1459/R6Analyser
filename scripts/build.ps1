$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Split-Path -Parent $script_dir)

# Determine the venv bin
if ((Test-Path "venv-prod\Scripts") -And (Test-Path "venv\Scripts")) {
    $VENV_DEV_BIN = "venv\Scripts"
    $VENV_PROD_BIN = "venv-prod\Scripts"
}
elseif ((Test-Path "venv-prod\bin") -And (Test-Path "venv\bin")) {
    $VENV_DEV_BIN = "venv\bin"
    $VENV_PROD_BIN = "venv-prod\bin"
}
else {
    Write-Error "Error: Cannot find 'Scripts' or 'bin' directory in the virtual environment."
    exit 1
}

## tests my be run with the dev venv, not prod
& "$VENV_DEV_BIN\Activate.ps1"

Write-Host "Running tests..."
& "$VENV_DEV_BIN\pytest.exe" --cov=src

deactivate
& "$VENV_PROD_BIN\Activate.ps1"

Write-Host "Running build script..."
& "$VENV_PROD_BIN\pyinstaller.exe" build.spec
& "$VENV_PROD_BIN\python.exe" scripts\build.py -c build_config.json -b build\bundle -d dist -O dist\R6Analyser.zip

deactivate