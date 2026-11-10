$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ first.' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git from https://git-scm.com/ first.' }
uv sync --locked --python 3.11
if ($LASTEXITCODE -ne 0) { throw 'Failed to install pixcli dependencies.' }

& (Join-Path $PSScriptRoot 'scripts/setup-animation.ps1')

$cli = Join-Path $PSScriptRoot '.venv/Scripts/pixcli.exe'
foreach ($profile in @('image', 'animation')) {
    Write-Host "Downloading and verifying $profile models..."
    & $cli models download -profile $profile
    if ($LASTEXITCODE -ne 0) { throw "Failed to download or verify $profile models. Resolve the error and rerun setup." }
}
Write-Host 'Setup complete. Activate the environment and run pixcli doctor.'
