param([string]$OutputDirectory = 'outputs/smoke')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$cli = Join-Path $PWD '.venv\Scripts\pixcli.exe'
$python = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path $cli)) { throw 'Run .\setup.ps1 first.' }
if (Test-Path -LiteralPath $OutputDirectory) { throw "Output directory already exists: $OutputDirectory. Choose a new -OutputDirectory." }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$source = Join-Path $OutputDirectory 'knight.png'
$edit = Join-Path $OutputDirectory 'edit.png'
$mask = Join-Path $OutputDirectory 'mask.png'
$masked = Join-Path $OutputDirectory 'masked.png'
& $cli doctor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cli models verify
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cli gen --prompt 'pixel art, a knight with blue armor, centered, plain background' --out $source --seed 42 --steps 8 --inference-width 768 --inference-height 768 --pixel-width 96 --pixel-height 96 --save-original --offline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cli edit --input $source --prompt 'pixel art, the same knight with red armor' --strength 0.5 --out $edit --seed 42 --steps 8 --inference-width 768 --inference-height 768 --offline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/smoke_mask.py mask $source $mask
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cli edit --input $source --mask $mask --prompt 'pixel art, red armor' --strength 0.5 --out $masked --seed 42 --steps 8 --inference-width 768 --inference-height 768 --offline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/smoke_mask.py verify $source $mask $masked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Image smoke commands completed. Visually inspect the outputs. Animation is tested separately.'
