param(
    [string]$InputImage = 'outputs/smoke/knight.png',
    [string]$OutputDirectory = 'outputs/smoke/walk',
    [ValidateSet('fp8', 'q3')]
    [string]$TextEncoder = 'q3',
    [ValidateSet('q2', 'q3')]
    [string]$ExpertQuant = 'q2',
    [ValidateSet('low-vram', 'gpu-resident')]
    [string]$Preset = 'low-vram',
    [ValidateSet('cpu', 'cuda')]
    [string]$AdapterDevice = 'cpu',
    [ValidateSet('normal', 'mmap')]
    [string]$WeightStorage = 'mmap',
    [double]$MaxPrivateGiB = 14,
    [double]$MinFreeCommitGiB = 2,
    [double]$MinFreePhysicalGiB = 2
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$cli = Join-Path $PWD '.venv\Scripts\pixcli.exe'
$python = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $cli)) { throw 'Run .\setup.ps1 first.' }
if (-not (Test-Path -LiteralPath $InputImage)) { throw "Input image not found: $InputImage" }
if (Test-Path -LiteralPath $OutputDirectory) { throw "Output directory already exists: $OutputDirectory. Choose a new -OutputDirectory." }
$profile = if ($ExpertQuant -eq 'q2' -and $TextEncoder -eq 'q3') { 'animation' } elseif ($ExpertQuant -eq 'q2') { 'animation-q2-experts' } elseif ($TextEncoder -eq 'q3') { 'animation-q3-text' } else { 'animation-q3-fp8' }
& $cli models verify --profile $profile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cli animate --input $InputImage --prompt 'the knight walks in place, fixed camera' --out $OutputDirectory --seed 42 --preset $Preset --text-encoder $TextEncoder --expert-quant $ExpertQuant --adapter-device $AdapterDevice --weight-storage $WeightStorage --max-private-gib $MaxPrivateGiB --min-free-commit-gib $MinFreeCommitGiB --min-free-physical-gib $MinFreePhysicalGiB --inference-width 256 --inference-height 256 --frames 5 --fps 8 --pixel-width 96 --pixel-height 96 --offline
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/verify_animation.py $OutputDirectory
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Animation export checks passed. Inspect the original frames, preview and sprite sheet visually.'
