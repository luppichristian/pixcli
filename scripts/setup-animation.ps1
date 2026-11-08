$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectDir '.backend'
$comfyDir = Join-Path $backendDir 'ComfyUI'
$pythonExe = Join-Path $backendDir '.venv/Scripts/python.exe'
$lockFile = Join-Path $projectDir 'backend/requirements.lock'
$readyFile = Join-Path $backendDir 'ready.json'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required; install uv, then rerun this script.' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git is required; install Git, then rerun this script.' }
if (-not (Test-Path -LiteralPath $lockFile)) { throw 'The pinned ComfyUI dependency lock is missing; backend setup cannot continue.' }
New-Item -ItemType Directory -Force -Path $backendDir | Out-Null
if (Test-Path -LiteralPath $readyFile) { Remove-Item -LiteralPath $readyFile -Force }
function Checkout-Pinned([string]$url, [string]$path, [string]$revision) {
    if (-not (Test-Path -LiteralPath $path)) {
        & git clone --no-checkout $url $path
        if ($LASTEXITCODE -ne 0) { throw "Unable to clone $url" }
        & git -C $path checkout --detach $revision
        if ($LASTEXITCODE -ne 0) { throw "Unable to check out $revision in $path" }
    } else {
        $actual = (& git -C $path rev-parse HEAD 2>$null)
        if ($LASTEXITCODE -ne 0 -or $actual -ne $revision) { throw "Existing checkout $path is not at the required revision $revision; preserve it and choose a fresh backend directory." }
        $dirty = (& git -C $path status --porcelain --untracked-files=no)
        if ($LASTEXITCODE -ne 0 -or $dirty) { throw "Existing checkout $path has modified tracked files; refusing to modify it." }
    }
}
Checkout-Pinned 'https://github.com/Comfy-Org/ComfyUI.git' $comfyDir '63e5bf594339f676b4b15e072529873e3448409e'
Checkout-Pinned 'https://github.com/city96/ComfyUI-GGUF.git' (Join-Path $comfyDir 'custom_nodes/ComfyUI-GGUF') '6ea2651e7df66d7585f6ffee804b20e92fb38b8a'
Checkout-Pinned 'https://github.com/princepainter/ComfyUI-PainterI2V.git' (Join-Path $comfyDir 'custom_nodes/ComfyUI-PainterI2V') '714044a6536ad37293f0658e247c5c6be1622745'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & uv venv --python 3.11 (Join-Path $backendDir '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the isolated backend environment.' }
}
$env:UV_LOCK_TIMEOUT = '1800'
& uv pip sync --python $pythonExe $lockFile --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'Failed to install locked backend dependencies.' }
& uv pip check --python $pythonExe
if ($LASTEXITCODE -ne 0) { throw 'The locked backend dependencies are incompatible; no readiness marker was written.' }
$nodeDir = Join-Path $comfyDir 'custom_nodes/pixcli_wan'
New-Item -ItemType Directory -Force -Path $nodeDir | Out-Null
Copy-Item -LiteralPath (Join-Path $projectDir 'pixcli/comfy_custom_nodes/pixcli_wan/__init__.py') -Destination (Join-Path $nodeDir '__init__.py') -Force
Copy-Item -LiteralPath (Join-Path $projectDir 'pixcli/comfy_custom_nodes/pixcli_wan/mmap_loader.py') -Destination (Join-Path $nodeDir 'mmap_loader.py') -Force
$ready = @{
    schema_version = 1
    lock_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $lockFile).Hash.ToLowerInvariant()
    node_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $nodeDir '__init__.py')).Hash.ToLowerInvariant()
    mmap_node_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $nodeDir 'mmap_loader.py')).Hash.ToLowerInvariant()
    commits = @{
        ComfyUI = '63e5bf594339f676b4b15e072529873e3448409e'
        'ComfyUI-GGUF' = '6ea2651e7df66d7585f6ffee804b20e92fb38b8a'
        'ComfyUI-PainterI2V' = '714044a6536ad37293f0658e247c5c6be1622745'
    }
}
$readyTemp = Join-Path $backendDir 'ready.json.tmp'
$ready | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 -LiteralPath $readyTemp
Move-Item -LiteralPath $readyTemp -Destination $readyFile -Force
Write-Host 'Animation backend installed.'

