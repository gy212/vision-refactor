# Start local Qwen3-VL llama-server for student practice (vision coach).
# Does not modify the vision repo models/ directory; GGUF lives outside the repo.
#
# Defaults (override with env):
#   QWEN_ROOT     = E:\AI\qwen3-vl
#   QWEN_PORT     = 8091
#   QWEN_HOST     = 127.0.0.1
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_qwen_server.ps1

$ErrorActionPreference = "Stop"

$Root = if ($env:QWEN_ROOT) { $env:QWEN_ROOT } else { "E:\AI\qwen3-vl" }
$HostAddr = if ($env:QWEN_HOST) { $env:QWEN_HOST } else { "127.0.0.1" }
$Port = if ($env:QWEN_PORT) { [int]$env:QWEN_PORT } else { 8091 }

$Runtime = Join-Path $Root "runtime"
$Models = Join-Path $Root "models"
$ServerExe = Join-Path $Runtime "llama-server.exe"
$Model = Join-Path $Models "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
$Mmproj = Join-Path $Models "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"

if (-not (Test-Path $ServerExe)) {
    Write-Error "llama-server not found: $ServerExe"
}
if (-not (Test-Path $Model)) {
    Write-Error "model not found: $Model"
}
if (-not (Test-Path $Mmproj)) {
    Write-Error "mmproj not found: $Mmproj"
}

Write-Host "Starting Qwen3-VL llama-server on http://${HostAddr}:${Port}"
Write-Host "  model : $Model"
Write-Host "  mmproj: $Mmproj"
Write-Host "Health : http://${HostAddr}:${Port}/health"
Write-Host "Press Ctrl+C to stop."

# CUDA DLLs live next to the server binary.
$env:PATH = "$Runtime;$env:PATH"

& $ServerExe `
    -m $Model `
    --mmproj $Mmproj `
    --host $HostAddr `
    --port $Port `
    -ngl 99 `
    --ctx-size 8192
