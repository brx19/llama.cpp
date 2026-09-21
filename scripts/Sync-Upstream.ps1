param([switch]$WhatIf)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $python = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'py' }
    $syncArgs = @('.github/scripts/sync_blackwell.py')
    if (-not $WhatIf) { $syncArgs += '--push' }
    $failed = @()
    foreach ($branch in @('main', 'blackwell/cpasync', 'blackwell/cutlass', 'blackwell/tma')) {
        & $python @syncArgs --branch $branch
        if ($LASTEXITCODE -ne 0) {
            $failed += $branch
            if ($branch -eq 'main') { break }
        }
    }
    if ($failed.Count) { throw "Sync failed for: $($failed -join ', '). See the errors above." }
} finally {
    Pop-Location
}
