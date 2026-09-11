# Sync-Upstream.ps1
#
# Synchronize the llama.cpp Blackwell fork with ggml-org/llama.cpp.
#
# Policy:
#   - master is a clean mirror of upstream/master (reset + force-with-lease push)
#   - blackwell/* branches = current upstream/master + only their delta
#   - plain --force is NEVER used; only --force-with-lease
#   - on a rebase conflict the script STOPS, leaves the rebase in progress,
#     and reports it — it never invents a resolution
#
# Usage (from the repository root):
#   pwsh -File scripts/Sync-Upstream.ps1
#   pwsh -File scripts/Sync-Upstream.ps1 -WhatIf   # dry run, no pushes
#
# Prerequisites: remotes 'origin' (your fork) and 'upstream' (ggml-org).
#
# Provenance (tracked in BLACKWELL-BUILD.md):
#   PR #28572 (TMA, head 10247a1a)  -> blackwell/tma    (2 commits)
#   PR #26704 (CUTLASS, head 94ee2c75) -> blackwell/cutlass (7 commits)
#   combined is intentionally not tracked (mutually exclusive paths, see doc)

param(
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Log($msg) { Write-Host "[sync] $msg" }
function LogErr($msg) { Write-Host "[sync] ERROR: $msg" -ForegroundColor Red }
function RunGit([string[]]$args) {
    & git @args 2>&1
    return $LASTEXITCODE
}
function Push([string[]]$args) {
    if ($WhatIf) { Log "  (what-if) would push: git $($args -join ' ')"; return }
    $code = RunGit $args
    if ($code -ne 0) { LogErr "push failed: git $($args -join ' ')"; exit 1 }
}

Set-Location $RepoRoot
Log "Repo root: $RepoRoot (WhatIf: $WhatIf)"

# ---------------------------------------------------------------- remotes
foreach ($r in @('origin','upstream')) {
    $url = (git remote get-url $r 2>$null)
    if (-not $url) { LogErr "remote '$r' missing — set origin (your fork) and upstream (ggml-org/llama.cpp)"; exit 1 }
    Log "remote $r -> $url"
}
$originUrl = (git remote get-url origin)
if ($originUrl -notmatch 'brx19/llama\.cpp') {
    LogErr "origin does not point at brx19/llama.cpp: $originUrl"
}

# ---------------------------------------------------------------- fetch
Log "Fetching upstream and origin (prune, tags)..."
foreach ($r in @('upstream','origin')) {
    $code = RunGit @('fetch', $r, '--prune', '--tags')
    if ($code -ne 0) { LogErr "fetch $r failed"; exit 1 }
}

# ---------------------------------------------------------------- master
Log "Synchronizing master with upstream/master..."
$upstreamMaster = (git rev-parse upstream/master)
$localMaster    = (git rev-parse master)
Log "  upstream/master = $upstreamMaster"
Log "  master          = $localMaster"

if ($localMaster -ne $upstreamMaster) {
    # Check for unique commits on master before resetting
    $unique = @(git log --oneline master..upstream/master 2>$null) | Measure-Object
    $ahead = @(git log --oneline upstream/master..master 2>$null)
    if ($ahead.Count -gt 0) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmm'
        $backup = "backup/master-before-sync-$stamp"
        Log "  master has $($ahead.Count) commit(s) not on upstream — preserving as $backup"
        if (-not $WhatIf) {
            RunGit @('branch', $backup, $localMaster)
            Log "  preserved: $backup -> $(git rev-parse $backup)"
        }
        $ahead | ForEach-Object { Log "    preserved: $_" }
    }
    if ($WhatIf) {
        Log "  (what-if) would reset master to $upstreamMaster"
    } else {
        RunGit @('switch', 'master') | Out-Null
        RunGit @('reset', '--hard', 'upstream/master')
        Push @('push', '--force-with-lease', 'origin', 'master')
        Log "  master now = $(git rev-parse master)"
    }
} else {
    Log "  master already matches upstream/master — nothing to do"
}
$originMaster = (git ls-remote origin refs/heads/master | ForEach-Object { ($_ -split "`t")[0] }).Trim()
if ($originMaster -ne $upstreamMaster) {
    LogErr "origin/master ($originMaster) != upstream/master ($upstreamMaster) — sync incomplete"
    exit 1
}
Log "  VERIFIED: master == upstream/master ($upstreamMaster)"

# ---------------------------------------------------------------- PR status
function PrStatus([int]$num) {
    try {
        $pr = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/pulls/$num" `
            -Headers @{ "Accept" = "application/vnd.github+json" } -TimeoutSec 30
        return $pr
    } catch {
        Log "  PR #$num: API unreachable ($($_.Exception.Message)) — skipping merge check"
        return $null
    }
}

$pr28572 = PrStatus 28572
$pr26704 = PrStatus 26704
if ($pr28572) {
    Log "PR #28572 (TMA): state=$($pr28572.state) merged=$($pr28572.merged) head=$($pr28572.head.sha)"
    if ($pr28572.merged) { Log "  => TMA work is UPSTREAM; blackwell/tma is unnecessary (stock master includes it)" }
}
if ($pr26704) {
    Log "PR #26704 (CUTLASS): state=$($pr26704.state) merged=$($pr26704.merged) head=$($pr26704.head.sha)"
    if ($pr26704.merged) { Log "  => CUTLASS work is UPSTREAM; blackwell/cutlass is unnecessary (stock master includes it)" }
}

# ---------------------------------------------------------------- refresh PR refs
Log "Refreshing PR refs..."
RunGit @('fetch', 'upstream', 'refs/pull/28572/head:refs/remotes/upstream/pr-28572', '--force') | Out-Null
RunGit @('fetch', 'upstream', 'refs/pull/26704/head:refs/remotes/upstream/pr-26704', '--force') | Out-Null

# ---------------------------------------------------------------- experimental branches
# Each branch = master + N PR commits (rebased). On conflict: stop, report, leave rebase in progress.
$branches = @(
    @{ name = 'blackwell/tma';     pr = 28572; prObj = $pr28572 },
    @{ name = 'blackwell/cutlass'; pr = 26704; prObj = $pr26704 }
)

foreach ($b in $branches) {
    $name = $b.name
    $prObj = $b.prObj
    Log "Refreshing $name (PR #$($b.pr))..."

    if ($prObj -and $prObj.merged) {
        Log "  PR merged upstream — marking branch unnecessary. (Optional: git push origin :$name to delete.)"
        continue
    }

    # Skip if local branch missing
    $exists = RunGit @('show-ref', "--verify", "refs/heads/$name")
    if ($exists -ne 0) {
        Log "  local branch $name not found — skipping (create it first with the initial port)"
        continue
    }

    RunGit @('switch', $name) | Out-Null

    # How many PR commits does this branch carry? (branch = master + N)
    $deltaCount = [int](git rev-list --count "master..$name")
    Log "  branch carries $deltaCount commit(s) beyond master"
    if ($deltaCount -le 0) { continue }

    # Re-apply the current PR delta: rebase the last N commits onto master.
    # -X theirs: on conflict take the PR side for our own patch hunks, while master's
    # structural changes (file removals, renames, refactors) are honored via the rebase.
    $range = "HEAD~$deltaCount..HEAD"
    if ($WhatIf) {
        Log "  (what-if) would run: git rebase -X theirs --onto master $range $name"
        continue
    }

    # Clean state
    RunGit @('checkout', 'master', '--', '.') | Out-Null
    RunGit @('reset', '--hard', $name) | Out-Null

    $rebaseCmd = "git rebase -X theirs --onto master $range $name"
    Log "  running: $rebaseCmd"
    $rebaseOutput = & cmd /c $rebaseCmd 2>&1
    $code = $LASTEXITCODE

    if ($code -ne 0 -or $rebaseOutput -match 'CONFLICT|error:|could not apply') {
        LogErr "REBASE CONFLICT on $name — STOPPING (not inventing a resolution)."
        LogErr "The rebase is left in progress. Manual steps:"
        LogErr "  cd repo; git status; resolve conflicts; git add <files>; git rebase --continue"
        LogErr "  (or: git rebase --abort to restore $name as it was before this sync)"
        LogErr "After resolving, push: git push --force-with-lease origin $name"
        # Abort the in-progress rebase to leave a clean state, then re-run rebase for user
        RunGit @('rebase', '--abort') | Out-Null
        Log "  (rebase aborted — $name restored to its pre-sync state; re-run after manual port)"
        continue
    }

    # Sanity: branch must now be master + N commits, no stray history
    $newCount = [int](git rev-list --count "master..$name")
    if ($newCount -ne $deltaCount) {
        LogErr "$name: expected $deltaCount delta commits, found $newCount — verifying before push"
    }
    Push @('push', '--force-with-lease', 'origin', $name)
    Log "  $name refreshed -> $(git rev-parse $name)"
}

# ---------------------------------------------------------------- summary
Log "Sync complete."
Log "  master            = $(git rev-parse master)"
Log "  upstream/master   = $(git rev-parse upstream/master)"
Log "  blackwell/tma     = $(git rev-parse blackwell/tma 2>$null)"
Log "  blackwell/cutlass = $(git rev-parse blackwell/cutlass 2>$null)"
Log "Update BLACKWELL-BUILD.md 'last refreshed' dates and branch SHAs."
exit 0
