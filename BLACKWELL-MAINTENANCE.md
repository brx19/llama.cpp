# RTX 5090 fork maintenance

`main` contains upstream source plus the fork's build/sync tooling. The only experimental branches are `blackwell/cpasync`, `blackwell/cutlass`, and `blackwell/tma`.

## Sync

`Sync upstream and Blackwell branches` runs daily at 07:00 UTC and can be run manually. It fetches only upstream `master`, without tags. It merges without force pushes and retains the fork's complete workflow directory. A source-parity check prevents unnoticed source changes on `main`.

Each experimental branch is updated independently from its current tip, preserving Windows runtime fixes. Workflow conflicts are resolved by keeping fork tooling. Source conflicts stop only the affected branch, leave its remote tip intact, and appear as a failed job. They must be reviewed; the sync never chooses an arbitrary side of a CUDA conflict.

On Windows, from the repository root:

```powershell
.\scripts\Sync-Upstream.ps1 -WhatIf
```

Omit `-WhatIf` to push successful merges. Python 3.9+ and Git are required. Existing working-tree files are not changed; merges use temporary worktrees.

## Build

Run `Build RTX 5090 Windows packages` on `main`, selecting `all` or one variant. The workflow resolves branch tips once, then builds the CPU tools and CUDA backend from each exact same SHA. CUDA 13.3 and `120a-real` retain the previous comparison baseline. CUTLASS is enabled only for its variant.

Download the final `llama-win-cuda13.3-sm120a-<variant>-<sha>` artifact, not intermediate CPU/CUDA artifacts. These packages still require runtime testing on the local RTX 5090.

## Archived branches

The one-time cleanup only removes branch/SHA pairs recorded in `.github/blackwell-branch-archive.json`. It first preserves each tip in `archive/branches-20260921/<old-branch>` tags in the same atomic push. Branches created later or changed since review are never silently removed. The current default branch is always protected.

Set the GitHub default branch to `main` before removing the former `master`. The archive workflow can be run again after that change. It has no schedule.

Old release tags are not changed. In particular, tag collisions such as `b10932` cannot block sync because upstream tags are never fetched.
