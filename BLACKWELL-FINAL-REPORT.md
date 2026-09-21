# BLACKWELL LLAMA.CPP FORK — FINAL STATUS REPORT
Date: 2026-09-11 | Operator: Hermes (brx19 / BrX) | Upstream: ggml-org/llama.cpp

## 1. GITHUB
- Fork: **brx19/llama.cpp** (public), default branch `master`
- Origin (fork): `https://github.com/brx19/llama.cpp`
- Upstream: `https://github.com/ggml-org/llama.cpp` (remote `upstream`)
- Visibility: **public** (standard GitHub-hosted runners, no paid private minutes)
- Auth: HTTPS with PAT from `~/.git-credentials` (SSH not authorized on this WSL box)

## 2. MASTER BRANCH
- Fork `master` HEAD: `ba8a244d5808`
- `upstream/master` HEAD: `5bda51bfbc62` (advanced since fork was built)
- Policy (user-confirmed): master = **upstream source + CI tooling**. All source files byte-identical to upstream; only `.github/`, `scripts/`, `BLACKWELL-BUILD.md` are additions (source parity, NOT SHA equality).
- **Sync pending**: upstream moved `16378d93f → 5bda51bfbc62`. Run `scripts/Sync-Upstream.ps1` to reset master to upstream + re-apply tooling, then rebase `blackwell/*` branches. (Not yet done this session.)

## 3. CURRENT UPSTREAM BUILD CONTEXT
- Windows runner: `windows-2022` (GitHub-hosted)
- CUDA: 13.3 (nvcc 13.3.33, cuBLAS 13.5.1.27)
- Generator: Ninja Multi-Config, `CMAKE_BUILD_TYPE=Release`
- `GGML_NATIVE=OFF` (no host GPU on runner)
- SM120a: CMake auto-appends `120a-real` when CUDA >= 12.8 and `GGML_NATIVE=OFF`

## 4. PR STATUS (upstream ggml-org/llama.cpp)
- **#28572** (TMA/cp.async pipelined NVFP4 MMQ): **open, unmerged**. head `10247a1a2077`, base `e71b80510c84`. Title: "ggml-cuda: pipeline NVFP4 MMQ tile loads with cp.async + mbarriers".
- **#26704** (CUTLASS SM120 MoE prefill): **open, unmerged**. head `94ee2c759734`, base `171974745106`. Title: "CUDA: Add experimental SM120 CUTLASS MoE prefill".
- **No combined TMA+CUTLASS branch**: the two are disjoint code paths (zero file overlap); no NVFP4 workload activates both simultaneously. Documented in `BLACKWELL-BUILD.md`.

## 5. BRANCHES
| Branch | Base | HEAD | Delta from master |
|--------|------|------|-------------------|
| `master` | upstream `16378d93f` (source) | `ba8a244d5808` | CI/workflow + docs only |
| `blackwell/tma` | master | `00cef0cb0b82` | 2 commits (TMA/cp.async NVFP4 MMQ: `308752871` + `00cef0cb0`) |
| `blackwell/cutlass` | master | `63a6c0d35a8c` | 8 commits (CUTLASS SM120 MoE, rebased via `-X theirs`, + `glu_limit` fix `63a6c0d35`) |
| `blackwell/combined` | — | **not created** (disjoint paths) | — |

## 6. GITHUB ACTIONS
- Workflow: `.github/workflows/build-blackwell-windows.yml` (trigger `workflow_dispatch` only)
- Inputs: `variant` = `stock` | `tma` | `cutlass` | `all`; `publish_release` = boolean
- Runner: `windows-2022`; CUDA 13.3; Ninja Multi-Config; `GGML_NATIVE=OFF`
- Architecture: auto-selected by llama.cpp (`120a-real` appended; verify step warns on no-GPU `native`)
- Sync tooling: `.github/workflows/sync-upstream.yml` (scheduled) + `scripts/Sync-Upstream.ps1`
- Docs: `BLACKWELL-BUILD.md`

## 7. BUILDS (verified CI run 34598315657, SHA `ba8a244d5808`)
| Variant | Job status | Artifact | Source SHA |
|---------|-----------|----------|-----------|
| **stock** | **SUCCESS** | `llama-win-cuda13.3-sm120a-stock-ba8a244d58.zip` (~44.5 MB) | `ba8a244d58` (master) |
| **tma** | **SUCCESS** | `llama-win-cuda13.3-sm120a-tma-00cef0cb0b.zip` (~44.5 MB) | `00cef0cb0b` (blackwell/tma) |
| **cutlass** | **SUCCESS** | `llama-win-cuda13.3-sm120a-cutlass-63a6c0d35a.zip` (~44.7 MB) | `63a6c0d35a` (blackwell/cutlass) |
| **combined** | N/A | — (not built; disjoint paths) | — |

## 8. RELEASE — **PUBLISHED** ✅
- **Release:** `blackwell-build-20260911-43f3dda623` — "Blackwell Windows builds (20260911)"
- **Publishing run:** 34603606613 (SHA `068b75caa`) — **all 4 jobs success** (stock, tma, cutlass build + publish-release)
- **Assets (3, ~42 MB each):**
  - `llama-win-cuda13.3-sm120a-stock-068b75caa3.zip`
  - `llama-win-cuda13.3-sm120a-tma-00cef0cb0b.zip`
  - `llama-win-cuda13.3-sm120a-cutlass-63a6c0d35a.zip`
- Each ZIP: `llama-server.exe`, `llama-bench.exe`, `ggml-cuda.dll`, `ggml.dll`, `BUILD-METADATA.json`, `SHA256SUMS.txt`
- **publish-release job fix** (commit `068b75caa`): added `actions/checkout@v4` + `GH_REPO: brx19/llama.cpp` so `gh release create` knows the target repo; wrapped `git ls-remote` in try/catch; made `gh release create` failures fatal (`throw` on non-zero exit). The first release-build (34600950163) failed at publish-release due to missing repo context; re-dispatched on `068b75caa` → success.

## 9. VALIDATION (per-variant, from CI logs of run 34598315657)
- ZIP structure: `llama-server.exe`, `llama-bench.exe`, `ggml-cuda.dll`, `ggml.dll`, `BUILD-METADATA.json`, `SHA256SUMS.txt` — all present; "ZIP validation OK" for all three.
- SHA256: `SHA256SUMS.txt` generated from actual binary hashes; validated by the workflow's own step.
- Native binaries: `llama-server.exe`, `llama-bench.exe` (Windows x64 PE).
- Metadata: `BUILD-METADATA.json` records `variant`, `source_sha`, `cuda_version` (13.3), `cuda_architecture`, `cutlass_enabled` (true for cutlass, false otherwise), `msvc_version`, toolchain versions.
- GPU runtime: **NOT validated on-device** (runner has no GPU; compile-only target). Runtime perf on RTX 5090 is out of CI scope — treat CI compile-success as a build gate, not a performance proof.
- Executable version resource (`ProductName`/`ProductVersion`): empty in the PE resource (linker does not populate it for these builds) — metadata step succeeds; version is tracked in `BUILD-METADATA.json` instead.

## 10. ISSUES / LIMITATIONS
1. **No-GPU arch resolution**: runner resolves `CMAKE_CUDA_ARCHITECTURES` to `native` (no device); llama.cpp still appends `120a-real` to the arch list so the RTX 5090 gets a native SASS image. The verify step is warn-only. To force SM120a-*only* (smaller binary) requires a repo-side CMake hook — not done (would diverge from upstream source).
2. **Upstream drift**: `upstream/master` advanced `16378d93f → 5bda51bfbc62` after the fork was built. `master`/`blackwell/*` are based on the older base. A `Sync-Upstream.ps1` run is needed to catch up (master reset + branch rebase).
3. **CUTLASS `glu_limit` rebase artifact**: `-X theirs` rebase had pulled master's `mmvf.cu`/`mmvq.cu` (reading `fusion.glu_limit`) over the CUTLASS PR's `common.cuh` (which dropped that member). Fixed by removing the `glu_limit` mechanism from both files on `blackwell/cutlass` (commit `63a6c0d35`). The SWIGLU_CLAMP op is not part of the CUTLASS MoE path, so the fix is behavior-preserving for the target workload.
4. **Release asset download**: the `api-version=2` raw artifact endpoint rejects the PAT (401/403), so external ZIP re-download isn't possible with this token — release attachment is done by the in-runner `publish-release` job instead.
5. **Inherited `model-naming` workflow**: upstream's `model-naming` workflow targets a self-hosted runner unavailable on the fork → its jobs stay `queued`. Non-blocking (runs in parallel on the build job), but clutters the run list. Could be deleted from `.github/workflows/` on the fork if desired.

## 11. COMMITS PUSHED (this session, on master)
- CI workflow + sync + docs (initial) + 6 fix commits:
  1. `ci: fix matrix env interpolation`
  2. `ci: use checkout@v4 + upload/download-artifact@v4 (node20)`
  3. `ci: fix PowerShell quoting (unterminated string in toolchain step)`
  4. `ci: fix $exe: drive-reference parser error`
  5. `ci: let llama.cpp auto-select CUDA arch (120a-real)`
  6. `ci: arch verify is now warn-not-fail`
- `docs: clarify master = upstream source + CI tooling`
- `chore: Sync-Upstream.ps1 preserves tooling on master`
- **blackwell/cutlass**: `fix(cutlass): remove glu_limit references (rebase artifact)` (`63a6c0d35`)

## 12. JIT-GATEWAY / PRODUCTION
- **JIT-Gateway: NOT modified** (per constraint).
- **Production llama.cpp: NOT touched** (no clone into `E:\AI_DEV\lama.cpp`; build is CI-only, no local/Docker/WSL build).

## 13. ROOT-CAUSE LOG (diagnosed during bring-up)
| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Runs "complete with no jobs" / `VARIANT: $_` | `actions/checkout@v6` needs node24 (absent on windows-2022) → workflow ran with unprocessed `${{}}` | checkout/upload/download-artifact @v4 |
| 2 | `build ($_, ...)` literal | matrix value inlined into PowerShell | job-level `env: VARIANT` + `$env:VARIANT` |
| 3 | `nvcc fatal: Cannot find cl.exe` | one-shot `cmd /c vcvarsall` doesn't export MSVC env to CMake CUDA detection | source vcvarsall, export env to session |
| 4 | `Cannot find build\bin\Release\llama-server.exe` | Ninja Multi-Config + Release outputs to `build\bin\` | search both `build\bin\` and `build\bin\Release\` |
| 5 | cutlass `glu_limit` no-member | `-X theirs` rebase mixed master's .cu with PR's .cuh | remove `glu_limit` from mmvf.cu/mmvq.cu |
| 6 | `Using CMAKE_CUDA_ARCHITECTURES=native` | `enable_language(CUDA)` shadows `-D` arch with `native` | drop `-DCMAKE_CUDA_ARCHITECTURES`; let llama.cpp auto-add `120a-real` |
| 7 | `Variable reference is not valid: ':'` | `$exe:` in double-quoted string = drive-ref | `${exe}:` |
| 8 | publish-release `gh release create` fails (exit 1), "fatal: not a git repository" | publish-release job had **no `actions/checkout`** → `gh` has no repo context, `git` fails outside a repo | add `actions/checkout@v4` + `GH_REPO: brx19/llama.cpp`; wrap `git ls-remote` in try/catch; make `gh release create` failures fatal |

## 14. NEXT STEPS (optional / user-driven)
1. **Release is published** (`blackwell-build-20260911-43f3dda623`) — no further action needed there.
2. Run `scripts/Sync-Upstream.ps1` to sync master to `upstream/master` (`5bda51bf`) + rebase branches (upstream drifted after the fork was built).
3. Optionally delete the inherited `model-naming` workflow to declutter the fork's run list.
4. On the RTX 5090 host: extract a ZIP, confirm `llama-server.exe` loads the RTX 5090 (CC 12.0), then A/B `stock` vs `tma` vs `cutlass` on the Qwen3.8-27B NVFP4 MTP workload (~196K context) for real perf numbers.

---

## 15. ARCHITECTURE REDESIGN (2026-09-13) — supersedes sections 6, 9, 13

The monolithic single-job TMA build was **replaced** with a split
architecture mirroring the current official upstream Windows release
pipeline (`release.yml`). The sections above describe the **old**
monolithic architecture; this section supersedes them for the workflow.

### What changed

* **No more monolithic build.** The single `build-tma-cuda` job that built
  `llama-server` + the full CUDA package is gone.
* **No more custom packaging.** The recursive CUDA directory scan
  (`cudaSubDirs`), the embedded Python in PowerShell, the manually
  reconstructed Release runtime (`build-runtime-manifest.json`), and the
  hand-maintained 54-file official parity list
  (`official-win-cuda-13.3-x64-files.txt`) are **all deleted**.
* **No more variant matrix.** `stock` / `cutlass` / `all` / `combined` are
  **not built** (TMA only, per scope).
* The workflow now mirrors upstream `release.yml`:
  * **Job A `build-windows-cpu`** — full toolset from `blackwell/tma`, CUDA
    OFF, BoringSSL ON (upstream `windows-cpu` recipe).
  * **Job B `build-tma-cuda`** — only `ggml-cuda.dll` (120a-real, CUDA 13.3)
    (upstream `windows-cuda` recipe).
  * **Job C `cuda-runtime`** — CUDA 13.3 runtime DLLs via upstream's exact
    `robocopy` commands.
  * **Job D `merge-tma-package`** — merges A+B+C into one self-contained
    `llama-win-cuda13.3-sm120a-tma-<sha>.zip`, writes `BUILD-METADATA.json`
    + `SHA256SUMS.txt`, runs `pe_validate.py` as a final sanity check.

### Phase 1 verification (backend-only)

The TMA delta (PR #28572) was verified to be **CUDA-backend-only**:

```
git diff --name-only 16378d93f...blackwell/tma
  ggml/src/ggml-cuda/cp-async.cuh    (new)
  ggml/src/ggml-cuda/mmq-vec-dot.cuh
  ggml/src/ggml-cuda/mmq.cu
  ggml/src/ggml-cuda/mmq.cuh
```

Upstream base SHA: `16378d93f94012d4228c8c7683adce3f286aee5d`. No
ABI-relevant files outside `ggml/src/ggml-cuda/` are touched, so the CPU
toolset built from the same tree is the official upstream-style Windows x64
toolset.

### Validation

`pe_validate.py` is now a **final sanity check only** (not a build system):
it fails on Debug CRT imports and verifies the PE dependency closure of the
root executables. The primary correctness property now comes from using the
same packaging structure as upstream.

### Definition of done

Success = one workflow run produces
`llama-win-cuda13.3-sm120a-tma-<sha>.zip` containing: the complete
upstream-style Windows x64 toolset, the TMA `ggml-cuda.dll`, matching CUDA
13.3 runtime DLLs, no Debug CRT dependencies, and complete metadata +
hashes. Only after this succeeds will the workflow be generalized to Stock
and CUTLASS.
