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

## 8. RELEASE
- Tag: `blackwell-windows-cuda13.3-sm120a`
- Status: **in progress** — release-publishing run **34600950163** dispatched (`variant=all, publish_release=true`); rebuilding all three variants then creating the Release with the three ZIPs + SHA256SUMS attached. (Poller proc_3d56b79bc558.)
- The `publish-release` job attaches the build artifacts as release assets.

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

## 14. NEXT STEPS (optional / user-driven)
1. Wait for release run 34600950163 to finish → confirm Release `blackwell-windows-cuda13.3-sm120a` has the 3 ZIPs + SHA256SUMS.
2. Run `scripts/Sync-Upstream.ps1` to sync master to `upstream/master` (`5bda51bf`) + rebase branches (upstream drifted).
3. Optionally delete the inherited `model-naming` workflow to declutter the fork's run list.
4. On the RTX 5090 host: extract a ZIP, confirm `llama-server.exe` loads the RTX 5090 (CC 12.0), then A/B `stock` vs `tma` vs `cutlass` on the Qwen3.8-27B NVFP4 MTP workload (~196K context) for real perf numbers.
