# Blackwell (RTX 5090 / SM120a) Windows CUDA Build

Native Windows x64 CUDA builds of llama.cpp for the **NVIDIA GeForce RTX 5090**
(Blackwell, compute capability **SM120**), built on GitHub-hosted Windows
runners. Target workload: Qwen3.8-27B NVFP4 MTP, long context (~196K).

> **Scope note:** This repository maintains a clean upstream mirror on
> `master`, experimental Blackwell branches, and the build/packaging
> pipeline. GPU runtime validation and benchmarking are done **locally** on
> the RTX 5090 — CI runners have no GPU. JIT-Gateway integration is a
> separate, later task and is NOT touched here.

---

## 1. Repository architecture

```
origin    = https://github.com/brx19/llama.cpp        (your public fork)
upstream  = https://github.com/ggml-org/llama.cpp.git (official)
```

Local development clone:

```
/home/ubuntu/Projects/llama.cpp-blackwell   (WSL)
≡ \\wsl$\Ubuntu\home\ubuntu\Projects\llama.cpp-blackwell   (from Windows)
```

> The task brief suggested `E:\AI_DEV\llama.cpp-blackwell`. Because the
> agent operates inside WSL, the clone lives in the WSL filesystem, which is
> visible from Windows at the UNC path above. `E:\AI_DEV\llama.cpp` (your
> existing production runtime) and `E:\AI_DEV\lama.cpp\JIT-Gateway` are NOT
> touched. If you prefer the clone on the E: drive, copy/symlink it there
> and update this doc.

### Branch map

| Branch | Purpose | Based on |
|---|---|---|
| `master` | Clean mirror of `upstream/master` — the **stock** baseline | upstream |
| `blackwell/tma` | TMA / cp.async NVFP4 MMQ pipeline (PR #28572) | master + 2 commits |
| `blackwell/cutlass` | CUTLASS SM120 MoE prefill, MXFP4/NVFP4 (PR #26704) | master + 7 commits |
| `blackwell/combined` | **Not created** — see section 6 | — |

### Invariant

```
master == upstream/master   (exact, same SHA)
```

Never commit custom changes directly to `master`. All experimental work lives
in `blackwell/*` branches.

---

## 2. Master sync policy

```powershell
git fetch upstream --prune --tags
git fetch origin   --prune --tags

# Divergence check (left=origin/master, right=upstream/master):
git log --oneline --left-right origin/master...upstream/master

# If master has unique commits, preserve them first:
git branch backup/master-before-sync-YYYYMMDD-HHMM master

# Synchronize:
git switch master
git reset --hard upstream/master
git push --force-with-lease origin master

# Verify (must match exactly):
git rev-parse master
git rev-parse upstream/master
```

**Plain `--force` is never used** — only `--force-with-lease`.

The automated version is `scripts/Sync-Upstream.ps1` (run with `-WhatIf` for a
dry run). A scheduled GitHub workflow (`.github/workflows/sync-upstream.yml`)
runs the same safe logic daily and files an issue if a rebase conflicts.

---

## 3. TMA branch (`blackwell/tma`)

* **Source:** upstream PR **#28572** — "ggml-cuda: pipeline NVFP4 MMQ tile
  loads with cp.async and TMA on Blackwell (+14% pp)" (jesdga95).
* **Original PR head SHA:** `10247a1a207746b92050c7cbb6855f45f8242c9f`
  (2 commits: `0ed55dbcb` cp.async pipelined NVFP4 MMQ, `10247a1a2` TMA
  pipelined NVFP4 MMQ tiles with mbarriers).
* **Upstream base SHA (original PR):** `e71b80510c848c00175924ecf3c40333ccae8eb5`
* **Status at time of port (2026-09-11):** OPEN, not merged.
* **Delta files:** `ggml/src/ggml-cuda/cp-async.cuh` (new),
  `mmq-vec-dot.cuh`, `mmq.cu`, `mmq.cuh`.

The branch was created by **cherry-picking** the 2 PR commits onto current
`upstream/master` (clean apply, no conflicts). It represents:

```
current upstream/master  +  only the TMA/Blackwell optimization delta
```

### How the TMA path is selected (runtime)

`ggml_cuda_mmq_encode_tmap_nvfp4()` in `mmq.cu` dynamically loads
`cuTensorMapEncodeTiled` via `dlsym`. On an RTX 5090 (CC 12.0, CUDA ≥ 12.8)
the call succeeds, so the NVFP4 MMQ kernel receives a populated `ggml_cuda_tmap`
and uses TMA + `cp.async` pipelined tile loads. On older GPUs the symbol is
absent and the kernel falls back to the standard path — the same binary is
safe across GPUs, but the Blackwell-specific speedup only materializes on
SM120. **GPU validation required to confirm the +14% pp claim.**

---

## 4. CUTLASS branch (`blackwell/cutlass`)

* **Source:** upstream PR **#26704** — "CUDA: Add experimental SM120 CUTLASS
  MoE prefill for MXFP4 and NVFP4" (leonardHONG).
* **Original PR head SHA:** `94ee2c759734c697ecaa0cb05058cda4f46b42e3`
  (7 commits: dense CUTLASS block-scaled path → repacked NVFP4 decode →
  generalize repacked fusion + tests → limit repack to dense models →
  CMake CUTLASS license → chunked uploads for repacked weights).
* **Upstream base SHA (original PR):** `17197474510622a3b4ea7d0909d70b606f542b96`
  (merge-base with master: `749f688f`).
* **Status at time of port (2026-09-11):** OPEN, not merged.
* **Delta files:** `ggml/CMakeLists.txt` (`GGML_CUDA_CUTLASS` option),
  `ggml/src/ggml-cuda/CMakeLists.txt` (CUTLASS FetchContent),
  `mmq-cutlass.cu/.cuh` (new), `repack-cutlass-blockscaled.cu/.cuh` (new),
  `mmvq.cu`, `ggml-cuda.cu`, `common.cuh`, `mmvf.cu`,
  `src/llama-model-loader.cpp`, `src/llama-model.cpp`,
  `tests/test-backend-ops.cpp`.

The branch was created by **rebasing the 7 PR commits** onto current
`upstream/master` (`git rebase -X theirs --onto master master upstream/pr-26704`).
All 7 applied cleanly; the result is exactly `master + 7 commits` with no
stray upstream history.

### Build option

```
-DGGML_CUDA_CUTLASS=ON
```

CUTLASS version is **whatever the branch pins** in
`ggml/src/ggml-cuda/CMakeLists.txt` — do not bump it arbitrarily.

### How the CUTLASS path is selected (runtime)

Weights are **repacked** into a CUTLASS blockscaled layout at load time
(`llama-model-loader.cpp`). `ggml-cuda.cu` dispatches `ggml_cuda_mul_mat` as:
if `src0` is a repacked buffer → `ggml_cuda_mul_mat_vec_q` (decode / small
batch) or `ggml_cuda_cutlass_mul_mat` (MoE prefill, dense models only).
The repack decision is model-dependent, so the CUTLASS path is used on the
NVFP4 model when repacking applies. **GPU validation required.**

---

## 5. Build workflow

File: `.github/workflows/build-blackwell-windows.yml`

* **Runner:** `windows-2022` (same as the official llama.cpp
  `.github/workflows/build-cuda-windows.yml`).
* **CUDA:** **13.3** x64, installed from NVIDIA redist archives via the
  official composite action `.github/actions/windows-setup-cuda`
  (nvcc **13.3.33**, cuBLAS **13.5.1.27**).
* **Generator:** Ninja Multi-Config (choco ninja), Release config.
* **Architecture:** `-DCMAKE_CUDA_ARCHITECTURES=120a-real` — the
  architecture-specific Blackwell target (FP4 tensor cores are NOT
  forward-compatible; they require `12Xa`). The workflow verifies the CMake
  resolution (`Using CMAKE_CUDA_ARCHITECTURES=120a-real`) and greps the build
  log for `sm_120a` evidence.
* **Base CMake flags** (mirroring the official Windows CI):
  `-DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF`
  plus, for the cutlass variant only, `-DGGML_CUDA_CUTLASS=ON`.
* **No** `GGML_CUDA_FORCE_CUBLAS` / `GGML_CUDA_FORCE_MMQ` — stock dispatch is
  preserved so all variants are A/B-comparable.

### Triggering a build

```powershell
gh workflow run build-blackwell-windows.yml --ref master -f variant=all
# or a single variant:
gh workflow run build-blackwell-windows.yml --ref master -f variant=stock
```

GitHub UI: Actions → "CI (Blackwell, windows)" → Run workflow → pick variant.

### Variants

| Variant | Source branch | CMake delta |
|---|---|---|
| `stock` | `master` | — |
| `tma` | `blackwell/tma` | — |
| `cutlass` | `blackwell/cutlass` | `-DGGML_CUDA_CUTLASS=ON` |
| `all` | (matrix) | as above |
| `combined` | **not built** | — |

### Artifact naming

```
llama-win-cuda13.3-sm120a-stock-<shortsha>.zip
llama-win-cuda13.3-sm120a-tma-<shortsha>.zip
llama-win-cuda13.3-sm120a-cutlass-<shortsha>.zip
```

### ZIP contents

```
llama-server.exe
llama-bench.exe
ggml*.dll                    (ggml.dll, ggml-cuda.dll, etc.)
cudart64_13.dll             (CUDA runtime)
cublas64_13.dll, cublasLt64_13.dll
nvvm64.dll, nvrtc64_130_*.dll
nvinfer/nvonnxparser (if present)
BUILD-METADATA.json
SHA256SUMS.txt
```

`BUILD-METADATA.json` contains: variant, build timestamp, source branch,
source SHA, upstream base SHA, original PR SHA, Actions run ID, runner image,
MSVC version, CUDA version, CUDA component versions, CMake version, Ninja
version, CMake arguments, CUDA architecture, CUTLASS enabled + revision,
target GPU, and the GPU-validation marker.

`SHA256SUMS.txt` covers every packaged `.exe`/`.dll`.

### Validation performed on CI (no GPU)

* build completes,
* expected EXEs/DLLs exist,
* CMake resolved `CMAKE_CUDA_ARCHITECTURES=120a-real`,
* build log contains `sm_120a` device-code evidence,
* ZIP created and re-validated (structure, metadata JSON, all hashes),
* executable version metadata queried (no CUDA init).

**GPU runtime correctness is NOT verified on CI.** Every package is marked
`REQUIRES LOCAL RTX 5090 VALIDATION`.

### Downloading artifacts

* **Long-term:** GitHub Releases (see below).
* **Convenience:** Actions artifacts (14-day retention) —
  Actions → run → Artifacts.

### Publishing releases

The workflow has a `publish_release` input (default `false`). When true,
successful builds are attached to a deterministic release tag:

```
blackwell-build-<YYYYMMDD>-<upstream-sha10>
```

Failed runs never publish. To publish manually after a build:

```powershell
gh release create blackwell-build-$(Get-Date -Format yyyyMMdd)-<sha10> `
  llama-win-cuda13.3-sm120a-stock-*.zip `
  llama-win-cuda13.3-sm120a-tma-*.zip `
  llama-win-cuda13.3-sm120a-cutlass-*.zip `
  --title "Blackwell Windows builds"
```

---

## 6. Combined variant — deliberately NOT created

Per the branch-porting analysis (2026-09-11):

1. **Zero file overlap** between the TMA delta and the CUTLASS delta — they
   touch different files and compile independently, so a combined *build* is
   technically possible.
2. **However, the two paths are mutually exclusive at runtime for NVFP4:**
   * TMA accelerates the **canonical MMQ kernel** (`mmq.cu`, canonical weight
     layout, prefill).
   * CUTLASS requires weights **repacked** into a blockscaled layout at load
     time; `ggml_cuda_mul_mat` dispatches repacked buffers to
     `ggml_cuda_mul_mat_vec_q` / `ggml_cuda_cutlass_mul_mat`, which **bypass
     the MMQ kernel entirely**.
   * On the target model (dense Qwen3.8-27B NVFP4), CUTLASS's prefill path is
     limited to **dense models only**, and once repacking is active the TMA
     MMQ tile-loading optimization is never reached for the same matmul.

Therefore combining them is **not additive** — the CUTLASS path replaces the
very kernel the TMA path optimizes. Building `combined` would yield a binary
identical in effect to `cutlass`, with the TMA delta dead code for NVFP4.

**Policy:** `combined` is not built. If upstream merges both PRs and the
dispatch logic changes so both can be active simultaneously (e.g. TMA used for
the non-repacked fallback inside a CUTLASS build), revisit this section and
re-derive the combined branch from that state.

---

## 7. Updating from upstream

```powershell
# Manual:
pwsh -File scripts/Sync-Upstream.ps1 -WhatIf     # dry run
pwsh -File scripts/Sync-Upstream.ps1             # real
```

The script:

1. fetches `upstream` and `origin` (prune, tags),
2. verifies/updates `master` (backs up unique master commits first),
3. pushes clean master with `--force-with-lease`,
4. re-fetches PR #28572 / #26704 refs,
5. checks whether either PR was merged upstream (if so, marks the branch
   unnecessary),
6. rebases each experimental branch's PR delta onto current master
   (`-X theirs`),
7. pushes with `--force-with-lease`,
8. **stops on conflict** — aborts the rebase, leaves the branch intact, and
   prints the manual resolution steps (never invents a resolution).

The scheduled workflow `.github/workflows/sync-upstream.yml` (daily 07:00 UTC,
plus manual `workflow_dispatch`) runs the same safe logic and opens a GitHub
issue when a branch cannot be auto-refreshed.

---

## 8. Interpretation of build metadata

`BUILD-METADATA.json` fields:

| Field | Meaning |
|---|---|
| `variant` | stock / tma / cutlass |
| `source_branch` / `source_sha` | branch and full SHA that was built |
| `upstream_base_sha` | `origin/master` SHA at build time (should equal `source_sha` for stock) |
| `original_pr_sha` | head SHA of the source PR (tma/cutlass) |
| `github_actions_run_id` | Actions run for traceability |
| `msvc_version` | `cl.exe` version (14.4x from VS2022) |
| `cuda_version` | nvcc release (13.3) |
| `cuda_component_versions` | per-component DLL versions (cudart, libcublas, …) |
| `cmake_version` / `ninja_version` | tool versions |
| `cmake_arguments` | exact configure flags |
| `cuda_architecture` | `120a-real` |
| `cutlass_enabled` / `cutlass_revision` | CUTLASS flag + pinned revision |
| `gpu_validation` | `REQUIRES LOCAL RTX 5090 VALIDATION` |

A future local benchmark can reconstruct the exact build from this file.

---

## 9. Local benchmarking (later phase — out of scope here)

Expected comparison: STOCK vs TMA vs CUTLASS with
`Qwen3.8-27B-NVFP4-MTP-LOW` on the RTX 5090:

```
context = 196608, batch = 4096, ubatch = 1024, parallel = 1
Flash Attention = ON, KV = q8_0/q8_0, MTP = draft-mtp,
spec_draft_n_max = 3, backend sampling = OFF, CUDA Graph = OFF
```

No variant-specific defaults were changed, so A/B testing is fair. Do NOT
treat CI compilation success as runtime performance evidence.

---

## 10. Repository hygiene

Committed: workflows, `scripts/`, this doc, and the experimental-branch
source deltas.

**Never commit:** compiled EXEs/DLLs, ZIP artifacts, CUDA installers,
temporary build directories, model GGUF files, local credentials, benchmark
JSON results. Binaries belong in GitHub Releases / Actions artifacts.

---

## 11. Branch provenance

| Branch | Upstream base SHA | Source PR | Original PR SHA | Final branch SHA (2026-09-11) | Last refreshed |
|---|---|---|---|---|---|
| `blackwell/tma` | `16378d93f94012d4228c8c7683adce3f286aee5d` | #28572 (open) | `10247a1a207746b92050c7cbb6855f45f8242c9f` | see `git rev-parse blackwell/tma` | 2026-09-11 |
| `blackwell/cutlass` | `16378d93f94012d4228c8c7683adce3f286aee5d` | #26704 (open) | `94ee2c759734c697ecaa0cb05058cda4f46b42e3` | see `git rev-parse blackwell/cutlass` | 2026-09-11 |

Update this table (and the "last refreshed" dates) after every sync.
