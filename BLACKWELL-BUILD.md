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
master = upstream/master source + CI/tooling additions
```

* All **source files** on `master` are byte-identical to `upstream/master`
  (verified by `git diff upstream/master...master -- ggml/ examples/ tests/
  CMakeLists.txt` being empty).
* `master` additionally carries **only** build tooling:
  `.github/workflows/build-blackwell-windows.yml`,
  `.github/workflows/sync-upstream.yml`, `scripts/Sync-Upstream.ps1`, and
  `BLACKWELL-BUILD.md`. These live on the default branch so that GitHub
  Actions picks them up automatically.
* Because of those additions, `git rev-parse master` does NOT equal
  `git rev-parse upstream/master`. The invariant is therefore checked as
  **source parity**, not SHA equality.
* Never commit llama.cpp *source* changes to `master`. All experimental
  work lives in `blackwell/*` branches.

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
# Re-apply tooling-only additions on top (workflows, scripts, docs):
git cherry-pick <tooling-commits>   # or re-add the 4 tooling files
# Verify source parity:
git diff --name-only upstream/master...master
# (must list only .github/, scripts/, BLACKWELL-BUILD.md)
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

> **Architecture (redesigned 2026-09-13):** the workflow now mirrors the
> **current official ggml-org/llama.cpp Windows release pipeline**
> (`.github/workflows/release.yml`) rather than a single monolithic TMA
> build. It builds the CPU toolset and the CUDA backend **separately**,
> then merges them — exactly as upstream assembles its Windows CUDA release
> from `windows-cpu` + `windows-cuda` + the CUDA runtime.

* **Scope:** **TMA only** (stock / CUTLASS are NOT built until this
  architecture has produced one successful TMA ZIP).
* **Trigger:** `workflow_dispatch` only; input `publish_release` (boolean).
* **Runner:** `windows-2022` (same as upstream `windows-cuda`).
* **CUDA:** **13.3** x64, installed from NVIDIA redist archives via the
  official composite action `.github/actions/windows-setup-cuda`
  (nvcc **13.3.33**, cuBLAS **13.5.1.27**).
* **Generator:** Ninja Multi-Config (choco ninja), **`--config Release`**
  (explicit — Ninja Multi-Config ignores `-DCMAKE_BUILD_TYPE`).

### Job graph

| Job | Purpose | Output |
|---|---|---|
| **A `build-windows-cpu`** | Full Windows toolset from the **`blackwell/tma`** tree, **CUDA OFF**, BoringSSL ON. Upstream `windows-cpu` recipe. | `llama-bin-win-cpu-x64.zip` (whole `build\bin\Release\`) |
| **B `build-tma-cuda`** | **Only `ggml-cuda.dll`** from the **`blackwell/tma`** tree, CUDA 13.3, `-DCMAKE_CUDA_ARCHITECTURES=120a-real`. Upstream `windows-cuda` recipe (`--target ggml-cuda`). | `llama-bin-win-cuda-13.3-x64-tma.zip` |
| **C `cuda-runtime`** | CUDA 13.3 runtime DLLs staged with upstream's **exact** `robocopy` commands from the installed toolkit. | `cudart-llama-bin-win-cuda-13.3-x64.zip` |
| **D `merge-tma-package`** | Injects A + B + C into ONE self-contained package, writes `BUILD-METADATA.json` + `SHA256SUMS.txt`, runs `pe_validate.py` as a final sanity check, creates the final ZIP. | `llama-win-cuda13.3-sm120a-tma-<sha>.zip` |
| `publish-release` | Optional (input `publish_release`). | GitHub release asset |

### ABI-coherence note

Jobs A and B **both build from the same `blackwell/tma` source tree**. The
TMA delta (PR #28572) is confined to `ggml/src/ggml-cuda/*` (verified:
`cp-async.cuh`, `mmq-vec-dot.cuh`, `mmq.cu`, `mmq.cuh`), so building the CPU
toolset from that tree yields the official upstream-style Windows x64 toolset
while the TMA backend is exactly the replaced `ggml-cuda.dll`. The merged
package is therefore ABI-coherent. The "CPU runtime source SHA" in
`BUILD-METADATA.json` is `blackwell/tma` head — its tree is
upstream-base + the 4 CUDA-backend files.

> The final package is intentionally a **superset** of the official CUDA
> backend ZIP: upstream distributes the CUDA runtime DLLs separately
> (`cudart-llama-bin-win-cuda-13.3-x64.zip`); we bundle them so the package
> is directly runnable without merging another archive.

### Triggering a build

```powershell
gh workflow run build-blackwell-windows.yml --ref master
# optional: publish to a GitHub release
gh workflow run build-blackwell-windows.yml --ref master -f publish_release=true
```

GitHub UI: Actions → "CI (Blackwell TMA, windows)" → Run workflow.

### Artifact naming

```
llama-win-cuda13.3-sm120a-tma-<sha>.zip
```

### ZIP contents

```
llama-server.exe
llama-server-impl.dll
llama-bench.exe
llama.dll
llama-common.dll
ggml.dll
ggml-base.dll
ggml-cpu-*.dll                 (CPU backend variants)
ggml-cuda.dll                  (TMA / Blackwell backend, 120a-real)
cudart64_*.dll                (CUDA 13.3 runtime)
cublas64_*.dll
cublasLt64_*.dll
BUILD-METADATA.json
SHA256SUMS.txt
PE-VALIDATE-REPORT.json
```

`BUILD-METADATA.json` contains: variant (`tma`), build timestamp,
upstream base SHA, TMA SHA, original PR #28572 head SHA, CUDA version
(13.3), `CMAKE_CUDA_ARCHITECTURES` (`120a-real`), configuration
(`Release`), CPU runtime source SHA, CUDA backend source SHA, MSVC version,
CUDA/CMake/Ninja versions, Actions run ID, runner image, target GPU, and the
GPU-validation marker.

`SHA256SUMS.txt` covers every packaged `.exe`/`.dll`.

### Validation performed on CI (no GPU)

* build completes (A: full toolset; B: `ggml-cuda.dll` only),
* expected EXEs/DLLs exist in each job,
* merged staging validated for the full required file set,
* ZIP created and re-validated (structure, metadata JSON, all SHA256),
* `pe_validate.py` run as a **final sanity check** on the merged staging:
  * fails on any Debug CRT import (`ucrtbased.dll`, `msvcp140d.dll`,
    `vcruntime140d.dll`, `vcruntime140_1d.dll`),
  * verifies the PE dependency closure of the root executables (no missing
    non-system, non-NVIDIA-driver imports — `nvcuda64.dll` is host-provided),
  * writes `PE-VALIDATE-REPORT.json`.

**GPU runtime correctness is NOT verified on CI.** Every package is marked
`REQUIRES LOCAL RTX 5090 VALIDATION`.

### Downloading artifacts

* **Long-term:** GitHub Releases (see below).
* **Convenience:** Actions artifacts (14-day retention) —
  Actions → run → Artifacts.

### Publishing releases

The workflow has a `publish_release` input (default `false`). When true, the
successful TMA build is attached to a deterministic release tag:

```
blackwell-tma-<YYYYMMDD>-<sha10>
```

Failed runs never publish.

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

## 9b. Troubleshooting (root causes found during bring-up)

Three CI/compile issues were diagnosed and fixed while bringing the workflow
to a green state. Recorded here so they are not re-encountered.

**1. `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` (configure failure)**

The original configure step used `cmd /c vcvarsall.bat x64 && cmake ...`. The
one-shot `cmd /c` does not reliably export the MSVC environment to CMake's
CUDA compiler-id detection sub-processes, so `nvcc` could not locate `cl.exe`
as the CUDA host compiler. Fix: the configure step now sources `vcvarsall.bat`
into the PowerShell session (captures `set` output and applies it via
`[Environment]::SetEnvironmentVariable`) and passes
`-DCMAKE_CUDA_HOST_COMPILER=<cl.exe path>` explicitly.

**2. `Cannot find path 'build\bin\Release\llama-server.exe'` (collect failure)**

TMA compiled and linked successfully but the collect step looked for binaries
in `build\bin\Release\`. With the `Ninja Multi-Config` generator plus
`-DCMAKE_BUILD_TYPE=Release`, outputs land in `build\bin\` (not a
`Release\` subdirectory). Fix: the collect step checks `build\bin` first and
falls back to `build\bin\Release`.

**3. `class "ggml_cuda_mm_fusion_args_device" has no member "glu_limit"`
(cutlass compile failure)**

A rebase artifact. `git rebase -X theirs` onto master pulled master's
`mmvf.cu`/`mmvq.cu` (which read `fusion.glu_limit` and handled
`GGML_GLU_OP_SWIGLU_CLAMP`) over the CUTLASS PR's versions. The cutlass
branch's `ggml_cuda_mm_fusion_args_device` struct (in `common.cuh`) has no
`glu_limit` member, and `ggml-cuda.cu` does not populate it. Fix: removed the
`glu_limit` variable, the `fusion.glu_limit` reads, the `SWIGLU_CLAMP` case,
and the unused-vars entries from `mmvf.cu`/`mmvq.cu` on the cutlass branch.
The CUTLASS MoE prefill path uses `SWIGLU_OAI` (not `SWIGLU_CLAMP`), so this
is a no-op for the intended workload. Master's other `mmvq.cu` improvements
(e.g. DGX Spark prefetch) are preserved.

**General note:** the workflow deliberately avoids a dynamic matrix
(`expand` job → `fromJson` → matrix → env). A literal `$_` matrix value broke
earlier runs. Variant→branch mapping uses three fixed jobs, each with a static
`VARIANT` env and an `if:` gate on the `variant` dispatch input.

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
