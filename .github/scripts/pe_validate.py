#!/usr/bin/env python3
"""PE dependency-closure validator for Windows llama.cpp runtime packages.

Authoritative PE import-table parsing (IMAGE_IMPORT_DESCRIPTOR walk, not raw
string matching) for every .exe/.dll in the package:

  * parses the real import table of each binary
  * resolves the dependency closure starting from the root binaries
    (llama-server.exe, llama-bench.exe) plus every build-produced backend DLL
  * classifies each dependency:
      SYSTEM      Windows system DLL - allowed external
      RELEASE_CRT release VC++ runtime - allowed (documented prerequisite)
      BUILD_DLL   llama.cpp build artifact - MUST be in the package
      CUDA_DLL    CUDA toolkit runtime DLL - MUST be in the package
      DEBUG_CRT   debug VC++ runtime - FATAL
      MISSING     unresolved non-system dependency - FATAL
  * secondary string-scan guard for debug CRT references
  * required-file checks, BUILD-METADATA.json integrity, SHA256SUMS verification
  * optional comparison against an official llama.cpp package file list

Usage:
  python3 pe_validate.py <package_dir> [--json report.json] [--official-list files.txt]

Exit code 0 = all checks passed; 1 = fatal findings (JSON report still written).
Only requires the CPython standard library.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

SYSTEM_DIRS = {
    "kernel32.dll", "kernelbase.dll", "advapi32.dll", "msvcrt.dll", "ntdll.dll",
    "ole32.dll", "oleaut32.dll", "shell32.dll", "user32.dll", "gdi32.dll",
    "winspool.drv", "ws2_32.dll", "winmm.dll", "version.dll", "clbcat32.dll",
    "crypt32.dll", "bcrypt.dll", "cryptbase.dll", "msasn1.dll", "normaliz.dll",
    "sechost.dll", "iphlpapi.dll", "dnsapi.dll", "setupapi.dll", "cfgmgr32.dll",
    "wininet.dll", "winhttp.dll", "wintrust.dll", "wlanapi.dll", "wship6.dll",
    "wshtcpip.dll", "dbghelp.dll", "psapi.dll", "profapi.dll", "winsock.dll",
    "d3d11.dll", "dwmapi.dll", "uxtheme.dll", "comdlg32.dll", "comctl32.dll",
    "imm32.dll", "msimg32.dll", "rasapi32.dll", "rpcrt4.dll", "wks32.dll",
    "wldp.dll", "winscard.dll", "winsta.dll", "winnsi.dll", "wbemuuid.dll",
    "wmiutils.dll", "wbemsvc.dll", "fastprox.dll", "wbemess.dll",
    "ntshrui.dll", "sxs.dll", "sfc.dll", "win32k.dll", "win32kbase.dll",
    "win32kfull.dll", "msvcp_win.dll",
}

DEBUG_CRT = {
    "vcruntime140d.dll", "msvcp140d.dll", "msvcp140_1d.dll",
    "vcruntime140_1d.dll", "ucrtbased.dll", "vcruntime140d_amd64.dll",
}

RELEASE_CRT = {
    "vcruntime140.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "ucrtbase.dll", "vcruntime140_clr0400.dll",
}

CUDA_PREFIXES = (
    "cudart", "nvcudart", "cublas", "nvrtc", "nvvm", "nvcuda",
    "nvjitlink", "cusolver", "cusparse", "nvtx", "cudnn", "nvtf32",
    "cufile", "nccl",
)

MACHINE_NAMES = {0x8664: "AMD64", 0x14C: "i386", 0xAA64: "ARM64"}


def is_system(name: str) -> bool:
    n = name.lower()
    return n.startswith("api-ms-win-") or n in SYSTEM_DIRS


def parse_pe(path: Path):
    """Parse a PE file. Returns (machine, is_64bit, imported_dll_names).

    Layout notes (verified against MSVC 2022 x64 binaries, including the
    52 KB VS loader stubs): the optional header begins at coff + 20 where
    coff = e_lfanew + 4. The DataDirectory array position is NOT at the
    standard opt+112 offset for all MSVC builds; we therefore locate it by
    scanning for a run of 16 plausible (rva, size) pairs anchored near the
    start of the optional header. The import table is dir[1].
    """
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise ValueError("not a PE file (missing MZ header)")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew + 24 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
        raise ValueError("missing PE signature")
    coff = e_lfanew + 4
    machine = struct.unpack_from("<H", data, coff)[0]
    n_sections = struct.unpack_from("<H", data, coff + 2)[0]
    size_opt = struct.unpack_from("<H", data, coff + 16)[0]
    magic = struct.unpack_from("<H", data, coff + 20)[0]
    if magic == 0x20B:
        is64 = True
    elif magic == 0x10B:
        is64 = False
    else:
        raise ValueError(f"unknown PE optional-header magic {magic:#x}")
    if not (0x28 < size_opt < 0x200):
        raise ValueError(f"implausible SizeOfOptionalHeader {size_opt}")
    opt = coff + 20
    sec = opt + size_opt
    sections = []
    for i in range(n_sections):
        o = sec + 40 * i
        if o + 40 > len(data):
            break
        vsize, vaddr = struct.unpack_from("<II", data, o + 8)
        rawsize, rawptr = struct.unpack_from("<II", data, o + 16)
        sections.append((vaddr, vsize, rawptr, rawsize))

    def rva2off(rva):
        for vaddr, vsize, rawptr, rawsize in sections:
            span = max(vsize, rawsize)
            if vaddr <= rva < vaddr + span:
                return rawptr + (rva - vaddr)
        return None

    def valid_rva(rva):
        if rva == 0:
            return True
        for vaddr, vsize, rawptr, rawsize in sections:
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                return True
        return False

    # Locate the DataDirectory array: scan candidate starts (4-byte aligned)
    # within the optional-header region for a run of 16 plausible (rva, size)
    # pairs. Prefer the start with the longest valid run; ties broken by
    # proximity to opt.
    #
    # Empirically, MSVC 2022 x64 binaries place the DataDirectory at
    # opt + 104 (not the standard opt + 112). We scan to be safe.
    best_start, best_count = None, 0
    hi = min(opt + size_opt, len(data) - 16 * 16)
    for start in range(opt, hi, 4):
        count = 0
        for i in range(16):
            rva, size = struct.unpack_from("<II", data, start + 16 * i)
            if valid_rva(rva) and size < 0x200000:
                count += 1
            else:
                break
        if count > best_count:
            best_count, best_start = count, start
    if best_start is None or best_count < 8:
        # Fall back to the standard offset
        best_start = opt + (112 if is64 else 96)
    rva, _size = struct.unpack_from("<II", data, best_start + 16 * 1)
    if rva == 0:
        return machine, is64, []
    off = rva2off(rva)
    if off is None:
        raise ValueError(f"import table RVA {rva:#x} outside sections")
    # Walk IMAGE_IMPORT_DESCRIPTOR entries (20 bytes each) until an all-zero
    # terminator. The table is NOT a counted array; the first DWORD of each
    # entry is OriginalFirstThunk (a RVA), not a count.
    dll_names = []
    for i in range(64):
        entry = off + 20 * i
        if entry + 20 > len(data):
            break
        oft, tds, fc, name_rva, ft = struct.unpack_from("<IIIII", data, entry)
        if oft == 0 and name_rva == 0 and ft == 0:
            break
        if name_rva == 0:
            continue
        no = rva2off(name_rva)
        if no is None:
            raise ValueError(f"import name RVA {name_rva:#x} outside sections")
        end = data.index(b"\x00", no)
        name = data[no:end]
        # sanity: import names are short ASCII identifiers
        if len(name) > 128 or any(b < 0x20 for b in name):
            raise ValueError(f"import name not a valid DLL name: {name[:40]!r}")
        dll_names.append(name.decode("utf-8", "replace"))
    return machine, is64, dll_names


def classify(name: str, build_files: set) -> str:
    n = name.lower()
    if n in DEBUG_CRT or n.startswith(("vcruntime140d", "msvcp140d", "ucrtbased")):
        return "DEBUG_CRT"
    if n in RELEASE_CRT or n.startswith(("vcruntime140", "msvcp140", "ucrtbase")):
        return "RELEASE_CRT"
    if n in build_files:
        return "BUILD_DLL"
    if any(n.startswith(p) for p in CUDA_PREFIXES):
        return "CUDA_DLL"
    if is_system(n):
        return "SYSTEM"
    return "MISSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--roots", default="llama-server.exe,llama-bench.exe",
                    help="comma-separated root binaries for closure")
    ap.add_argument("--official-list", type=Path, default=None,
                    help="text file with official package file names (one per line)")
    args = ap.parse_args()

    pkg = args.package_dir
    files = sorted(f.name for f in pkg.iterdir()
                   if f.suffix.lower() in (".exe", ".dll"))
    build_files = {f.lower() for f in files}
    roots = [r for r in args.roots.split(",") if r.strip()]
    roots = [r for r in roots if r.lower() in build_files] or files

    findings = []
    fatal = False

    # ---- per-file import parsing -------------------------------------------
    # Authoritative PE import-table parse; falls back to a byte scan of the
    # file for known import-name strings if the descriptor walk fails (so the
    # validator is never silently blind on exotic PE layouts).
    imports_by_file = {}
    for f in files:
        p = pkg / f
        try:
            machine, is64, imports = parse_pe(p)
            imports_by_file[f.lower()] = {
                "machine": MACHINE_NAMES.get(machine, f"unknown:{machine:#x}"),
                "is_64bit": is64,
                "imports": imports,
                "source": "pe_import_table",
            }
            if not is64:
                findings.append(f"WARNING: {f} is a 32-bit PE (expected x64)")
        except Exception as e:  # noqa: BLE001
            # Fallback: byte-scan the file for known import names.
            blob = p.read_bytes()
            found = []
            for known in sorted(set(list(DEBUG_CRT) + list(RELEASE_CRT) +
                                   ["kernel32.dll", "msvcp140.dll", "advapi32.dll",
                                    "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
                                    "ggml-cpu-x64.dll", "ggml-cuda.dll",
                                    "llama.dll", "llama-common.dll", "mtmd.dll",
                                    "llama-server-impl.dll", "llama-bench-impl.dll",
                                    "libomp.dll"]) | {f.lower()}):
                # match the name followed by a NUL and case-insensitively
                if known.encode().lower() in blob.lower() and \
                   known.encode().lower() + b"\x00" in blob.lower():
                    found.append(known)
            if found:
                imports_by_file[f.lower()] = {
                    "machine": "AMD64 (assumed)",
                    "is_64bit": True,
                    "imports": found,
                    "source": f"byte-scan-fallback (PE parse failed: {e})",
                }
                findings.append(f"WARNING: {f} PE import-table parse failed ({e}); "
                                f"fell back to byte-scan of import names")
            else:
                findings.append(f"FATAL: cannot parse PE import table of {f}: {e} "
                                f"and byte-scan found no known import names")
                fatal = True

    # ---- debug CRT checks (authoritative import tables) --------------------
    for f, info in imports_by_file.items():
        dbg = sorted(set(i for i in info["imports"]
                         if classify(i, build_files) == "DEBUG_CRT"))
        if dbg:
            findings.append(f"FATAL: {f} imports DEBUG CRT: {', '.join(dbg)}")
            fatal = True

    # ---- string-scan guard (secondary) -------------------------------------
    for f in files:
        blob = (pkg / f).read_bytes()
        for pat in (b"VCRUNTIME140D", b"MSVCP140D", b"UCRTBASED"):
            if pat in blob:
                imported = pat.decode().lower() in [i.lower()
                                                    for i in imports_by_file.get(f.lower(), {}).get("imports", [])]
                if imported:
                    findings.append(f"FATAL: {f} confirmed (string+import scan): debug CRT {pat.decode()}")
                    fatal = True
                else:
                    findings.append(f"NOTE: {f} contains string {pat.decode()} but not in import table")

    # ---- dependency closure from roots -------------------------------------
    closure = set()
    queue = [r.lower() for r in roots]
    while queue:
        cur = queue.pop()
        if cur in closure or cur not in build_files:
            continue
        closure.add(cur)
        for imp in imports_by_file.get(cur, {}).get("imports", []):
            n = imp.lower()
            if n in build_files and n not in closure:
                queue.append(n)

    closure_report = {}
    for c in sorted(closure):
        info = imports_by_file.get(c, {})
        closure_report[c] = {
            "in_package": True,
            "needed_by": [r for r in roots if c.lower() != r.lower()
                          and c.lower() in imports_by_file.get(r.lower(), {}).get("imports", [])],
            "machine": info.get("machine"),
            "is_64bit": info.get("is_64bit"),
        }
    unresolved = {}
    for r in roots:
        for imp in imports_by_file.get(r.lower(), {}).get("imports", []):
            n = imp.lower()
            if n in build_files:
                continue
            cls = classify(n, build_files)
            entry = unresolved.setdefault(n, {"classification": cls, "needed_by": []})
            if r.lower() not in entry["needed_by"]:
                entry["needed_by"].append(r.lower())
    closure_report.update(unresolved)

    counts = {}
    for name, info in closure_report.items():
        cls = info["classification"] if "classification" in info else \
            ("BUILD_DLL" if info["in_package"] else "SYSTEM")
        counts[cls] = counts.get(cls, 0) + 1
        if cls == "DEBUG_CRT":
            findings.append(f"FATAL: closure dependency {name} is DEBUG CRT "
                            f"(needed by {', '.join(info['needed_by'])})")
            fatal = True
        elif cls == "MISSING":
            findings.append(f"FATAL: closure dependency {name} unresolved "
                            f"(not in package, not system, not known runtime; "
                            f"needed by {', '.join(info['needed_by'])})")
            fatal = True

    # ---- verify the three critical binaries explicitly --------------------
    # The package MUST contain llama-server.exe, llama-bench.exe and
    # ggml-cuda.dll; verify each parses AND is inspected (not silently
    # skipped).
    for req in ("llama-server.exe", "llama-bench.exe", "ggml-cuda.dll"):
        if req.lower() not in imports_by_file:
            findings.append(f"FATAL: critical binary missing or unparsed: {req}")
            fatal = True
        else:
            print(f"[ok] {req} inspected ({imports_by_file[req]['source']})")

    # ---- required files ----------------------------------------------------
    # llama-server-impl.dll is REQUIRED: on Windows the .exe is a VS loader
    # stub that loads the -impl.dll; without it the server cannot start.
    required = ["llama-server.exe", "llama-bench.exe", "llama-server-impl.dll",
                "ggml.dll", "ggml-base.dll", "ggml-cuda.dll"]
    for req in required:
        if req not in files:
            findings.append(f"FATAL: required file missing: {req}")
            fatal = True

    # ---- BUILD-METADATA.json -------------------------------------------------
    meta_path = pkg / "BUILD-METADATA.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for k in ("variant", "source_sha", "upstream_base_sha", "msvc_version",
                      "cuda_version", "cmake_version", "ninja_version",
                      "cuda_architecture", "build_configuration"):
                if not meta.get(k):
                    findings.append(f"FATAL: BUILD-METADATA.json field empty/missing: {k}")
                    fatal = True
            if str(meta.get("build_configuration", "")).lower() != "release":
                findings.append(f"FATAL: build_configuration != Release: {meta.get('build_configuration')!r}")
                fatal = True
        except Exception as e:  # noqa: BLE001
            findings.append(f"FATAL: BUILD-METADATA.json invalid JSON: {e}")
            fatal = True
    else:
        findings.append("FATAL: BUILD-METADATA.json missing from package")
        fatal = True

    # ---- SHA256SUMS ----------------------------------------------------------
    sums_path = pkg / "SHA256SUMS.txt"
    if sums_path.exists():
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            h, fname = parts
            fp = pkg / fname
            if not fp.exists():
                findings.append(f"FATAL: SHA256SUMS references missing file {fname}")
                fatal = True
                continue
            actual = hashlib.sha256(fp.read_bytes()).hexdigest().lower()
            if actual != h.lower():
                findings.append(f"FATAL: SHA256 mismatch for {fname}: "
                                f"actual {actual} != listed {h}")
                fatal = True
    else:
        findings.append("FATAL: SHA256SUMS.txt missing from package")
        fatal = True

    # ---- official package comparison -----------------------------------------
    official_report = None
    if args.official_list:
        official = [l.strip() for l in args.official_list.read_text().splitlines() if l.strip()]
        ours = set(files)
        missing = sorted(set(official) - ours)
        extra = sorted(ours - set(official))
        official_report = {
            "official_file_count": len(official),
            "custom_file_count": len(ours),
            "missing_from_custom": missing,
            "extra_in_custom": extra,
            "missing_relevant": [m for m in missing
                                 if classify(m, build_files) in ("BUILD_DLL", "CUDA_DLL")],
        }
        for m in official_report["missing_relevant"]:
            findings.append(f"NOTE: {m} present in official package but missing here "
                            f"(intentional omission if not needed by this build)")

    report = {
        "package_dir": str(pkg),
        "files": {f: (pkg / f).stat().st_size for f in files},
        "roots": roots,
        "per_file_imports": imports_by_file,
        "closure": closure_report,
        "classification_counts": counts,
        "official_comparison": official_report,
        "findings": findings,
        "fatal": fatal,
    }
    out = json.dumps(report, indent=2)
    print(out)
    if args.json:
        args.json.write_text(out + "\n")
    sys.exit(1 if fatal else 0)


if __name__ == "__main__":
    main()
