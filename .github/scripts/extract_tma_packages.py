#!/usr/bin/env python3
"""Extract the 3 TMA package ZIPs into a staging dir.

Used by the merge-tma-package job. Python's zipfile is more robust against
Windows file-locks than PowerShell's Expand-Archive.

Usage: python extract_tma_packages.py <src_dir> <stage_dir>
"""
import zipfile
import os
import sys


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_dir> <stage_dir>")
        sys.exit(1)

    src = sys.argv[1]
    stage = sys.argv[2]

    os.makedirs(stage, exist_ok=True)

    for name in ['cpu.zip', 'cuda.zip', 'cudart.zip']:
        path = os.path.join(src, name)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)
        print(f"Extracting {path} -> {stage}")
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(stage)
        print(f"Done {name}")

    print("All extraction complete.")


if __name__ == '__main__':
    main()
