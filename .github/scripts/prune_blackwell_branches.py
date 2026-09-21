#!/usr/bin/env python3
"""Archive and remove only branch/SHA pairs from the reviewed snapshot."""
import json
import os
from pathlib import Path
import subprocess
import urllib.request

KEEP = {"main", "blackwell/cpasync", "blackwell/cutlass", "blackwell/tma"}
PREFIX = "archive/branches-20260921/"


def git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def remote_refs(pattern):
    return {ref: sha for sha, ref in (line.split() for line in git("ls-remote", "origin", pattern).splitlines())}


def run():
    if os.environ.get("GITHUB_REPOSITORY") != "brx19/llama.cpp":
        raise RuntimeError("This cleanup is restricted to brx19/llama.cpp")
    request = urllib.request.Request("https://api.github.com/repos/brx19/llama.cpp", headers={
        "Authorization": "Bearer " + os.environ["GH_TOKEN"]})
    with urllib.request.urlopen(request, timeout=30) as response:
        default = json.load(response)["default_branch"]
    snapshot = json.loads(Path(".github/blackwell-branch-archive.json").read_text())
    current = remote_refs("refs/heads/*")
    tags = remote_refs("refs/tags/" + PREFIX + "*")
    targets = []
    for name, sha in snapshot.items():
        if name in KEEP or name == default:
            continue
        ref = "refs/heads/" + name
        if ref not in current:
            continue
        if current[ref] != sha:
            raise RuntimeError(f"Branch moved since review: {name}; refusing cleanup")
        tag = "refs/tags/" + PREFIX + name
        if tag in tags and tags[tag] != sha:
            raise RuntimeError(f"Archive tag already has another SHA: {tag}")
        targets.append((ref, tag, sha))
    print(f"Keeping {sorted(KEEP)} and current default {default}; archiving {len(targets)} branches", flush=True)
    if not targets:
        return
    # Fetch objects without touching the release tag namespace.
    git("fetch", "--no-tags", "origin", "+refs/heads/*:refs/remotes/cleanup/*")
    for offset in range(0, len(targets), 40):
        batch = targets[offset:offset + 40]
        leases = [f"--force-with-lease={ref}:{sha}" for ref, _, sha in batch]
        updates = []
        for ref, tag, sha in batch:
            updates.extend([f"{sha}:{tag}", f":{ref}"])
        # Archive and delete together. Concurrent branch updates reject the batch.
        git("push", "--atomic", *leases, "origin", *updates)
        print(f"Archived and removed {min(offset + 40, len(targets))}/{len(targets)}", flush=True)
    remaining = remote_refs("refs/heads/*")
    missing = KEEP - {ref.removeprefix("refs/heads/") for ref in remaining}
    if missing:
        raise RuntimeError(f"Required branches missing: {missing}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"Archived and removed {len(targets)} branches. Remaining: {len(remaining)}. Archive tags: `{PREFIX}*`.\n")


if __name__ == "__main__":
    run()
