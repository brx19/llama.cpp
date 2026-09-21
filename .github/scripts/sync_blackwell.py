#!/usr/bin/env python3
"""Merge upstream without importing its workflows, branches, or tags."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

BRANCHES = ("main", "blackwell/cpasync", "blackwell/cutlass", "blackwell/tma")
TOOL_FILES = (
    ".github/scripts/sync_blackwell.py",
    ".github/scripts/prune_blackwell_branches.py",
    ".github/blackwell-branch-archive.json",
    ".github/official-win-cuda-13.3-x64-files.txt",
    ".github/scripts/extract_tma_packages.py",
    ".github/scripts/pe_validate.py",
    "scripts/Sync-Upstream.ps1",
    "BLACKWELL-BUILD.md",
    "BLACKWELL-FINAL-REPORT.md",
    "BLACKWELL-MAINTENANCE.md",
)


def git(*args, cwd=None, check=True):
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)}\n{result.stdout}{result.stderr}")
    return result


def output(key, value):
    print(f"{key}={value}", flush=True)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def tooling(path):
    return path.startswith(".github/workflows/") or path in TOOL_FILES


def merge_source(work, target, control, main=False):
    result = git("-c", "user.name=github-actions[bot]", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                 "merge", "--no-ff", "--no-commit", target, cwd=work, check=False)
    print(result.stdout + result.stderr, flush=True)
    conflicts = git("diff", "--name-only", "--diff-filter=U", cwd=work).stdout.splitlines()
    source_conflicts = [p for p in conflicts if not tooling(p)]
    if source_conflicts:
        git("merge", "--abort", cwd=work)
        raise RuntimeError("Source conflicts; remote branch unchanged: " + ", ".join(source_conflicts))
    if result.returncode and not conflicts:
        raise RuntimeError("Merge failed without resolvable workflow conflicts")
    # The fork owns all workflow files, including upstream additions and deletions.
    git("rm", "-r", "-f", "--ignore-unmatch", ".github/workflows", cwd=work)
    git("restore", f"--source={control}", "--staged", "--worktree", "--", ".github/workflows", cwd=work)
    for path in TOOL_FILES:
        if git("cat-file", "-e", f"{control}:{path}", cwd=work, check=False).returncode == 0:
            git("restore", f"--source={control}", "--staged", "--worktree", "--", path, cwd=work)
    if main:
        differences = git("diff", "--cached", "--name-only", target, cwd=work).stdout.splitlines()
        unexpected = [p for p in differences if not tooling(p)]
        if unexpected:
            raise RuntimeError("Main source parity check failed: " + ", ".join(unexpected))
    pending_merge = git("rev-parse", "-q", "--verify", "MERGE_HEAD", cwd=work, check=False).returncode == 0
    staged_changes = git("diff", "--cached", "--quiet", cwd=work, check=False).returncode != 0
    if pending_merge or staged_changes:
        git("-c", "user.name=github-actions[bot]", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit", "-m", f"ci: sync upstream {target[:12]} while preserving Blackwell tooling", cwd=work)
    return git("rev-parse", "HEAD", cwd=work).stdout.strip()


def run(branch, upstream_sha=None, publish=False):
    if branch not in BRANCHES:
        raise ValueError(f"Unsupported branch: {branch}")
    origin = git("remote", "get-url", "origin").stdout.strip().removesuffix(".git")
    if origin not in ("https://github.com/brx19/llama.cpp", "git@github.com:brx19/llama.cpp"):
        raise RuntimeError("Refusing to modify an origin other than brx19/llama.cpp")
    shallow = git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    fetch = ["fetch", "--no-tags"] + (["--unshallow"] if shallow else [])
    refs = [f"+refs/heads/{b}:refs/remotes/origin/{b}" for b in sorted({"main", branch})]
    git(*fetch, "origin", *refs)
    git("fetch", "--no-tags", "https://github.com/ggml-org/llama.cpp.git", "+refs/heads/master:refs/remotes/upstream/master")
    target = git("rev-parse", "refs/remotes/upstream/master").stdout.strip()
    if upstream_sha:
        git("merge-base", "--is-ancestor", upstream_sha, target)
        target = upstream_sha
    before = git("rev-parse", f"refs/remotes/origin/{branch}").stdout.strip()
    control = git("rev-parse", "refs/remotes/origin/main").stdout.strip()
    with tempfile.TemporaryDirectory(prefix="blackwell-sync-") as temp:
        work = str(Path(temp) / "worktree")
        git("worktree", "add", "--detach", work, before)
        try:
            after = merge_source(work, target, control, main=branch == "main")
            git("merge-base", "--is-ancestor", before, after)
            if publish and after != before:
                # A normal push rejects concurrent changes and never rewrites history.
                git("push", "origin", f"{after}:refs/heads/{branch}")
            output("upstream_sha", target)
            output("source_sha", after)
            output("changed", str(after != before).lower())
        finally:
            git("worktree", "remove", "--force", work, check=False)
    print(f"{branch}: {'published' if publish else 'dry run'}; {before} -> {after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=BRANCHES, default="main")
    parser.add_argument("--upstream-sha")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    run(args.branch, args.upstream_sha, args.push)
