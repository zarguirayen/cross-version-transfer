"""Verify the pre-registration chain of the repository.

For every study, the file ``docs/PREREG*_SHA256.txt`` lists the SHA-256 hashes of the protocol and of the analysis
code as they were when the study was frozen by its git tag (``prereg-v1`` ... ``prereg-v8``). This script checks
two things:

1. **Each freeze is intact**: every file listed in a hash file, read *at its tag*, has the recorded hash.
2. **The frozen library was only extended afterwards**: for the shared library files (``src/cvt/*.py``) that
   later studies extended with new functions, no line that existed at the tag has been removed or changed in the
   current version. The estimators used by an earlier study are therefore still present, unchanged, at ``HEAD``.

Files are compared at the tag rather than in the working tree because ``src/cvt/graph.py`` grew from one study to
the next (v3 added the model-averaged estimator, v4 GREG, v5 cross-fitted GREG, v7 learned neighbours); its current
version matches only the last hash.

Usage
-----
    python tools/verify_preregistration.py          # exit code 0 if every check passes

Requires ``git`` and the tags ``prereg-v1`` to ``prereg-v8`` (``git fetch --tags`` after cloning).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (git tag, hash file) for each pre-registered study.
FREEZES = [
    ("prereg-v1", "docs/PREREG_SHA256.txt"),
    ("prereg-v2", "docs/PREREG_V2_SHA256.txt"),
    ("prereg-v3", "docs/PREREG_V3_SHA256.txt"),
    ("prereg-v4", "docs/PREREG_V4_SHA256.txt"),
    ("prereg-v5", "docs/PREREG_V5_SHA256.txt"),
    ("prereg-v6", "docs/PREREG_V6_SHA256.txt"),
    ("prereg-v7", "docs/PREREG_V7_SHA256.txt"),
    ("prereg-v8", "docs/PREREG_V8_SHA256.txt"),
]


def git(*args: str) -> bytes:
    """Run a git command in the repository and return its raw standard output."""
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def file_at(tag: str, path: str) -> bytes:
    """Content of ``path`` as committed at ``tag``."""
    return git("show", f"{tag}:{path}")


def hashed_files(tag: str, hash_file: str) -> tuple[list[tuple[str, str]], bool]:
    """Return the ``(sha256, path)`` pairs listed in ``hash_file``, and whether the file was committed with the tag.

    For v1 and v2 the hash file was committed shortly after the tag (with the results), so it is read from the
    current version; the hashes are still checked against the files *as committed at the tag*. Lines that are not
    hash entries (for example the commits of the external data sources) are skipped.
    """
    try:
        text, in_tag = file_at(tag, hash_file).decode(), True
    except subprocess.CalledProcessError:
        text, in_tag = (ROOT / hash_file).read_text(), False
    return re.findall(r"^([0-9a-f]{64})\s+(\S+)$", text, flags=re.M), in_tag


def removed_lines(tag: str, path: str) -> int:
    """Number of lines of ``path`` present at ``tag`` and removed or modified in the working tree."""
    diff = git("diff", "--unified=0", tag, "--", path).decode()
    return sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))


def main() -> int:
    """Run both checks for every freeze, print one line per check and return the number of failures."""
    failures = 0
    for tag, hash_file in FREEZES:
        try:
            git("rev-parse", "--verify", f"{tag}^{{commit}}")
        except subprocess.CalledProcessError:
            print(f"{tag}: tag missing (run `git fetch --tags`)")
            failures += 1
            continue
        entries, in_tag = hashed_files(tag, hash_file)
        bad = [p for h, p in entries if hashlib.sha256(file_at(tag, p)).hexdigest() != h]
        where = "" if in_tag else " (hash file committed after the tag)"
        print(f"{tag}: {len(entries) - len(bad)}/{len(entries)} hashes match the files at the tag{where}"
              + (f" — MISMATCH {bad}" if bad else ""))
        failures += len(bad)
        for _, path in entries:
            if path.startswith("src/cvt/"):
                n = removed_lines(tag, path)
                if n:
                    print(f"    {path}: {n} line(s) of the frozen version changed since {tag}")
                    failures += 1
    print("all checks passed" if failures == 0 else f"{failures} failure(s)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
