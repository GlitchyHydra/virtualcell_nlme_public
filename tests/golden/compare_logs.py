"""Compare the numbers two logs print, in order (old eda/ script vs the ported experiment).

    .venv/Scripts/python.exe tests/golden/compare_logs.py OLD.log NEW.log [--skip REGEX ...] [--grep REGEX]

Lines matching a skip pattern (timings, paths, progress) are ignored; every other number must agree exactly as printed.
--grep keeps only the lines matching REGEX in both logs (for a script split into several experiments).
Exit 0 and "same numbers (N)" if they agree, else the first differing line pair.
"""
from __future__ import annotations

import re
import sys

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf")
SKIP = [r"\d+(\.\d+)? ?(min|s)\b", r"\bin \d+(\.\d+)?s", r"[A-Za-z]:[\\/]", r"^\s*(cells|file|block) \d",
        r"^\s*pass( \d)?: \d+M entries",
        r"targets?, \d+\.\d+G stored", r"\d+/\d+ targets", r"saved ", r"wrote "]


def numbers(path, skip, grep=None):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if grep and not re.search(grep, line):
            continue
        if any(re.search(s, line) for s in skip):
            continue
        nums = NUM.findall(line.replace("_", ""))
        if nums:
            out.append((nums, line.rstrip()))
    return out


def main(argv) -> int:
    a, b = argv[0], argv[1]
    skip = SKIP + [argv[i + 1] for i, x in enumerate(argv) if x == "--skip"]
    grep = next((argv[i + 1] for i, x in enumerate(argv) if x == "--grep"), None)
    A, B = numbers(a, skip, grep), numbers(b, skip, grep)
    for k, ((na, la), (nb, lb)) in enumerate(zip(A, B)):
        if na != nb:
            print(f"line {k}: differ\n  old: {la}\n  new: {lb}")
            return 1
    if len(A) != len(B):
        print(f"different number of numeric lines: old {len(A)}, new {len(B)}")
        extra = A[len(B):] or B[len(A):]
        print("  first extra:", extra[0][1] if extra else "")
        return 1
    print(f"same numbers ({sum(len(n) for n, _ in A)} values on {len(A)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
