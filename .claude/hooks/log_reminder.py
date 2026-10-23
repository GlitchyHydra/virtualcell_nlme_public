"""Stop hook: ask for an experiment-log entry when pipeline or experiment work is newer than docs/EXPERIMENT_LOG.md.

Looks at the code in src/vcell/ and experiments/ (recursive) and the logs/results in out/ and out/experiments/
(non-recursive; eda/out/ too while it exists). If any is newer than the log, Claude is asked once
to record the run (and, where it applies, the walkthrough and the article notes). It nags once per state: the newest
mtime it reminded about is kept in .claude/hooks/.last_reminder, and a stop that is already a reminder continuation
is let through.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(ROOT, "docs", "EXPERIMENT_LOG.md")
STATE = os.path.join(ROOT, ".claude", "hooks", ".last_reminder")
RESULTS = (".log", ".json", ".csv", ".txt")
WATCH = [(os.path.join(ROOT, "src", "vcell"), (".py",), True), (os.path.join(ROOT, "experiments"), (".py",), True),
         (os.path.join(ROOT, "out"), RESULTS, False), (os.path.join(ROOT, "out", "experiments"), RESULTS, False),
         (os.path.join(ROOT, "eda", "out"), RESULTS, False)]
SLACK = 60  # seconds
ACTIVE = 120  # seconds: a file written this recently belongs to a job still running, judged once it goes quiet


def files(folder, recursive):
    if recursive:
        for d, _, names in os.walk(folder):
            if "__pycache__" not in d:
                for n in names:
                    yield os.path.join(d, n)
        return
    try:
        yield from (e.path for e in os.scandir(folder) if e.is_file())
    except OSError:
        return


def newest():
    best, name, now = 0.0, "", time.time()
    for folder, exts, recursive in WATCH:
        for path in files(folder, recursive):
            if path.lower().endswith(exts):
                m = os.path.getmtime(path)
                if m > best and now - m > ACTIVE:
                    best, name = m, os.path.relpath(path, ROOT).replace("\\", "/")
    return best, name


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = {}
    if data.get("stop_hook_active"):
        return
    work, name = newest()
    try:
        logged = os.path.getmtime(LOG)
    except OSError:
        logged = 0.0
    if work <= logged + SLACK:
        return
    try:
        with open(STATE) as f:
            if abs(float(f.read().strip() or 0) - work) < 1:
                return
    except (OSError, ValueError):
        pass
    with open(STATE, "w") as f:
        f.write(str(work))
    json.dump({"decision": "block", "reason": (
        f"{name} changed after docs/EXPERIMENT_LOG.md. If an experiment, run, upload or data check happened, add it to "
        "docs/EXPERIMENT_LOG.md (next H/E/L/F number, what changed, numbers, verdict) and update its 'Current state' "
        "section. If a better algorithm was adopted, update docs/ALGORITHM_WALKTHROUGH.md; "
        "Do not create new docs. If nothing worth logging happened, just stop."
    )}, sys.stdout)


if __name__ == "__main__":
    main()
