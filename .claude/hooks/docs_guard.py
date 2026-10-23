"""PreToolUse hook (Write|Edit): keep project notes in the agreed files.

Blocks writes to any Markdown file in the repository except the allowed ones listed below, so new findings go into the
experiment log, the walkthrough or the article notes instead of new ad-hoc documents. docs/archive/ is read-only.
"""
import json
import os
import sys

ROOT = os.path.normcase(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ALLOWED = {
    "docs/experiment_log.md",
    "docs/algorithm_walkthrough.md",
    "docs/final_runbook.md",
    "docs/scoring.md",
    "docs/dataset_inventory.md",
    "docs/dataset_formats.md",
    "claude.md",
    "readme.md",
    "experiments/readme.md",
    "src/vcell/qc/readme.md",
    "src/vcell/etl/readme.md",
    "src/vcell/model/readme.md",
    "src/vcell/evaluation/readme.md",
}
IGNORED_DIRS = (".venv/", "data/", "eda/out/", ".claude/", ".git/")
ROUTING = (
    "Write it into the right existing file instead: experiments, runs, uploads, data checks and the current "
    "state/to-do -> docs/EXPERIMENT_LOG.md; a better algorithm that is adopted -> docs/ALGORITHM_WALKTHROUGH.md; "
    "build/packaging steps -> docs/FINAL_RUNBOOK.md. "
    "docs/archive/ is read-only (superseded docs). The experiment index is experiments/README.md, the project overview README.md."
)


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return
    path = (data.get("tool_input") or {}).get("file_path") or ""
    if not path.lower().endswith(".md"):
        return
    full = os.path.normcase(os.path.abspath(path))
    if not full.startswith(ROOT + os.sep):
        return
    rel = full[len(ROOT) + 1:].replace("\\", "/").lower()
    if rel.startswith(IGNORED_DIRS) or rel in ALLOWED:
        return
    why = "docs/archive/ is read-only." if rel.startswith("docs/archive/") else f"'{rel}' is not one of the project's note files."
    json.dump({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": f"{why} {ROUTING}",
    }}, sys.stdout)


if __name__ == "__main__":
    main()
