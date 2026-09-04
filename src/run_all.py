"""Reproduce every number in results/REPORT.md.

    python -m src.run_all                 # all phases (Phase 3 needs a GPU and cached HF models)
    python -m src.run_all --phase 1 2     # selected phases
    python -m src.run_all --report-only   # rebuild REPORT.md from results/*.json

Each phase writes results/phaseN.json; REPORT.md is regenerated from those files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess

from .common import RESULTS, env_note, ensure_dirs


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_report() -> str:
    from . import phase1, phase2, phase3
    parts = ["# LLM-embeds rerun — report", "",
             f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `python -m src.run_all` at git {_git_rev()}; {env_note()}.",
             "Seed 2026 throughout. Original notebooks untouched; code in `src/`.", ""]
    for n, mod in ((1, phase1), (2, phase2), (3, phase3)):
        p = RESULTS / f"phase{n}.json"
        if p.exists():
            parts.append(mod.report(json.load(open(p))))
        else:
            parts.append(f"## Phase {n}\n\n_Not run (no results/phase{n}.json)._\n")
    notes = RESULTS / "NOTES.md"
    if notes.exists():
        parts += ["## Notes and deviations", "", notes.read_text()]
    text = "\n".join(parts)
    (RESULTS / "REPORT.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--models", nargs="*", help="Phase 3: HF model ids to extract (default: auto)")
    args = ap.parse_args()
    ensure_dirs()
    if not args.report_only:
        if 1 in args.phase:
            from . import phase1
            phase1.run()
            build_report()
        if 2 in args.phase:
            from . import phase2
            phase2.run()
            build_report()
        if 3 in args.phase:
            from . import phase3
            phase3.run(models=args.models)
            build_report()
    text = build_report()
    print("\n".join(text.splitlines()[-30:]))


if __name__ == "__main__":
    main()
