#!/usr/bin/env python3
"""Evaluate a Phase 8 evidence file and exit non-zero while safe mode is required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.operations.release_gate import evaluate_release_evidence, load_release_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_release_evidence(load_release_evidence(args.evidence))
    except ValueError as exc:
        print(json.dumps({"release_gate_passed": False, "decision": "KEEP_SAFE_MODE", "error": str(exc)}, indent=2))
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["release_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
