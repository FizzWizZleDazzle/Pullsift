#!/usr/bin/env python3
"""Baseline: flag first-time contributors, pass everyone else. The
crudest policy a maintainer can turn on today, as a scored predictor.

Usage: bench/baselines/first_timer.py [--corpus DIR] > predictions.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "corpus",
    )
    args = ap.parse_args()

    for line in (args.corpus / "inputs.jsonl").read_text().splitlines():
        r = json.loads(line)
        # Point in time: no visible prior PR to this repo when it opened.
        first = r.get("first_pr_to_repo", True)
        print(json.dumps({"id": r["id"], "score": 0.9 if first else 0.1}))


if __name__ == "__main__":
    main()
