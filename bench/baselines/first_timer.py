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

    for name in ("slop.jsonl", "ham.jsonl"):
        for line in (args.corpus / name).read_text().splitlines():
            r = json.loads(line)
            first = r.get("author_association") in ("FIRST_TIME_CONTRIBUTOR", "NONE")
            print(
                json.dumps(
                    {
                        "id": f"{r['repo']}#{r['number']}",
                        "score": 0.9 if first else 0.1,
                    }
                )
            )


if __name__ == "__main__":
    main()
