#!/usr/bin/env python3
"""Baseline: score by author account age alone. Younger account, higher
score. This is the 'autoclose PRs from new accounts' policy as a scored
predictor; a real bot needs to beat it.

Usage: bench/baselines/account_age.py [--corpus DIR] > predictions.jsonl
"""

import argparse
import json
from datetime import datetime, timezone
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
        score = 0.75  # unknown age reads as suspicious, not neutral
        user = ((r.get("dossier") or {}).get("data") or {}).get("user") or {}
        created = user.get("createdAt")
        pr_at = r.get("created_at")
        if created and pr_at:
            try:
                c = datetime.fromisoformat(created.replace("Z", "+00:00"))
                p = datetime.fromisoformat(pr_at.replace("Z", "+00:00"))
                age_days = max((p - c).total_seconds() / 86400.0, 0.0)
                score = 1.0 / (1.0 + age_days / 90.0)
            except ValueError:
                pass
        print(json.dumps({"id": r["id"], "score": round(score, 6)}))


if __name__ == "__main__":
    main()
