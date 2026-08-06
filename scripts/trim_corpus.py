#!/usr/bin/env python3
"""Apply corpus curation caps retroactively: label-mined slop is limited to
REPO_CAP_LABELED records per repo (mirrors the cap in mine_corpus.py), so a
single spam-farm repo cannot dominate the fit. Rewrites the jsonl in place.
"""

import json
from pathlib import Path

from mine_corpus import OUT_DIR, REPO_CAP_LABELED

def is_capped(source):
    return source.startswith(("label:", "window:")) or source in (
        "october-invalid",
        "invalid-window",
    )


def main():
    path = OUT_DIR / "slop.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    counts = {}
    kept = []
    for r in rows:
        if is_capped(r.get("source", "")):
            n = counts.get(r["repo"], 0)
            if n >= REPO_CAP_LABELED:
                continue
            counts[r["repo"]] = n + 1
        kept.append(r)
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"kept {len(kept)} of {len(rows)}")


if __name__ == "__main__":
    main()
