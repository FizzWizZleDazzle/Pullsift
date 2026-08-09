#!/usr/bin/env python3
"""Bring already-collected records up to the current record format.

The miner's format grows: a bigger diff cap, commit timestamps for the
authoring-rate rule. Records collected earlier keep whatever they were
collected with, and a field that exists on only the newest records is
worse than a field that exists nowhere. A rule keyed on it would learn
which mining pass a record came from.

So this walks the archive and re-fetches what is missing or stale, in
place. Safe to re-run: each record records the cap it was fetched under,
and a run after a network failure picks up only what is still outstanding.

Runs on a machine with an authenticated gh CLI.

Usage: scripts/backfill_records.py [--dir PATH]
"""

import argparse
import json
import collections
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_corpus import BODY_CAP, DIFF_CAP, gh_api  # noqa: E402

WORKERS = 6
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "bench" / "corpus" / "archive"


def needs_diff(r):
    return r.get("diff_cap", 0) < DIFF_CAP


def needs_commit_dates(r):
    commits = r.get("commits") or []
    return bool(commits) and any(not c.get("date") for c in commits)


def backfill(record, lock, counts):
    repo, number = record["repo"], record["number"]

    if needs_diff(record):
        diff = gh_api(
            f"repos/{repo}/pulls/{number}",
            accept="application/vnd.github.diff",
            ok_statuses=("404", "406", "422"),
        )
        with lock:
            if diff is None:
                counts["diff failed"] += 1
            else:
                if len(diff) > len(record.get("diff") or ""):
                    counts["diff grew"] += 1
                record["diff"] = diff[:DIFF_CAP]
                record["diff_bytes"] = len(diff)
                record["diff_cap"] = DIFF_CAP

    if needs_commit_dates(record):
        raw = gh_api(f"repos/{repo}/pulls/{number}/commits?per_page=50", ok_statuses=("404",))
        with lock:
            if raw is None:
                counts["dates failed"] += 1
                return
            try:
                fetched = json.loads(raw)
            except json.JSONDecodeError:
                counts["dates failed"] += 1
                return
            # Match on commit message: the API returns the same commits in
            # the same order, but a force-push between runs can change that.
            by_message = {
                (c["commit"].get("message") or "")[:BODY_CAP]: (c["commit"].get("author") or {})
                for c in fetched
            }
            filled = 0
            for i, c in enumerate(record.get("commits") or []):
                author = by_message.get(c.get("message", ""))
                if author is None and i < len(fetched):
                    author = fetched[i]["commit"].get("author") or {}
                if author and author.get("date"):
                    c["date"] = author["date"]
                    filled += 1
            counts["dates filled" if filled else "dates absent"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    for name in ("slop", "ham"):
        path = args.dir / f"{name}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        stale = [r for r in records if needs_diff(r) or needs_commit_dates(r)]
        print(f"{name}: {len(stale)} of {len(records)} to backfill", file=sys.stderr)
        if not stale:
            continue

        lock = threading.Lock()
        counts = collections.Counter()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda r: backfill(r, lock, counts), stale))

        tmp = path.with_suffix(".jsonl.new")
        with tmp.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        tmp.replace(path)
        print(f"{name}: {dict(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
