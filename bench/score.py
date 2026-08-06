#!/usr/bin/env python3
"""Score a predictions file against the slopbench corpus.

Usage: bench/score.py <predictions.jsonl> [--corpus DIR] [--full] [--json]

Predictions are JSONL, one object per PR, either form:

    {"id": "owner/repo#number", "score": 0.87}
    {"repo": "owner/repo", "number": 123, "score": 0.87}

Scores are probabilities in [0, 1]; higher means more likely slop. The
report covers the held-out test split (see README for the split rule).
Pass --full to also report on the whole corpus. Every test-split record
needs a prediction; missing records fail the run so silent abstention
cannot inflate results.

No dependencies beyond the Python standard library.
"""

import argparse
import json
import sys
from pathlib import Path

FOLDS = 5
TEST_FOLD = 4
FPR_TARGETS = (0.05, 0.01, 0.001)


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def is_test(author: str) -> bool:
    return fnv1a64(author.lower().encode()) % FOLDS == TEST_FOLD


def load_corpus(corpus_dir: Path):
    records = []
    for name in ("slop.jsonl", "ham.jsonl"):
        path = corpus_dir / name
        if not path.exists():
            sys.exit(f"missing corpus file: {path}")
        for i, line in enumerate(path.read_text().splitlines(), 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{name}:{i}: bad JSON ({e})")
            records.append(
                {
                    "id": f"{r['repo']}#{r['number']}",
                    "is_slop": r["label"] == "slop",
                    "author": r["author"],
                    "source": r.get("source", ""),
                }
            )
    return records


def fingerprint(records) -> str:
    """Corpus revision id: hash of the sorted (id, label) pairs. Two
    reports are comparable only when their fingerprints match."""
    lines = sorted(f"{r['id']}\t{int(r['is_slop'])}" for r in records)
    joined = "\n".join(lines)
    return format(fnv1a64(joined.encode()), "016x")


def load_predictions(path: Path):
    preds = {}
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path.name}:{i}: bad JSON ({e})")
        pid = p.get("id") or f"{p.get('repo')}#{p.get('number')}"
        score = p.get("score")
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            sys.exit(f"{path.name}:{i}: score must be a number in [0, 1]")
        preds[pid] = float(score)
    return preds


def auc(pairs):
    """Rank-statistic AUC with half credit for ties."""
    pos = sorted(s for s, y in pairs if y)
    neg = sorted(s for s, y in pairs if not y)
    if not pos or not neg:
        return None
    wins = ties = 0
    import bisect

    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        wins += lo
        ties += hi - lo
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def threshold_at_fpr(neg_scores, target):
    """Highest cut whose FPR on neg_scores stays at or under target."""
    if not neg_scores:
        return 1.0
    ranked = sorted(neg_scores, reverse=True)
    allowed = int(target * len(ranked))
    if allowed == 0:
        return min(ranked[0] + 1e-9, 1.0)
    return min(ranked[allowed] + 1e-9, 1.0)


def evaluate(rows):
    """rows: list of (score, is_slop, source)."""
    pairs = [(s, y) for s, y, _ in rows]
    neg = [s for s, y in pairs if not y]
    n_pos = sum(1 for _, y in pairs if y)
    out = {
        "n": len(pairs),
        "n_slop": n_pos,
        "n_ham": len(neg),
        "auc": auc(pairs),
        "recall_at_fpr": {},
    }
    for target in FPR_TARGETS:
        t = threshold_at_fpr(neg, target)
        caught = sum(1 for s, y in pairs if y and s >= t)
        fp = sum(1 for s in neg if s >= t)
        out["recall_at_fpr"][str(target)] = {
            "threshold": t,
            "recall": caught / n_pos if n_pos else None,
            "observed_fpr": fp / len(neg) if neg else None,
        }
    op = threshold_at_fpr(neg, 0.01)
    per_source = {}
    for s, y, src in rows:
        if not y:
            continue
        hit, total = per_source.get(src, (0, 0))
        per_source[src] = (hit + (1 if s >= op else 0), total + 1)
    out["per_source_recall_at_1pct_fpr"] = {
        src: {"caught": h, "total": t} for src, (h, t) in sorted(per_source.items())
    }
    return out


def report(name, ev):
    print(f"== {name}: {ev['n']} PRs ({ev['n_slop']} slop, {ev['n_ham']} ham)")
    a = ev["auc"]
    print(f"  AUC: {a:.4f}" if a is not None else "  AUC: undefined (single class)")
    for target, r in ev["recall_at_fpr"].items():
        rec = r["recall"]
        rec_s = f"{rec:.3f}" if rec is not None else "n/a"
        print(
            f"  recall at {float(target):.1%} FPR: {rec_s}"
            f" (threshold {r['threshold']:.4f}, observed FPR {r['observed_fpr']:.4f})"
        )
    print("  per-source recall at 1% FPR:")
    for src, c in ev["per_source_recall_at_1pct_fpr"].items():
        print(f"    {src:20} {c['caught']}/{c['total']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().parent / "corpus",
    )
    ap.add_argument("--full", action="store_true", help="also report on the whole corpus")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    records = load_corpus(args.corpus)
    preds = load_predictions(args.predictions)

    test = [r for r in records if is_test(r["author"])]
    missing = [r["id"] for r in test if r["id"] not in preds]
    if missing:
        for m in missing[:10]:
            print(f"missing prediction: {m}", file=sys.stderr)
        sys.exit(f"{len(missing)} test-split records lack predictions; aborting")

    rows_test = [(preds[r["id"]], r["is_slop"], r["source"]) for r in test]
    results = {"corpus_fingerprint": fingerprint(records), "test": evaluate(rows_test)}
    if args.full:
        covered = [r for r in records if r["id"] in preds]
        if len(covered) < len(records):
            print(
                f"note: --full covers {len(covered)}/{len(records)} records",
                file=sys.stderr,
            )
        rows_all = [(preds[r["id"]], r["is_slop"], r["source"]) for r in covered]
        results["full"] = evaluate(rows_all)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"corpus fingerprint: {results['corpus_fingerprint']}")
        report("test split", results["test"])
        if "full" in results:
            report("full corpus", results["full"])


if __name__ == "__main__":
    main()
