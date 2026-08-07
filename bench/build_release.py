#!/usr/bin/env python3
"""Build the released benchmark files from the curation archive.

Reads bench/corpus/archive/{slop,ham}.jsonl and writes:

- bench/corpus/inputs.jsonl: what a bot reads. Outcome fields (label,
  source, state, merged, pr_labels) are removed, the author dossier is
  filtered to history that existed when the PR was opened, dossier nodes
  in the scored PR's own repo lose their outcome fields, and repo stars
  are coarsened to an order of magnitude.
- bench/corpus/labels.jsonl: what the scorer reads. id, label, source,
  author.

Predictions must be computed from inputs.jsonl alone; the archive keeps
raw records for curation and audit.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

# author_association is an outcome field: GitHub computes it at read time,
# so merging promotes the author from NONE to CONTRIBUTOR. In the archive
# it separates the classes almost perfectly, which is a record of the
# outcome, not a signal available when the PR arrived. It is replaced by
# point-in-time fields reconstructed from history predating the PR.
DROP = (
    "label",
    "source",
    "state",
    "merged",
    "pr_labels",
    "repo_stars",
    "audit",
    "kind",
    "author_association",
)


def point_in_time_history(record):
    """What the author's public record looked like when this PR opened."""
    user = ((record.get("dossier") or {}).get("data") or {}).get("user") or {}
    nodes = (user.get("pullRequests") or {}).get("nodes") or []
    cutoff = record.get("created_at") or ""
    repo = record["repo"].lower()
    prior = prior_here = merged_here = 0
    for n in nodes:
        created = n.get("createdAt") or ""
        if not cutoff or not created or created >= cutoff:
            continue
        prior += 1
        if ((n.get("repository") or {}).get("nameWithOwner") or "").lower() == repo:
            prior_here += 1
            if n.get("merged"):
                merged_here += 1
    return {
        # No visible prior PR to this repo when the PR was opened.
        "first_pr_to_repo": prior_here == 0,
        "prior_prs_visible": prior,
        "prior_prs_this_repo": prior_here,
        "prior_merged_this_repo": merged_here,
    }


def scrub_dossier(record):
    dossier = record.get("dossier")
    user = ((dossier or {}).get("data") or {}).get("user")
    if not user:
        return dossier
    prs = user.get("pullRequests") or {}
    nodes = prs.get("nodes") or []
    cutoff = record.get("created_at") or ""
    kept = []
    for n in nodes:
        created = n.get("createdAt") or ""
        if cutoff and created and created >= cutoff:
            continue
        n = dict(n)
        repo = ((n.get("repository") or {}).get("nameWithOwner") or "").lower()
        if repo == record["repo"].lower():
            for k in ("labels", "merged", "state"):
                n.pop(k, None)
        kept.append(n)
    user = dict(user)
    user["pullRequests"] = {"nodes": kept}
    return {"data": {"user": user}}


def main():
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "corpus"
    archive = corpus / "archive"
    inputs, labels = [], []
    for name in ("slop.jsonl", "ham.jsonl"):
        path = archive / name
        if not path.exists():
            sys.exit(f"missing {path}")
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rid = f"{r['repo']}#{r['number']}"
            labels.append(
                {
                    "id": rid,
                    "label": r["label"],
                    "source": r.get("source", ""),
                    "author": r["author"],
                    # Audit-assigned slop kind: ai | human | unclear.
                    # Empty when not yet tagged.
                    "kind": r.get("kind", ""),
                }
            )
            slim = {k: v for k, v in r.items() if k not in DROP}
            slim["id"] = rid
            stars = r.get("repo_stars") or 0
            slim["repo_stars_magnitude"] = int(math.log10(stars)) if stars > 0 else 0
            slim.update(point_in_time_history(r))
            slim["dossier"] = scrub_dossier(r)
            inputs.append(slim)

    (corpus / "inputs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in inputs)
    )
    (corpus / "labels.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in labels)
    )

    by_source = Counter((l["label"], l["source"]) for l in labels)
    n_slop = sum(1 for l in labels if l["label"] == "slop")
    print(f"released {len(inputs)} records ({n_slop} slop, {len(labels) - n_slop} ham)")

    # Health checks: the anti-shortcut sources must exist, or the corpus
    # contradicts the README and provenance shortcuts go unpunished.
    required = ("agent-merged", "ai-topic-merged")
    missing = [s for s in required if by_source.get(("ham", s), 0) == 0]
    if missing:
        sys.exit(f"anti-shortcut ham sources empty: {missing}; refusing to release")

    # Date-separation report: eras should overlap or a date feature scores.
    months = {"slop": Counter(), "ham": Counter()}
    by_id = {f"{r['id']}": r for r in inputs}
    for l in labels:
        m = (by_id[l["id"]].get("created_at") or "")[:7]
        months[l["label"]][m] += 1
    overlap = sum(
        min(months["slop"][m], months["ham"][m])
        for m in set(months["slop"]) | set(months["ham"])
    )
    print(f"date overlap: {overlap} records share a month across labels")


if __name__ == "__main__":
    main()
