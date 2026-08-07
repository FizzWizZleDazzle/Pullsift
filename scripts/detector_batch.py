#!/usr/bin/env python3
"""Score a corpus archive with the self-hosted AI-text detector, writing
a sidecar the tuner reads: {corpus-dir}/detector.jsonl with one
{"id", "probability"} per scorable record. Records without enough prose
are omitted (the rule stays silent for them).

Usage: detector_batch.py <corpus-dir> [more-dirs...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_common import extract_prose, load_model, usable


def main():
    _, _, score = load_model()
    for dir_arg in sys.argv[1:]:
        d = Path(dir_arg)
        sidecar = d / "detector.jsonl"
        done = {}
        if sidecar.exists():
            for line in sidecar.read_text().splitlines():
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        out = []
        n_skipped = n_cached = 0
        for name in ("slop.jsonl", "ham.jsonl"):
            path = d / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                rid = f"{r['repo']}#{r['number']}"
                if rid in done:
                    out.append(done[rid])
                    n_cached += 1
                    continue
                prose = extract_prose(f"{r.get('title', '')}\n{r.get('body', '')}")
                if not usable(prose):
                    n_skipped += 1
                    continue
                out.append({"id": rid, "probability": round(score(prose), 6)})
        sidecar.write_text("".join(json.dumps(r) + "\n" for r in out))
        print(
            f"{d}: {len(out)} scored ({n_cached} cached), abstained {n_skipped} (short prose)"
        )


if __name__ == "__main__":
    main()
