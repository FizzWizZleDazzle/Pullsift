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
        out = []
        n_skipped = 0
        for name in ("slop.jsonl", "ham.jsonl"):
            path = d / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                r = json.loads(line)
                prose = extract_prose(f"{r.get('title', '')}\n{r.get('body', '')}")
                if not usable(prose):
                    n_skipped += 1
                    continue
                out.append(
                    {
                        "id": f"{r['repo']}#{r['number']}",
                        "probability": round(score(prose), 6),
                    }
                )
        (d / "detector.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in out)
        )
        print(f"{d}: scored {len(out)}, abstained {n_skipped} (short prose)")


if __name__ == "__main__":
    main()
