#!/usr/bin/env python3
"""Mine a labeled PR corpus from the GitHub API into bench/corpus/.

Runs on a machine with an authenticated gh CLI. Slop positives are curated:
closed and unmerged, human author, author is not the repo owner, and the
repo clears a star floor. Ham comes from the same repos so the model cannot
learn repo identity. Re-runs skip PRs already collected.

Usage: scripts/mine_corpus.py [--limit-total N]
"""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKERS = 6
# Label-mined sources are capped per repo so one spam-farm repo cannot
# dominate the corpus. The express flood is intentionally uncapped: the
# whole wave is the point.
REPO_CAP_LABELED = 15

OUT_DIR = Path(__file__).resolve().parent.parent / "bench" / "corpus"
STAR_FLOOR_LABELED = 100
STAR_FLOOR_AGENT = 25
BODY_CAP = 4096
DIFF_CAP = 8192

DOSSIER_GQL = """
query($login: String!) {
  user(login: $login) {
    createdAt
    bio
    followers { totalCount }
    contributionsCollection { restrictedContributionsCount }
    pullRequests(last: 50) {
      nodes {
        state
        merged
        createdAt
        labels(first: 10) { nodes { name } }
        reviews(first: 1) { totalCount }
        comments(first: 50) { nodes { author { login } createdAt } }
        repository { nameWithOwner primaryLanguage { name } }
      }
    }
  }
}
"""


def sh(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}: {r.stderr.strip()[:300]}")
    return r


def gh_json(args):
    out = sh(["gh"] + args).stdout
    return json.loads(out) if out.strip() else None


def gh_api(path, accept=None, ok_statuses=()):
    args = ["gh", "api", path]
    if accept:
        args += ["--header", f"Accept: {accept}"]
    r = sh(args, check=False)
    if r.returncode != 0:
        err = r.stderr.strip()
        if any(f"HTTP {s}" in err for s in ok_statuses):
            return None
        if "HTTP 403" in err or "rate limit" in err.lower():
            print("  rate limited; sleeping 90s", file=sys.stderr)
            time.sleep(90)
            return gh_api(path, accept, ok_statuses)
        print(f"  skip {path}: {err[:160]}", file=sys.stderr)
        return None
    return r.stdout


def search_prs(extra_args, limit, retries=2):
    fields = "number,repository,author,title,body,createdAt,authorAssociation,state,labels"
    for attempt in range(retries + 1):
        try:
            return (
                gh_json(
                    ["search", "prs", "--limit", str(limit), "--json", fields]
                    + extra_args
                )
                or []
            )
        except RuntimeError as e:
            if "rate limit" in str(e).lower() and attempt < retries:
                print("  search rate limited; sleeping 120s", file=sys.stderr)
                time.sleep(120)
                continue
            print(f"  search failed ({e})", file=sys.stderr)
            return []
    return []


class Miner:
    def __init__(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.seen = set()
        self.star_cache = {}
        self.dossier_cache = {}
        self.files = {
            "slop": open(OUT_DIR / "slop.jsonl", "a"),
            "ham": open(OUT_DIR / "ham.jsonl", "a"),
        }
        for name in ("slop", "ham"):
            p = OUT_DIR / f"{name}.jsonl"
            if p.exists():
                for line in p.read_text().splitlines():
                    try:
                        r = json.loads(line)
                        self.seen.add((r["repo"], r["number"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
        self.counts = {}
        self.repo_counts = {}
        # Seed repo counts from what previous runs already collected.
        p = OUT_DIR / "slop.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                    if r.get("source", "").startswith("label:") or r.get("source") in ("october-invalid", "invalid-window"):
                        self.repo_counts[r["repo"]] = self.repo_counts.get(r["repo"], 0) + 1
                except (json.JSONDecodeError, KeyError):
                    pass

    def stars(self, repo):
        with self.lock:
            if repo in self.star_cache:
                return self.star_cache[repo]
        raw = gh_api(f"repos/{repo}", ok_statuses=("404",))
        val = json.loads(raw).get("stargazers_count", 0) if raw else -1
        with self.lock:
            self.star_cache[repo] = val
        return val

    def dossier(self, login):
        with self.lock:
            cached = self.dossier_cache.get(login)
        if cached is not None:
            return cached
        if True:
            r = sh(
                ["gh", "api", "graphql", "-f", f"query={DOSSIER_GQL}", "-F", f"login={login}"],
                check=False,
            )
            resp = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
            search_blocked = False
            nodes = (
                resp
                and resp.get("data", {}).get("user")
                and resp["data"]["user"]["pullRequests"]["nodes"]
            )
            if resp and not nodes:
                # Empty history: either brand new or flagged by GitHub.
                # The search API refuses flagged accounts.
                chk = sh(
                    ["gh", "search", "prs", "--author", login, "--limit", "1"],
                    check=False,
                )
                search_blocked = "cannot be searched" in chk.stderr
                time.sleep(2.5)  # search API is 30 requests/minute
            entry = {
                "response": resp,
                "search_blocked": search_blocked,
            }
            with self.lock:
                self.dossier_cache[login] = entry
        return entry

    def collect_all(self, hits, label, source, require_unmerged, star_floor):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(
                pool.map(
                    lambda h: self.collect(h, label, source, require_unmerged, star_floor),
                    hits,
                )
            )

    def collect(self, hit, label, source, require_unmerged, star_floor):
        repo = hit["repository"]["nameWithOwner"]
        number = hit["number"]
        capped = source.startswith("label:") or source in ("october-invalid", "invalid-window")
        with self.lock:
            if (repo, number) in self.seen:
                return False
            if capped and self.repo_counts.get(repo, 0) >= REPO_CAP_LABELED:
                return False
            # Reserve, so a concurrent worker cannot double-collect.
            self.seen.add((repo, number))
            if capped:
                self.repo_counts[repo] = self.repo_counts.get(repo, 0) + 1
        author = (hit.get("author") or {}).get("login", "")
        if not author or author.endswith("[bot]") or (hit.get("author") or {}).get(
            "is_bot"
        ):
            return False
        owner = repo.split("/")[0]
        if author.lower() == owner.lower():
            return False  # PRs to your own repo are not triage targets
        if self.stars(repo) < star_floor:
            return False

        raw = gh_api(f"repos/{repo}/pulls/{number}", ok_statuses=("404",))
        if not raw:
            return False
        detail = json.loads(raw)
        merged = bool(detail.get("merged_at"))
        if require_unmerged and merged:
            return False
        if label == "ham" and not merged:
            return False

        diff = gh_api(
            f"repos/{repo}/pulls/{number}",
            accept="application/vnd.github.diff",
            ok_statuses=("404", "406", "422"),
        )
        files_raw = gh_api(f"repos/{repo}/pulls/{number}/files?per_page=100")
        commits_raw = gh_api(f"repos/{repo}/pulls/{number}/commits?per_page=50")
        if diff is None or files_raw is None or commits_raw is None:
            return False
        dossier = self.dossier(author)

        record = {
            "label": label,
            "source": source,
            "repo": repo,
            "number": number,
            "title": detail.get("title") or "",
            "body": (detail.get("body") or "")[:BODY_CAP],
            "author": author,
            "author_association": detail.get("author_association") or "",
            "head_ref": (detail.get("head") or {}).get("ref", ""),
            "additions": detail.get("additions", 0),
            "deletions": detail.get("deletions", 0),
            "changed_files": detail.get("changed_files", 0),
            "state": detail.get("state", ""),
            "merged": merged,
            "created_at": detail.get("created_at", ""),
            "pr_labels": [l["name"] for l in detail.get("labels") or []],
            "commits": [
                {
                    "email": (c["commit"].get("author") or {}).get("email", ""),
                    "message": c["commit"].get("message", "")[:1000],
                }
                for c in json.loads(commits_raw)
            ],
            "files": [f["filename"] for f in json.loads(files_raw)],
            "diff": diff[:DIFF_CAP],
            "repo_stars": self.stars(repo),
            "dossier": dossier["response"],
            "search_blocked": dossier["search_blocked"],
        }
        with self.lock:
            self.files[label].write(json.dumps(record) + "\n")
            self.files[label].flush()
            self.counts[source] = self.counts.get(source, 0) + 1
        return True


def main():
    m = Miner()

    print("== slop: label-mined", file=sys.stderr)
    for lbl in ("spam", "invalid", "AI slop", "hacktoberfest-invalid", "ai-generated", "low quality"):
        hits = search_prs(["--label", lbl, "--state", "closed"], 100)
        time.sleep(2.5)
        m.collect_all(hits, "slop", f"label:{lbl}", True, STAR_FLOOR_LABELED)

    print("== slop: express flood", file=sys.stderr)
    for q in ("Update Readme", "Update README"):
        hits = search_prs([q, "--repo", "expressjs/express", "--state", "closed"], 40)
        time.sleep(2.5)
        m.collect_all(hits, "slop", "express-flood", True, 0)

    print("== slop: october invalid window", file=sys.stderr)
    hits = search_prs(
        ["--label", "invalid", "--state", "closed", "--created", "2025-10-01..2025-10-31"],
        80,
    )
    time.sleep(2.5)
    m.collect_all(hits, "slop", "october-invalid", True, STAR_FLOOR_LABELED)

    print("== slop: invalid windows", file=sys.stderr)
    for window in ("2025-11-01..2025-11-30", "2024-10-01..2024-10-31"):
        hits = search_prs(
            ["--label", "invalid", "--state", "closed", "--created", window], 80
        )
        time.sleep(2.5)
        m.collect_all(hits, "slop", "invalid-window", True, STAR_FLOOR_LABELED)

    print("== slop (weak): agent-marked, closed unmerged", file=sys.stderr)
    for q in (
        "Generated with Claude Code",
        "Generated with openclaw",
        "Co-Authored-By: Claude",
        "This PR was written by Devin",
        "Co-authored-by: Cursor Agent",
    ):
        hits = search_prs([q, "--state", "closed"], 60)
        time.sleep(2.5)
        m.collect_all(hits, "slop", "agent-closed", True, STAR_FLOOR_AGENT)

    print("== ham: merged agent-marked", file=sys.stderr)
    for q in ("Generated with Claude Code", "Co-Authored-By: Claude"):
        hits = search_prs([q, "--merged"], 60)
        time.sleep(2.5)
        m.collect_all(hits, "ham", "agent-merged", False, STAR_FLOOR_AGENT)

    print("== ham: merged PRs from healthy repos", file=sys.stderr)
    for repo in (
        "fastapi/fastapi",
        "vitejs/vite",
        "axios/axios",
        "pallets/flask",
        "tokio-rs/tokio",
        "psf/requests",
        "mui/material-ui",
        "prettier/prettier",
    ):
        hits = search_prs(["--repo", repo, "--merged"], 10)
        time.sleep(2.5)
        m.collect_all(hits, "ham", "healthy-merged", False, 0)

    print("== ham: merged PRs from the slop repos", file=sys.stderr)
    slop_repos = sorted(
        {r for (r, _) in m.seen}
    )
    for repo in slop_repos:
        if m.stars(repo) < STAR_FLOOR_LABELED and repo != "expressjs/express":
            continue
        hits = search_prs(["--repo", repo, "--merged"], 6)
        time.sleep(2.5)
        m.collect_all(hits, "ham", "same-repo-merged", False, 0)

    print(json.dumps(m.counts, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
