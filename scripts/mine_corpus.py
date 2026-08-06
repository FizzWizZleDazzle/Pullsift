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

OUT_DIR = Path(__file__).resolve().parent.parent / "bench" / "corpus" / "archive"

# Repos whose merges or closures carry no quality signal: merge-everything
# teaching repos and add-your-name lists. Matched as substrings.
REPO_BLOCKLIST = (
    "first-contribution",
    "first-pr",
    "your-first",
    "add-your-name",
    "contribute-to-this",
    "hello-world",
)
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
                    src = r.get("source", "")
                    if src.startswith(("label:", "window:")) or src in ("october-invalid", "invalid-window"):
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
        capped = source.startswith(("label:", "window:")) or source in (
            "october-invalid",
            "invalid-window",
        )
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
        if not author:
            return False
        # Bot-authored PRs stay in: agent traffic arrives via bot accounts
        # and a triage bot must score them. They are only barred from the
        # label sweeps, where closed bot PRs are superseded-dependency
        # noise rather than rejected slop.
        is_bot = author.endswith("[bot]") or (hit.get("author") or {}).get("is_bot")
        if is_bot and capped:
            return False
        owner = repo.split("/")[0]
        if author.lower() == owner.lower():
            return False  # PRs to your own repo are not triage targets
        if any(b in repo.lower() for b in REPO_BLOCKLIST):
            return False  # merge-everything repos teach nothing
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
            # Numeric account id: stable across renames, unlike the login.
            "author_id": (detail.get("user") or {}).get("id"),
            "author_association": detail.get("author_association") or "",
            "head_ref": (detail.get("head") or {}).get("ref", ""),
            # Merges into non-default branches are weaker ham evidence.
            "base_ref": (detail.get("base") or {}).get("ref", ""),
            "default_branch": ((detail.get("base") or {}).get("repo") or {}).get(
                "default_branch", ""
            ),
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

    print("== slop: windowed label sweeps", file=sys.stderr)
    windows = {
        "invalid": (
            "2025-12-01..2026-01-31",
            "2026-02-01..2026-03-31",
            "2026-04-01..2026-05-31",
            "2026-06-01..2026-07-31",
        ),
        "spam": (
            "2025-01-01..2025-06-30",
            "2025-07-01..2025-12-31",
            "2026-01-01..2026-07-31",
        ),
        "hacktoberfest-invalid": (
            "2019-10-01..2019-11-30",
            "2020-10-01..2020-11-30",
            "2021-10-01..2021-11-30",
        ),
    }
    for lbl, spans in windows.items():
        for span in spans:
            hits = search_prs(
                ["--label", lbl, "--state", "closed", "--created", span], 100
            )
            time.sleep(2.5)
            m.collect_all(hits, "slop", f"window:{lbl}", True, STAR_FLOOR_LABELED)

    print("== slop (weak): agent-marked, closed unmerged", file=sys.stderr)
    for q in (
        "Generated with Claude Code",
        "Generated with openclaw",
        "Co-Authored-By: Claude",
        "This PR was written by Devin",
        "Co-authored-by: Cursor Agent",
        "Generated with ChatGPT",
        "Co-authored-by: Copilot",
    ):
        hits = search_prs([q, "--state", "closed"], 60)
        time.sleep(2.5)
        m.collect_all(hits, "slop", "agent-closed", True, STAR_FLOOR_AGENT)

    print("== ham: merged agent-marked", file=sys.stderr)
    # The hard ham: an agent wrote it, a human reviewed it, a maintainer
    # merged it. A bot that convicts on provenance alone fails these.
    # Merged agent PRs are mostly owner-authored (people run agents on
    # their own repos), and owner PRs are excluded as non-triage-targets,
    # so this source needs volume and a low star floor to find the
    # member- and outsider-authored remainder.
    for q in (
        '"Generated with Claude Code"',
        '"Generated with Claude"',
        '"Co-Authored-By: Claude"',
        '"Co-authored-by: Copilot"',
        '"Co-authored-by: Cursor Agent"',
        '"Generated with ChatGPT"',
        '"\U0001f916 Generated with"',
    ):
        hits = search_prs([q, "--merged"], 100)
        time.sleep(2.5)
        m.collect_all(hits, "ham", "agent-merged", False, 5)

    print("== ham: AI-topic without AI authorship", file=sys.stderr)
    # Merged PRs about AI tooling (adding a Claude API client, an OpenAI
    # integration). They mention agents everywhere and carry no provenance
    # markers; a bot keying on the word 'claude' fails these.
    for q in (
        "add claude api",
        "anthropic sdk",
        "add openai integration",
        "claude support",
    ):
        hits = search_prs([q, "--merged"], 40)
        time.sleep(2.5)
        m.collect_all(hits, "ham", "ai-topic-merged", False, STAR_FLOOR_AGENT)

    print("== slop: marker-stripped, accused by maintainers", file=sys.stderr)
    # No provenance markers; the tell is the maintainer's reaction. The
    # adversarial frontier: separates content-reading bots from
    # metadata-reading ones. Audited before entering the scored corpus.
    for q in (
        '"looks AI-generated" in:comments',
        '"please stop submitting AI" in:comments',
        '"did you even test this" in:comments',
        '"this code does not compile" in:comments',
        '"hallucinated" in:comments',
        '"written by ChatGPT" in:comments',
        '"our AI policy" in:comments',
        '"do not accept AI-generated" in:comments',
        '"this function does not exist" in:comments',
        '"clearly generated" in:comments',
    ):
        hits = search_prs([q, "--state", "closed"], 40)
        time.sleep(2.5)
        m.collect_all(hits, "slop", "accused", True, STAR_FLOOR_LABELED)

    print("== slop: campaign artifacts with varied diffs", file=sys.stderr)
    hits = search_prs(
        [
            '"add CONTRIBUTING.md" in:title',
            "--state",
            "closed",
            "--created",
            "2024-01-01..2025-12-31",
        ],
        40,
    )
    time.sleep(2.5)
    m.collect_all(hits, "slop", "wave-artifact", True, STAR_FLOOR_LABELED)

    print("== twins: same query, both outcomes", file=sys.stderr)
    # Twin pairs force content-level discrimination: the same title shape
    # exists as merged ham and rejected slop.
    twins = (
        ('"fix typo" in:title', "typo"),
        ('"add unit tests" in:title', "tests"),
        ('"run prettier" in:title', "format"),
        ('"regenerate" in:title', "gendiff"),
        ('"security fix" in:title', "security"),
    )
    for q, tag in twins:
        hits = search_prs([q, "--merged"], 25)
        time.sleep(2.5)
        m.collect_all(hits, "ham", f"{tag}-merged", False, STAR_FLOOR_LABELED)
        hits = search_prs([q, "--state", "closed", "--label", "invalid"], 20)
        time.sleep(2.5)
        m.collect_all(hits, "slop", f"{tag}-rejected", True, STAR_FLOOR_LABELED)

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
        "sveltejs/svelte",
        "home-assistant/core",
        "denoland/deno",
        "astral-sh/ruff",
        "helix-editor/helix",
        "zed-industries/zed",
    ):
        hits = search_prs(["--repo", repo, "--merged"], 12)
        time.sleep(2.5)
        m.collect_all(hits, "ham", "healthy-merged", False, 0)

    print("== slop: issue-first plant-and-fix", file=sys.stderr)
    # File a bug, then "fix" it: satisfies linked-issue heuristics. Audited
    # before entering the scored corpus (the issue may be real).
    hits = search_prs(['"Fixes #" in:body', "--state", "closed", "--label", "invalid"], 60)
    time.sleep(2.5)
    m.collect_all(hits, "slop", "issue-first", True, STAR_FLOOR_LABELED)

    print("== governance-file gaming, both outcomes", file=sys.stderr)
    hits = search_prs(["CODEOWNERS in:title", "--state", "closed"], 30)
    time.sleep(2.5)
    m.collect_all(hits, "slop", "governance", True, STAR_FLOOR_LABELED)
    hits = search_prs(["CODEOWNERS in:title", "--merged"], 20)
    time.sleep(2.5)
    m.collect_all(hits, "ham", "governance-merged", False, STAR_FLOOR_LABELED)

    print("== slop: fabricated-vulnerability theater", file=sys.stderr)
    for q in ('"CVE-" in:title', '"fix vulnerability" in:title', '"prototype pollution" in:title'):
        hits = search_prs([q, "--state", "closed"], 30)
        time.sleep(2.5)
        m.collect_all(hits, "slop", "vuln-theater", True, STAR_FLOOR_LABELED)
    hits = search_prs(['"CVE-" in:title', "--merged"], 25)
    time.sleep(2.5)
    m.collect_all(hits, "ham", "cve-merged", False, STAR_FLOOR_LABELED)

    print("== bounty-incentive PRs, both outcomes", file=sys.stderr)
    hits = search_prs(['"/claim #" in:comments', "--state", "closed"], 40)
    time.sleep(2.5)
    m.collect_all(hits, "slop", "bounty-closed", True, STAR_FLOOR_LABELED)
    hits = search_prs(['"/claim #" in:comments', "--merged"], 20)
    time.sleep(2.5)
    m.collect_all(hits, "ham", "bounty-merged", False, STAR_FLOOR_LABELED)

    print("== ham: shapes that look like waves or churn", file=sys.stderr)
    # Backport trains, GitOps value bumps, image compression: legitimate
    # PR ecologies whose surface shape (near-identical diffs, trivial
    # content, no prose) matches slop heuristics.
    for q, tag, n in (
        ('"[backport" in:title', "backport-merged", 20),
        ('"bump image tag" in:title', "gitops-merged", 20),
        ('"compress images" in:title', "opaque-merged", 15),
    ):
        hits = search_prs([q, "--merged"], n)
        time.sleep(2.5)
        m.collect_all(hits, "ham", tag, False, 0)
    hits = search_prs(["--repo", "NixOS/nixpkgs", "--merged", "update in:title"], 25)
    time.sleep(2.5)
    m.collect_all(hits, "ham", "gitops-merged", False, 0)
    hits = search_prs(['"hash mismatch" in:comments', "--state", "closed"], 15)
    time.sleep(2.5)
    m.collect_all(hits, "slop", "gitops-rejected", True, STAR_FLOOR_LABELED)

    print("== ham: reputation-laundering ladders", file=sys.stderr)
    # Slop authors whose dossier shows merges predating their first flagged
    # PR: mine those merges as ham under the same author, so the corpus
    # carries mixed-outcome authors and "prior merges" cannot exculpate.
    ladder = []
    p = OUT_DIR / "slop.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = ((r.get("dossier") or {}).get("data") or {}).get("user") or {}
            nodes = (user.get("pullRequests") or {}).get("nodes") or []
            merged_before = [
                n
                for n in nodes
                if n.get("merged") and n.get("createdAt", "9999") < r.get("created_at", "")
            ]
            if len(merged_before) >= 2:
                ladder.append(r["author"])
    for login in sorted(set(ladder))[:25]:
        hits = search_prs(["--author", login, "--merged"], 8)
        time.sleep(2.5)
        m.collect_all(hits, "ham", "laundering-ham", False, 0)

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
