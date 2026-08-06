# slopbench

A benchmark for PR triage bots: real GitHub pull requests, labeled slop or
ham, with everything a bot could have seen at arrival time. Run your bot
over the corpus, emit one score per PR, and the scorer reports how well you
separate slop from legitimate work at fixed false-positive budgets.

## Why

Every triage bot publishes its own anecdotes. Comparing them needs a shared
corpus, a shared split, and a shared metric. This benchmark supplies all
three, built from mined production incidents: label-mined spam waves, the
express.js README flood, hacktoberfest spam, and agent-generated PRs both
rejected and merged. The merged agent PRs matter most: a bot that convicts
on "an AI wrote this" alone fails them, and should.

## Quick start

Score the shipped baselines:

```
python3 bench/baselines/first_timer.py > /tmp/ft.jsonl
python3 bench/score.py /tmp/ft.jsonl
```

Score your own bot: read `bench/corpus/{slop,ham}.jsonl`, write one
prediction per PR, run `bench/score.py` on the result. Everything the
benchmark needs lives under `bench/`; nothing else in this repository is
required to evaluate a bot.

## Corpus format

`bench/corpus/slop.jsonl` and `ham.jsonl`, one JSON object per PR:

| field | meaning |
|---|---|
| `label` | `slop` or `ham` (ham is merged; slop is closed unmerged) |
| `source` | how the PR was mined (see below) |
| `repo`, `number` | PR identity; `repo#number` is the join key |
| `title`, `body` | as posted (body capped) |
| `author`, `author_association` | login and GitHub's association enum |
| `head_ref` | source branch name |
| `additions`, `deletions`, `changed_files` | diff stats |
| `state`, `merged`, `created_at`, `pr_labels` | PR status |
| `commits` | author email and message per commit |
| `files` | changed paths |
| `diff` | unified diff (capped) |
| `repo_stars` | target repo stars at mining time |
| `dossier` | the author's profile GraphQL response (last 50 PRs) |
| `search_blocked` | the search API refused this author |

Sources: `label:*` (maintainer-labeled spam or invalid, star-floored,
capped per repo), `express-flood` (the README wave), `october-invalid` and
`invalid-window` (invalid-labeled time windows), `agent-closed` (agent
markers, closed unmerged; a weak positive), `agent-merged` and
`same-repo-merged` and `healthy-merged` (ham).

Labels are maintainer decisions, not ground truth annotations: a closed
unmerged PR with a spam label is slop because the maintainer said so.

## Split

The test split is by author, not by PR: an author whose lowercased login
FNV-1a-64 hash is 4 mod 5 belongs to the test split, roughly a fifth of
the corpus. Train on the rest. Fit nothing on test authors: the split is
enforceable only by convention, but the rule is deterministic and public,
so a violation is detectable from a described method.

## Predictions

JSONL, one object per PR, either shape:

```
{"id": "owner/repo#number", "score": 0.87}
{"repo": "owner/repo", "number": 123, "score": 0.87}
```

`score` is a probability in [0, 1]; higher means more likely slop. Every
test-split PR needs a prediction. A bot that exempts a PR (trusted bot
author, maintainer override) scores it 0; a bot that closes by policy
scores it 1. The scorer aborts on missing test predictions so abstaining
on hard cases cannot inflate results.

## Metrics

The headline number is recall at 1 percent false-positive rate on the
test split. The scorer also reports AUC, recall at 5 and 0.1 percent FPR,
and per-source recall, because a bot tuned only for one wave shows up as
a per-source hole. Thresholds are chosen on the evaluated set's own ham
scores, so every bot pays the same FPR budget.

Reference points to beat, regenerated with the corpus (see
`results.md`): the shipped baselines (account age, first-timer flag) and
slopcatcher's own out-of-fold predictions, produced by
`tune --dry --emit predictions.jsonl` via `scripts/remote.sh`.

## Reporting comparable results

The corpus grows over time, so a number without a corpus revision is not
comparable to anything. The scorer prints a corpus fingerprint (a hash of
the record ids and labels); report it next to your numbers. A comparison
is valid only between runs with the same fingerprint.

When you publish a result, include:

- the corpus fingerprint,
- the scorer's test-split output (or its `--json` form),
- one paragraph on what your bot reads (whole corpus fields, or a
  subset), and
- whether anything was fitted, and on what. Fitting on test-split authors
  invalidates the result; the split rule is public, so a described method
  can be checked.

Predictions must come from your bot reading the corpus records, not from
the labels; `label`, `state`, `merged`, and `pr_labels` are the outcome,
not the input. Reading them at prediction time is the same violation as
training on test.

## Data provenance

Every record is public GitHub data, collected via the public API, and
carries its origin (`repo`, `number`). Authors and maintainers who want a
record removed can request it; removal changes the fingerprint, which is
the point: results move to the new revision.

## Limitations

The corpus is mined from public search, so it over-represents slop waves
that got labeled and under-represents slop that was silently closed. Ham
is merged PRs, which excludes legitimate work that was rejected on its
merits. Clustering signals see only the waves the corpus happens to
contain. Diffs and bodies are capped. The dossier snapshot is
mining-time, not PR-time, so account features leak a little future.
`author_association` is also mining-time: merging promotes an author to
CONTRIBUTOR, so the field partially encodes the label, and a predictor
leaning on it scores better here than it would at arrival time. The
first-timer baseline exploits exactly this leak, which is why its AUC
flatters it. Results within these bounds compare bots fairly; absolute
recall numbers do not transfer to production unchanged.
