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

Score your own bot: read `bench/corpus/inputs.jsonl`, write one
prediction per PR, run `bench/score.py` on the result. Everything the
benchmark needs lives under `bench/`; nothing else in this repository is
required to evaluate a bot.

## Corpus format

Three files under `bench/corpus/`:

- `inputs.jsonl`: one JSON object per PR; the only file a bot reads.
- `labels.jsonl`: id, label, source, author; read by the scorer.
- `archive/`: raw mined records with outcome fields intact, kept for
  curation and audit. Not an input; a bot reading it is cheating by
  construction.

Outcome fields (`label`, `source`, `state`, `merged`, `pr_labels`) exist
only outside `inputs.jsonl`, so an honest bot cannot leak them by
accident. The author dossier inside `inputs.jsonl` is filtered to
history that existed when the PR was opened, dossier entries in the
scored PR's own repo carry no outcome fields, and repo stars are
coarsened to an order of magnitude.

Fields in `inputs.jsonl`:

| field | meaning |
|---|---|
| `id` | `repo#number`; the join key for predictions and labels |
| `repo`, `number` | PR identity |
| `title`, `body` | as posted (body capped) |
| `author`, `author_id` | login and (newer records) stable numeric id |
| `first_pr_to_repo` | no visible prior PR to this repo when it opened |
| `prior_prs_visible` | author PRs predating this one, across repos |
| `prior_prs_this_repo`, `prior_merged_this_repo` | same, this repo only |
| `head_ref`, `base_ref`, `default_branch` | branches (newer records) |
| `additions`, `deletions`, `changed_files` | diff stats |
| `created_at` | when the PR was opened |
| `commits` | author email and message per commit |
| `files` | changed paths |
| `diff` | unified diff (capped) |
| `repo_stars_magnitude` | order of magnitude of the repo's stars |
| `dossier` | author profile, filtered to pre-PR history |
| `search_blocked` | the search API refused this author |

In `labels.jsonl`, `label` is `slop` or `ham` (ham is merged; slop is
closed unmerged) and `source` is how the PR was mined.

Sources: `label:*` (maintainer-labeled spam or invalid, star-floored,
capped per repo), `window:*` (the same labels swept over time windows),
`express-flood` (the README wave), `october-invalid` and `invalid-window`
(invalid-labeled time windows), `agent-closed` (agent markers, closed
unmerged; a weak positive), `agent-merged`, `same-repo-merged`,
`healthy-merged`, and `ai-topic-merged` (ham).

Bot-authored PRs are in scope: agent traffic arrives through bot
accounts, and a triage bot that blanket-exempts `[bot]` authors passes
exactly the PRs this benchmark is about. Sanctioned automation
(dependabot-style merges) appears as ham; scoring it 0 is the correct
call when your bot's exemption policy covers it, and costs you when the
bot account was an unsupervised agent.

Two ham sources exist specifically to punish shortcut bots. `agent-merged`
is agent-written work a human reviewed and a maintainer merged: provenance
markers everywhere, and still ham, because a human answers for it.
`ai-topic-merged` is merged work about AI tooling (adding a Claude API
client, an OpenAI integration): agent vocabulary everywhere, no provenance
at all. A bot that convicts on "an AI wrote this" fails the first set; a
bot that keys on AI words fails the second.

`aipolicy-closed` and `aipolicy-merged` come from projects that published
what they will not accept from a generator, found by searching
contributing guides for the policy text. Both sides are drawn from the
same repos and both must carry real code, so the slice measures the pull
request rather than the venue. It exists because the label sweeps select
for drive-by edits: across the rest of the corpus the median record adds
one line of production source, which leaves nothing for a code rule to
read.

Labels are maintainer decisions, not ground truth annotations: a closed
unmerged PR with a spam label is slop because the maintainer said so.

## Split

The test split is by author, not by PR: an author whose lowercased login
FNV-1a-64 hash is 4 mod 5 belongs to the test split, roughly a fifth of
the corpus. Train on the rest. Fit nothing on test authors: the split is
enforceable only by convention, but the rule is deterministic and public,
so a violation is detectable from a described method.

The scorer also reports a repo-disjoint slice: test records whose repo
never appears outside the test split. A large gap between the headline
number and the repo-disjoint number means the bot leans on repo
identity, which is worthless in production where an installation sees
one repo. Logins are mutable on GitHub, so the author split can be
contaminated by renames; newer records carry `author_id` for detecting
this, and the split key stays the login until a corpus revision moves it.

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

The headline number is recall on AI-generated slop at 1 percent
false-positive rate on the test split. Human farming (hacktoberfest
badges, add-your-name churn) is a solved detection problem; AI slop is
not, so the benchmark weights it accordingly. Every slop record carries
an audit-assigned `kind` (`ai`, `human`, or `unclear`) in
`labels.jsonl`, and the scorer reports recall per kind. The FPR budget
is still paid on all ham: catching AI slop by convicting everything is
priced the same as any other false-positive spend.

The scorer also reports overall recall at 5, 1, and 0.1 percent FPR,
AUC, and per-source recall, because a bot tuned only for one wave shows
up as a per-source hole. Thresholds are chosen on the evaluated set's
own ham scores, so every bot pays the same FPR budget.

Reference points to beat, regenerated with the corpus (see
`results.md`): the shipped baselines (account age, first-timer flag) and
Pullsift's own out-of-fold predictions, produced by
`cargo run --release --bin tune -- --dry --emit predictions.jsonl`.

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

Predictions must come from your bot reading `inputs.jsonl` alone.
Outcome fields live only in `labels.jsonl` and `archive/`; reading
either at prediction time is the same violation as training on test.
The headline number carries a bootstrap confidence interval because at
1 percent FPR the ham budget is small and a single borderline record
moves the point estimate; treat overlapping intervals as a tie.

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
contain. Diffs and bodies are capped. The dossier's per-PR history is
filtered to entries that predate the scored PR, but account-level
aggregates (followers, restricted contribution counts) are mining-time
snapshots and leak a little future.
`author_association` is not in the released inputs at all. GitHub computes
it at read time, so merging promotes the author from NONE to CONTRIBUTOR
and the mined value records the outcome rather than the arrival state: it
was NONE for 75 percent of slop and 3 percent of ham. It is replaced by
`first_pr_to_repo` and the prior-PR counts, reconstructed from history
predating each PR. Removing it dropped the first-timer baseline from 0.918
AUC to 0.620. Results within these bounds compare bots fairly; absolute
recall numbers do not transfer to production unchanged.
