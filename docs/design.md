# Pullsift design

The scoring core, the three detection lanes, the challenge, the learner,
and the federation formats. Code references are the authority; this doc
explains why the pieces have the shape they do.

## Scoring core (`engine.rs`, `fit.rs`)

Every signal is a rule emitting a value in [0,1]. Each rule has a weight
interpreted as a log-likelihood ratio; a PR's score is
`bias + sum(w_i * x_i)` and its probability is the sigmoid of that. This is
the additive spam-filter design: explainable per verdict, and refittable
without touching detection code. Rules unknown to the weight table score
zero but are logged, so new rules ship dark and get priced later.

Tier thresholds are probabilities chosen on held-out data at fixed
false-positive targets: close at 0.1 percent, hold at 1 percent, label at 5
percent. The threshold search only ever raises a tier to enforce ordering,
because raising a threshold cannot raise its FPR.

Fitting is plain logistic regression by gradient descent (`fit.rs`), with
sample weights; maintainer corrections count five times. Negatives must be
merged PRs from the same repos as the positives, otherwise the model learns
"is a newcomer" instead of "is slop".

## Lane C: repo policy and PR shape (`policy.rs`, `config.rs`)

Runs first, costs nothing. The `mirror-no-prs` archetype closes every PR
with a policy message naming the real contribution channel; the message is
explicitly not a judgment of the change. Protected-paths and docs-only
checks inject rules instead of closing. Per-repo config comes from
`.github/pullsift.yml`, defaults conservative, dry-run on.

`ai_policy` encodes the repo's stance on AI assistance, which is
maintainer taste and therefore config rather than a fitted weight:
`welcome` zeroes the AI-provenance and AI-style rules (the dossier,
cluster, and grounding lanes still score), `neutral` leaves the fitted
table alone, `disclose` adds an UNDISCLOSED_AI rule when the detector
fires with no provenance markers, and `forbid` adds AI_FORBIDDEN when
any marker is present. The policy rules carry priors, not fitted
weights, and policy overrides intentionally trade the calibrated FPR
targets for the maintainer's explicit choice.

`shape_rules` adds tells mined from real drive-by and agent PRs:

- BRANCH_DRIVE_BY: the head branch is `patch-N` (GitHub's web editor) or
  `main`/`master` in a fork.
- TITLE_UPDATE_FILE and SINGLE_WEB_COMMIT: the web editor's default
  "Update README.md" title, alone or with exactly one commit.
- TEMPLATE_UNFILLED: most of the body is still inside the template's HTML
  comments.
- BODY_DIFF_MISMATCH: the body claims tests, the diff touches nothing
  test-shaped.
- BODY_TO_DIFF_RATIO: an essay body over a trivial diff.
- WHITESPACE_ONLY (in `diffsig.rs`): added lines carry no alphanumeric
  content.
- BEGGING: please-merge/kindly/assign-me phrasing.
- USERNAME_PATTERN: digit-heavy generated-looking logins.
- AGENT_BRANCH: agent-workflow branch prefixes (`copilot/`, `codex/`,
  `cursor-`, and peers).
- BODY_SCAFFOLD: the agent PR template (Summary/Changes/Testing
  headings, checkbox lists) in force.
- COMMENT_HEAVY (in `diffsig.rs`): generated code over-comments; the
  comment share of added lines, when there are enough of them.
- GROUNDING_MISS (in `pipeline.rs`): the body name-drops identifiers or
  paths, at least two, and none exists in the diff or changed files.
  Fabricated symbols are the highest-precision content tell reported by
  maintainers fighting AI slop.

New rules ship dark (zero weight) and get priced by the tuner before they
influence verdicts.

An optional self-hosted AI-text detector (`scripts/detector_server.py`,
DeBERTa-based, MIT-licensed weights) feeds a single DETECTOR_SCORE rule
via `DETECTOR_URL`. It scores only the prose of title and body, with
code, links, and template scaffolding stripped, and abstains under fifty
words: below that, and on non-native-English writing, detector scores
are noise, so the rule carries one fitted weight and can never reach an
enforcement tier alone. `scripts/detector_batch.py` writes the same
scores as a corpus sidecar so the tuner prices the rule on data.

## Lane A: campaign signatures (`diffsig.rs`, `textsig.rs`, `cluster.rs`)

Two similarity channels feed one union-find clustering:

- Diff: added lines normalized (whitespace, case, hunk headers), 4-token
  shingles, weighted 64-bit simhash. Candidate lookup is multi-index
  hashing over four 16-bit bands: recall is guaranteed within Hamming
  distance 3 by pigeonhole and probabilistic up to the verification cutoff
  of 6. Simhash tolerance shrinks with diff size, which is fine: real flood
  diffs are near-identical, and short spam clusters through the text
  channel.
- Text: title plus body, 3-token shingles, 128-permutation minhash, 16x8
  banded LSH, verified by estimated Jaccard at 0.7. Unfilled PR templates
  make flood bodies near-identical, which is exactly what this channel
  catches.

Burst is Poisson surprise: with the repo's baseline arrival rate as lambda,
`s = -ln P(X >= k)` for k cluster arrivals inside a six-hour window,
squashed to [0,1]. A cluster also carries the stylometry centroid of its
members; high cohesion binds it tighter. Cluster membership re-enters the
engine as rules (`CLUSTER_SIZE_LOG`, `CLUSTER_BURST`,
`CLUSTER_STYLE_COHESION`, `CLUSTER_XREPO`), so there is one scoring path.
Size, burst, and cohesion require at least two distinct authors in the
cluster: one person's batch of similar PRs (a docs sweep) is normal work,
and mined merged batches were the engine's main false-positive source.
`CLUSTER_XREPO` still fires for a single account spraying repos.

## Lane B: author dossier (`dossier.rs`, `stylometry.rs`)

The target property is "no human answers for this PR", measured:

- Abandonment ratio: of past PRs that received review, the fraction the
  author never followed up on. One GraphQL query builds the whole dossier;
  cached seven days.
- Provenance markers: agent commit emails, `Co-Authored-By` agent
  identities, generation footers.
- Latency: fork-to-PR minutes for a nontrivial diff, sub-minute median
  reply latency, commit-hour entropy flat around the clock.
- Account shape, at small weights: age, bio, followers, spread of unrelated
  repos.
- `ACCOUNT_UNSEARCHABLE`: the search API refuses the account and its PR
  connection comes back empty while its PRs are visible on the web. This
  correlates with accounts GitHub has flagged, but corpus data showed it
  also hits legitimate privacy-opaque contributors, so it carries a weak
  weight and is priced by the fit like everything else.

Ratios stay silent below minimum sample sizes; a pattern needs data.
Stylometry (em dashes, unicode punctuation, emoji, non-ASCII, an AI-phrase
lexicon, markdown structure density) fires as per-PR rules at full weight
and doubles as the cluster cohesion feature, where it compares PRs to each
other rather than to a norm.

## Challenge (`challenge.rs`)

Verdicts landing in the hold band get one interaction round instead of a
guess: the PR is held as draft and the bot asks for one sentence in the
author's own words. The comment carries an invisible markdown-comment
honeypot instructing automated readers to include a canary token. A human
reply lifts the hold; the canary or 72 hours of silence closes the PR, with
reopening one reply away.

## Offline tuning (`src/bin/tune.rs`, `scripts/mine_corpus.py`)

The miner builds a labeled corpus from the live API: label-mined spam and
invalid PRs (curated by star floor, human non-owner authors, capped per
repo so one spam farm cannot dominate), the express README flood, agent
marked PRs both merged and closed-unmerged, and merged PRs from the same
repos as ham. The tuner replays the corpus through the production
pipeline in real arrival order, cross-validates with author-grouped folds
(an author never appears on both sides), fits, and writes the weight
table with provenance metadata. Rules that never fired in the corpus keep
their prior weight: no data means the prior stands. Exact metrics live in
the tune output and the weights file, not here.

The corpus doubles as a public benchmark (`bench/`): a fixed
author-grouped test split, a prediction format, and a dependency-free
scorer, so other triage bots can report comparable numbers. `tune --emit`
writes Pullsift's own out-of-fold predictions in that format. The
benchmark contract lives in `bench/README.md`.

## Learner (`learn.rs`, `store.rs`)

Maintainer actions on scored PRs become labels; overrides of Pullsift
actions are corrections at five-fold weight. A nightly batch job refits and
promotes only behind guardrails: minimum corpus size, both classes present
in both splits, and held-out AUC within 0.005 of the incumbent. Promotion
inserts a new row in `weights_versions` and flips `active`; rollback flips
it back. Weights never mutate live.

## Federation (`federation.rs`, dormant)

Two record kinds with different physics:

- Campaign signatures: simhash, text band keys, path-set hash, style
  centroid, cluster size, TTL. No usernames, no diff content. Propagate on
  first cluster formation.
- Author verdicts: salted hash of the login, evidence hash, strength,
  installation id and owner, expiry. They bind only after corroboration by
  three independent installation owners; stacked verdicts from one owner
  count once. Automatic, but braked.

Envelopes are ed25519-signed per installation. The MVP exchanges records
through a local directory stub so the protocol is fully exercised by
tests; a relay replaces the transport, not the format. Before the relay
goes public the service needs data-subject tooling (access, correction,
expiry are already in the format) for the author-verdict lane.

## Reference cases (`tests/replay.rs`)

Three acceptance tests on captured fixtures gate any live install:

- express README flood: real spam PRs cluster via the unfilled-template
  text channel and escalate to hold or close; real merged PRs from the same
  repo never exceed label.
- linguist#8074: the author arrives at hold-or-above from the dossier
  alone, with no repo-local history.
- torvalds/linux: every PR closes by policy, unscored.
