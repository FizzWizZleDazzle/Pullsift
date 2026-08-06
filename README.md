# pullsift

A GitHub App that decides whether a pull request is worth a maintainer's
first look, and keeps the ones that are not out of the review queue.

## Why

Maintainers lose hours to PRs nobody should have to read: agent-generated
submissions with no human behind them, tutorial floods where hundreds of
new accounts send near-identical changes, drive-by accounts with a trail of
closed-as-invalid PRs elsewhere, and PRs against repos that accept none.
Existing tools score one PR or one account per repo, in isolation.
pullsift is built around sharing: repos in the network exchange campaign
signatures and corroborated author verdicts, so the first repo hit by a wave
inoculates the rest.

## Quick start

Run the service with a Postgres database and a GitHub App registration:

```
DATABASE_URL=postgres://... \
WEBHOOK_SECRET=... \
CANARY_SALT=... \
GITHUB_APP_ID=... \
GITHUB_PRIVATE_KEY_PATH=app.pem \
GITHUB_INSTALLATION_ID=... \
cargo run --release
```

Point the App's webhook at `/webhook` with `pull_request` and
`issue_comment` events. Per-repo settings live in
`.github/pullsift.yml`; every install starts in dry-run, which annotates
what it would have done without acting.

```yaml
dry_run: false
challenge: true
contribution_channel: "the mailing list"
exempt_users: [trusted-bot]
ai_policy: neutral
```

`ai_policy` sets the repo's stance on AI-assisted contributions, because
that is taste, not fact: `welcome` (AI involvement carries no weight;
only nobody-answers-for-this signals count), `neutral` (the default:
markers count for what the data says), `disclose` (AI is fine when
disclosed; undisclosed likely-AI prose is penalized), or `forbid` (any
provenance marker escalates).

## What it does

- Scores every PR with an explainable rule engine: each signal contributes
  a weighted value, and every verdict carries the full evidence table.
- Clusters incoming PRs by diff and prose similarity and detects arrival
  bursts, so a tutorial flood is caught as one campaign even when every
  author is a brand-new account.
- Builds an author dossier from public history: abandonment after review,
  agent provenance markers in commits, reply latency, and accounts GitHub
  itself has flagged.
- Enforces in tiers: label, hold as draft, or close with the evidence and a
  one-reply appeal path. Repos that accept no PRs close everything with a
  policy message instead of a judgment.
- Resolves uncertain cases with a challenge: one human sentence lifts the
  hold; silence or a honeypot-canary reply closes the PR.
- Learns from maintainer feedback: merges and reopens of flagged PRs become
  corrections, and a nightly refit promotes new weights only when held-out
  accuracy does not regress. Every weight version is kept; rollback is one
  row.
- Ships its training corpus as a public benchmark ([bench/](bench/README.md)):
  real labeled PRs, a fixed test split, and a scorer any triage bot can
  report against.

## Limitations

- Federation between installations is designed and tested but not yet
  switched on; each install currently learns alone.
- The dossier reads only public GitHub data, and GitHub hides the history
  of accounts its own spam systems flag; pullsift treats that flag
  itself as a signal.
- Tier thresholds are calibrated on a mined corpus of real PRs; treat
  close-tier automation as opt-in until your repo has feedback history.
- The App is not yet listed on the GitHub Marketplace; running it today
  means registering your own GitHub App and pointing it at a deployment.

Deeper design: [docs/design.md](docs/design.md).
