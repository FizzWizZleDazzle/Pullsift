#!/usr/bin/env bash
# Download the corpus for a benchmark revision. The data files are release
# assets, not git objects, so a clone stays small.
#
#   bench/fetch_corpus.sh            # latest published corpus
#   bench/fetch_corpus.sh corpus-3   # a specific revision
set -euo pipefail
cd "$(dirname "$0")"

REPO=FizzWizZleDazzle/Pullsift
TAG="${1:-}"

if [ -z "$TAG" ]; then
  TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | sed -n 's/.*"tag_name": "\([^"]*\)".*/\1/p')
fi
[ -n "$TAG" ] || { echo "no release found; pass a tag"; exit 1; }

mkdir -p corpus/archive
base="https://github.com/$REPO/releases/download/$TAG"
for f in inputs labels; do
  echo "fetching $f.jsonl from $TAG"
  curl -fsSL "$base/$f.jsonl.gz" | gunzip > "corpus/$f.jsonl"
done

if [ "${WITH_ARCHIVE:-0}" = "1" ]; then
  # Raw mined records with outcome fields; needed only to re-tune or audit,
  # never to evaluate a bot.
  for f in slop ham detector; do
    curl -fsSL "$base/archive-$f.jsonl.gz" 2>/dev/null | gunzip > "corpus/archive/$f.jsonl" \
      || echo "  ($f absent for this revision)"
  done
fi

echo "corpus $TAG ready in $(pwd)/corpus"
wc -l corpus/*.jsonl
