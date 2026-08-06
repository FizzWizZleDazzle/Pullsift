#!/usr/bin/env bash
# One-way mirror of this repo into the offload pod, then run cargo there.
# The pod copy is a build mirror, never the source of truth; edit locally.
#   scripts/remote.sh                 -> cargo test
#   scripts/remote.sh clippy -- -D warnings
set -euo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/kubeconfig}"

POD_DIR=/config/work/slopenatocatcher8080
pod=$(kubectl -n offload get pod -l app=desktop -o jsonpath='{.items[0].metadata.name}')

cd "$(git rev-parse --show-toplevel)"

# Mirror tracked and untracked-but-not-ignored files. Remove source dirs first
# so deleted files do not linger; keep target/ for incremental builds.
git ls-files -co --exclude-standard -z | tar --null -czf - -T - |
  kubectl -n offload exec -i "$pod" -- bash -c \
    "mkdir -p $POD_DIR && cd $POD_DIR && rm -rf src tests weights fixtures migrations scripts docs bench && tar -xzf -"

if [ $# -eq 0 ]; then set -- test; fi
rc=0
kubectl -n offload exec -i "$pod" -- bash -c \
  "cd $POD_DIR && cargo $(printf '%q ' "$@")" || rc=$?

# Bring the lockfile home so it is committed with the source.
kubectl -n offload cp "$pod:${POD_DIR#/}/Cargo.lock" ./Cargo.lock >/dev/null 2>&1 || true
exit $rc
