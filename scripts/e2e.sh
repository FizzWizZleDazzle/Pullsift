#!/usr/bin/env bash
# End-to-end test, run inside the build pod: real binary, real Postgres,
# fake GitHub API, real HMAC-signed webhooks.
#   scripts/e2e.sh postgres://user:pass@host/db
set -euo pipefail

PG_URL="${1:?usage: e2e.sh <postgres-url>}"
SECRET="e2e-webhook-secret"
API=127.0.0.1:9299
SVC=127.0.0.1:9288
DIR=$(mktemp -d)
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$DIR"' EXIT

cargo build --bins --quiet

openssl genrsa -out "$DIR/app.pem" 2048 2>/dev/null

FAKE_BIND=$API ./target/debug/fake_github &
DATABASE_URL="$PG_URL" \
WEBHOOK_SECRET="$SECRET" \
CANARY_SALT=e2e-salt \
GITHUB_APP_ID=1 \
GITHUB_PRIVATE_KEY_PATH="$DIR/app.pem" \
GITHUB_INSTALLATION_ID=1 \
GITHUB_API_BASE="http://$API" \
BIND_ADDR=$SVC \
./target/debug/pullsift &

for i in $(seq 1 40); do
  curl -sf "http://$SVC/healthz" >/dev/null 2>&1 && break
  sleep 0.5
  [ "$i" = 40 ] && { echo "service never became healthy"; exit 1; }
done
echo "service healthy"

send_pr() { # number author title body [action]
  local n=$1 author=$2 title=$3 body=$4 action=${5:-opened}
  cat > "$DIR/payload.json" <<EOF
{
  "action": "$action",
  "repository": { "full_name": "e2e/sandbox" },
  "pull_request": {
    "number": $n,
    "node_id": "PR_e2e_$n",
    "title": "$title",
    "body": "$body",
    "additions": 2,
    "changed_files": 1,
    "author_association": "FIRST_TIME_CONTRIBUTOR",
    "user": { "login": "$author" },
    "head": { "repo": { "fork": true } }
  }
}
EOF
  local sig
  sig=$(openssl dgst -sha256 -hmac "$SECRET" -hex < "$DIR/payload.json" | awk '{print $NF}')
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$SVC/webhook" \
    -H "content-type: application/json" \
    -H "x-github-event: pull_request" \
    -H "x-hub-signature-256: sha256=$sig" \
    --data-binary @"$DIR/payload.json")
  [ "$code" = 202 ] || { echo "webhook for PR $n returned $code"; exit 1; }
}

# A tampered signature must be rejected.
bad=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$SVC/webhook" \
  -H "content-type: application/json" \
  -H "x-github-event: pull_request" \
  -H "x-hub-signature-256: sha256=deadbeef" \
  --data '{"action":"opened"}')
[ "$bad" = 401 ] || { echo "tampered webhook not rejected: $bad"; exit 1; }
echo "PASS: tampered signature rejected"

# PR 901: agent slop. Body carries a generation footer; commits (from the
# fake) carry the agent email and trailer.
send_pr 901 ghostpipe "Improve documentation" \
  "This PR introduces a comprehensive enhancement. Generated with Claude Code."
# PR 101: ordinary contribution.
send_pr 101 casualdev "fix: strip quotes in parser" \
  "The parser kept surrounding quotes when reading string values. Adds unquote and a regression test."

for i in $(seq 1 40); do
  calls=$(curl -sf "http://$API/_calls")
  echo "$calls" | grep -q '"call":"close"' && break
  sleep 0.5
  [ "$i" = 40 ] && { echo "no close call recorded; calls: $calls"; exit 1; }
done

# A rescore of the passing PR must edit its score comment, not stack a
# second one.
send_pr 101 casualdev "fix: strip quotes in parser" \
  "The parser kept surrounding quotes when reading string values. Adds unquote and a regression test." \
  synchronize
for i in $(seq 1 40); do
  calls=$(curl -sf "http://$API/_calls")
  echo "$calls" | grep -q '"call":"update-comment"' && break
  sleep 0.5
  [ "$i" = 40 ] && { echo "no update-comment call recorded; calls: $calls"; exit 1; }
done

curl -sf "http://$API/_calls" > "$DIR/calls.json"
CALLS_FILE="$DIR/calls.json" python3 - <<'EOF'
import json, os
calls = json.load(open(os.environ["CALLS_FILE"]))
closes = [c for c in calls if c.get("call") == "close"]
comments = [c for c in calls if c.get("call") == "comment"]
labels = [c for c in calls if c.get("call") == "label"]
updates = [c for c in calls if c.get("call") == "update-comment"]
assert any(c["pr"] == 901 for c in closes), f"PR 901 must be closed: {calls}"
assert not any(c["pr"] == 101 for c in closes), f"PR 101 must not be closed: {closes}"
assert not any(c["pr"] == 101 for c in labels), f"PR 101 must not be labeled: {labels}"
evidence = [c for c in comments if c["pr"] == 901]
assert evidence and "DOSSIER_SPAM_LABELS" in json.dumps(evidence), "close comment must carry evidence"
scores = [c for c in comments if c["pr"] == 101]
assert len(scores) == 1, f"PR 101 must get exactly one score comment: {scores}"
assert "pullsift-score" in json.dumps(scores), f"score comment must carry the marker: {scores}"
assert "No action taken" in json.dumps(scores), f"score comment must state the verdict: {scores}"
assert updates, f"the rescore must edit the score comment: {calls}"
print("PASS: PR 901 closed with evidence, PR 101 scored with one edited comment")
EOF
