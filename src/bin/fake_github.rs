//! A fake GitHub API for end-to-end tests. Serves just enough of the REST
//! and GraphQL surface for the service to run a full webhook-to-action
//! round, records every mutating call, and dumps the record at `/_calls`.
//!
//! PR numbers steer behavior: 900-999 look like slop (agent commit email
//! and trailer, plus a spam-trail author dossier), everything else looks
//! like an ordinary contribution.

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct Calls(Arc<Mutex<Vec<Value>>>);

impl Calls {
    fn push(&self, v: Value) {
        self.0.lock().unwrap().push(v);
    }
}

#[tokio::main]
async fn main() {
    let calls = Calls::default();
    let app = Router::new()
        .route("/_calls", get(dump))
        .route("/app/installations/{id}/access_tokens", post(token))
        .route("/repos/{owner}/{repo}/pulls/{n}", get(pull).patch(close))
        .route("/repos/{owner}/{repo}/pulls/{n}/files", get(files))
        .route("/repos/{owner}/{repo}/pulls/{n}/commits", get(commits))
        .route("/repos/{owner}/{repo}/issues/{n}/comments", post(comment))
        .route("/repos/{owner}/{repo}/issues/{n}/labels", post(label))
        .route("/repos/{owner}/{repo}/contents/{*path}", get(contents))
        .route("/graphql", post(graphql))
        .with_state(calls);

    let addr = std::env::var("FAKE_BIND").unwrap_or_else(|_| "127.0.0.1:9299".into());
    eprintln!("fake github on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn dump(State(calls): State<Calls>) -> Json<Value> {
    Json(Value::Array(calls.0.lock().unwrap().clone()))
}

async fn token() -> Json<Value> {
    Json(json!({ "token": "test-installation-token" }))
}

fn is_slop(n: u64) -> bool {
    (900..1000).contains(&n)
}

async fn pull(
    Path((_, _, n)): Path<(String, String, u64)>,
    headers: HeaderMap,
) -> axum::response::Response {
    let accept = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if accept.contains("diff") {
        let diff = if is_slop(n) {
            "--- a/README.md\n+++ b/README.md\n@@ -1,2 +1,4 @@\n\
             +This project is a comprehensive framework for building apps.\n\
             +It delivers seamless integration and robust best practices.\n"
        } else {
            "--- a/src/parser.rs\n+++ b/src/parser.rs\n@@ -10,3 +10,6 @@\n\
             +fn unquote(s: &str) -> &str {\n\
             +    s.strip_prefix('\"').and_then(|s| s.strip_suffix('\"')).unwrap_or(s)\n\
             +}\n"
        };
        return axum::response::Response::builder()
            .header("content-type", "application/vnd.github.diff")
            .body(diff.to_string().into())
            .unwrap();
    }
    axum::response::Json(json!({ "number": n, "state": "open" })).into_response()
}

use axum::response::IntoResponse;

async fn files(Path((_, _, n)): Path<(String, String, u64)>) -> Json<Value> {
    let name = if is_slop(n) {
        "README.md"
    } else {
        "src/parser.rs"
    };
    Json(json!([ { "filename": name } ]))
}

async fn commits(Path((_, _, n)): Path<(String, String, u64)>) -> Json<Value> {
    if is_slop(n) {
        Json(json!([{ "commit": {
            "author": { "email": "noreply@anthropic.com" },
            "message": "docs: improve readme\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
        }}]))
    } else {
        Json(json!([{ "commit": {
            "author": { "email": "dev@example.com" },
            "message": "fix: strip quotes in parser"
        }}]))
    }
}

async fn close(
    State(calls): State<Calls>,
    Path((owner, repo, n)): Path<(String, String, u64)>,
    Json(body): Json<Value>,
) -> Json<Value> {
    calls
        .push(json!({ "call": "close", "repo": format!("{owner}/{repo}"), "pr": n, "body": body }));
    Json(json!({ "state": "closed" }))
}

async fn comment(
    State(calls): State<Calls>,
    Path((owner, repo, n)): Path<(String, String, u64)>,
    Json(body): Json<Value>,
) -> Json<Value> {
    calls.push(
        json!({ "call": "comment", "repo": format!("{owner}/{repo}"), "pr": n, "body": body }),
    );
    Json(json!({ "id": 1 }))
}

async fn label(
    State(calls): State<Calls>,
    Path((owner, repo, n)): Path<(String, String, u64)>,
    Json(body): Json<Value>,
) -> Json<Value> {
    calls
        .push(json!({ "call": "label", "repo": format!("{owner}/{repo}"), "pr": n, "body": body }));
    Json(json!([]))
}

async fn contents(Path((_, _, path)): Path<(String, String, String)>) -> axum::response::Response {
    if path.ends_with("slopcatcher.yml") {
        return axum::response::Response::builder()
            .body("dry_run: false\n".to_string().into())
            .unwrap();
    }
    axum::response::Response::builder()
        .status(404)
        .body("not found".to_string().into())
        .unwrap()
}

async fn graphql(State(calls): State<Calls>, Json(body): Json<Value>) -> Json<Value> {
    let query = body["query"].as_str().unwrap_or("");
    if query.contains("convertPullRequestToDraft") {
        calls.push(json!({ "call": "draft", "body": body }));
        return Json(json!({ "data": { "convertPullRequestToDraft":
            { "pullRequest": { "isDraft": true } } } }));
    }
    // Dossier query. Slop authors (login starts with "ghost") get the shape
    // the model convicts on: a young opaque account with a closed-unmerged,
    // spam-labeled trail it never followed up on. Everyone else has no
    // visible history.
    let login = body["variables"]["login"].as_str().unwrap_or("");
    if !login.starts_with("ghost") {
        return Json(json!({ "data": { "user": null } }));
    }
    let now = chrono::Utc::now();
    let created = (now - chrono::Duration::days(5)).to_rfc3339();
    let pr_at = (now - chrono::Duration::days(2)).to_rfc3339();
    let review_at = (now - chrono::Duration::days(1)).to_rfc3339();
    let prior: Vec<Value> = (0..6)
        .map(|i| {
            json!({
                "state": "CLOSED",
                "merged": false,
                "title": "Update README.md",
                "createdAt": pr_at,
                "labels": { "nodes": if i < 3 { json!([{ "name": "spam" }]) } else { json!([]) } },
                "reviews": { "totalCount": 1 },
                "comments": { "nodes": [
                    { "author": { "login": "maintainer" }, "createdAt": review_at }
                ] },
                "repository": { "nameWithOwner": format!("org{i}/repo{i}"),
                                "primaryLanguage": { "name": "Markdown" } }
            })
        })
        .collect();
    Json(json!({ "data": { "user": {
        "createdAt": created,
        "bio": null,
        "followers": { "totalCount": 0 },
        "contributionsCollection": { "restrictedContributionsCount": 12 },
        "pullRequests": { "nodes": prior }
    } } }))
}
