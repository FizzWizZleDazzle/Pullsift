//! The service: axum webhook intake, input assembly around the pure
//! pipeline, action execution, and the nightly learner.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use chrono::Utc;
use slopcatcher::actions::{action_name, PlannedAction, SUSPECT_LABEL};
use slopcatcher::challenge::{self, ChallengeState};
use slopcatcher::cluster::ClusterStore;
use slopcatcher::config::RepoConfig;
use slopcatcher::dossier;
use slopcatcher::engine::Weights;
use slopcatcher::github::{app_jwt, Client};
use slopcatcher::pipeline::{self, Outcome, ScoreInputs};
use slopcatcher::store::{PgStore, Store};
use slopcatcher::webhook::{self, Event};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, warn};

struct AppState {
    webhook_secret: String,
    canary_salt: String,
    app_id: String,
    private_key_pem: String,
    installation_id: u64,
    github: Client,
    store: PgStore,
    weights: RwLock<Weights>,
    clusters: Mutex<HashMap<String, ClusterStore>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let env = |k: &str| std::env::var(k).map_err(|_| anyhow::anyhow!("missing env {k}"));
    let database_url = env("DATABASE_URL")?;
    let store = PgStore::connect(&database_url).await?;

    let weights = match store.active_weights().await? {
        Some(w) => {
            info!("loaded active weights from database");
            w
        }
        None => {
            let w = Weights::default_table();
            store
                .promote_weights(&w, "bootstrap defaults", f64::NAN)
                .await?;
            info!("promoted embedded default weights");
            w
        }
    };

    let state = Arc::new(AppState {
        webhook_secret: env("WEBHOOK_SECRET")?,
        canary_salt: env("CANARY_SALT")?,
        app_id: env("GITHUB_APP_ID")?,
        private_key_pem: std::fs::read_to_string(env("GITHUB_PRIVATE_KEY_PATH")?)?,
        installation_id: env("GITHUB_INSTALLATION_ID")?.parse()?,
        // GITHUB_API_BASE overrides the API host; the e2e harness points it
        // at the fake server.
        github: match std::env::var("GITHUB_API_BASE") {
            Ok(base) => Client::with_base(&base),
            Err(_) => Client::new(),
        },
        store,
        weights: RwLock::new(weights),
        clusters: Mutex::new(HashMap::new()),
    });

    // Nightly learner: refit from stored feedback, promote behind guardrails.
    {
        let state = state.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(24 * 3600));
            tick.tick().await; // first tick fires immediately; skip it
            loop {
                tick.tick().await;
                if let Err(e) = run_learner(&state).await {
                    error!("learner: {e:#}");
                }
            }
        });
    }

    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/webhook", post(handle_webhook))
        .with_state(state);

    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".into());
    info!("listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_learner(state: &AppState) -> anyhow::Result<()> {
    let examples = state.store.load_examples().await?;
    let incumbent = state.weights.read().await.clone();
    let outcome = slopcatcher::learn::learn(&examples, &incumbent);
    info!("learner: {}", outcome.reason);
    if let (true, Some(w)) = (outcome.promoted, outcome.weights) {
        state
            .store
            .promote_weights(&w, &outcome.reason, outcome.candidate_auc)
            .await?;
        *state.weights.write().await = w;
    }
    Ok(())
}

async fn handle_webhook(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> StatusCode {
    let signature = headers
        .get("x-hub-signature-256")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !webhook::verify_signature(&state.webhook_secret, &body, signature) {
        return StatusCode::UNAUTHORIZED;
    }
    let event_name = headers
        .get("x-github-event")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return StatusCode::BAD_REQUEST;
    };

    let event = webhook::parse(&event_name, &payload);
    let state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = dispatch(&state, event, payload).await {
            error!("dispatch: {e:#}");
        }
    });
    StatusCode::ACCEPTED
}

async fn dispatch(
    state: &AppState,
    event: Event,
    payload: serde_json::Value,
) -> anyhow::Result<()> {
    match event {
        Event::PullRequest(ev) if webhook::is_scorable_action(&ev.action) => {
            state
                .store
                .record_event(&ev.repo, ev.number, &ev.author, &ev.action, &payload)
                .await?;
            score_and_act(state, ev).await
        }
        Event::Comment(c) if c.on_pull_request => handle_reply(state, c).await,
        _ => Ok(()),
    }
}

async fn token(state: &AppState) -> anyhow::Result<String> {
    let jwt = app_jwt(&state.app_id, &state.private_key_pem)?;
    state
        .github
        .installation_token(&jwt, state.installation_id)
        .await
}

/// Optional AI-text detector: DETECTOR_URL points at a locally hosted model
/// server (POST {"text": ...} -> {"probability": 0..1}). Fail-open; a slow
/// or absent detector never blocks scoring.
async fn detector_probe(title: &str, body: &str) -> Option<f64> {
    let url = std::env::var("DETECTOR_URL").ok()?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    let resp: serde_json::Value = client
        .post(&url)
        .json(&serde_json::json!({ "text": format!("{title}\n{body}") }))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    resp["probability"]
        .as_f64()
        .filter(|p| (0.0..=1.0).contains(p))
}

async fn score_and_act(state: &AppState, ev: webhook::PrEvent) -> anyhow::Result<()> {
    let now = Utc::now();
    let token = token(state).await?;
    let gh = &state.github;

    let config = match gh.file(&token, &ev.repo, ".github/slopcatcher.yml").await? {
        Some(yaml) => RepoConfig::parse(&yaml).unwrap_or_else(|e| {
            warn!("{}: bad slopcatcher.yml ({e}); using defaults", ev.repo);
            RepoConfig::default()
        }),
        None => RepoConfig::default(),
    };
    let template = gh
        .file(&token, &ev.repo, ".github/PULL_REQUEST_TEMPLATE.md")
        .await
        .unwrap_or(None);

    let diff = gh.pr_diff(&token, &ev.repo, ev.number).await?;
    let changed_paths = gh.pr_files(&token, &ev.repo, ev.number).await?;
    let (commit_emails, commit_messages) = gh.pr_commits(&token, &ev.repo, ev.number).await?;

    let dossier_facts = match state.store.get_dossier(&ev.author, now).await? {
        Some(f) => f,
        None => {
            let resp = gh
                .graphql(&token, &dossier::dossier_query(&ev.author))
                .await?;
            let facts = dossier::parse_dossier(&ev.author, &resp, now);
            state.store.put_dossier(&facts, now).await?;
            facts
        }
    };

    let detector_score = detector_probe(&ev.title, &ev.body).await;

    let inputs = ScoreInputs {
        config: &config,
        event: &ev,
        diff: &diff,
        changed_paths,
        commit_emails,
        commit_messages,
        dossier: dossier_facts,
        pr_labels: vec![],
        template: template.as_deref(),
        detector_score,
    };

    let outcome = {
        let mut clusters = state.clusters.lock().await;
        let store = clusters
            .entry(ev.repo.clone())
            .or_insert_with(|| ClusterStore::new(0.5));
        store.prune(now);
        let weights = state.weights.read().await.clone();
        pipeline::process(&inputs, &weights, store, &state.canary_salt, now)
    };

    match outcome {
        Outcome::Exempt => Ok(()),
        Outcome::PolicyClose { comment } => {
            if config.dry_run {
                state
                    .store
                    .log_action(
                        &ev.repo,
                        ev.number,
                        None,
                        "close-by-policy(dry)",
                        true,
                        &serde_json::json!({ "comment": comment }),
                    )
                    .await?;
                return Ok(());
            }
            state
                .github
                .post_comment(&token, &ev.repo, ev.number, &comment)
                .await?;
            state.github.close_pr(&token, &ev.repo, ev.number).await?;
            state
                .store
                .log_action(
                    &ev.repo,
                    ev.number,
                    None,
                    "close-by-policy",
                    false,
                    &serde_json::json!({ "comment": comment }),
                )
                .await
        }
        Outcome::Scored { verdict, planned } => {
            let verdict_id = state
                .store
                .save_verdict(&ev.repo, ev.number, &ev.author, &verdict)
                .await?;
            execute(state, &token, &ev, verdict_id, planned, config.dry_run).await
        }
    }
}

async fn execute(
    state: &AppState,
    token: &str,
    ev: &webhook::PrEvent,
    verdict_id: i64,
    planned: PlannedAction,
    dry_run: bool,
) -> anyhow::Result<()> {
    let gh = &state.github;
    let name = action_name(&planned).to_string();
    match &planned {
        PlannedAction::None => {}
        PlannedAction::Label { evidence_comment } => {
            gh.add_label(token, &ev.repo, ev.number, SUSPECT_LABEL)
                .await?;
            gh.post_comment(token, &ev.repo, ev.number, evidence_comment)
                .await?;
        }
        PlannedAction::Hold {
            evidence_comment,
            challenge_comment,
        } => {
            gh.add_label(token, &ev.repo, ev.number, SUSPECT_LABEL)
                .await?;
            gh.convert_to_draft(token, &ev.node_id).await?;
            gh.post_comment(token, &ev.repo, ev.number, evidence_comment)
                .await?;
            if let Some(c) = challenge_comment {
                gh.post_comment(token, &ev.repo, ev.number, c).await?;
                let canary = challenge::canary_token(&ev.repo, ev.number, &state.canary_salt);
                state
                    .store
                    .put_challenge(
                        &ev.repo,
                        ev.number,
                        &ChallengeState::Pending {
                            canary,
                            posted_at: Utc::now(),
                        },
                    )
                    .await?;
            }
        }
        PlannedAction::Close { comment } | PlannedAction::CloseByPolicy { comment } => {
            gh.post_comment(token, &ev.repo, ev.number, comment).await?;
            gh.close_pr(token, &ev.repo, ev.number).await?;
        }
    }
    state
        .store
        .log_action(
            &ev.repo,
            ev.number,
            Some(verdict_id),
            &name,
            dry_run,
            &serde_json::json!({}),
        )
        .await
}

/// Author replies on held PRs resolve pending challenges.
async fn handle_reply(state: &AppState, c: webhook::CommentEvent) -> anyhow::Result<()> {
    let Some(pending) = state.store.get_challenge(&c.repo, c.issue_number).await? else {
        return Ok(());
    };
    let next = challenge::evaluate_reply(&pending, &c.body);
    if next == pending {
        return Ok(());
    }
    state
        .store
        .put_challenge(&c.repo, c.issue_number, &next)
        .await?;
    let token = token(state).await?;
    match &next {
        ChallengeState::FailedCanary => {
            let msg = "The reply to the review-readiness check included the \
                       hidden canary token, which only automated pipelines \
                       reproduce. Closing; a maintainer can reopen.";
            state
                .github
                .post_comment(&token, &c.repo, c.issue_number, msg)
                .await?;
            state
                .github
                .close_pr(&token, &c.repo, c.issue_number)
                .await?;
            state
                .store
                .log_action(
                    &c.repo,
                    c.issue_number,
                    None,
                    "close-canary",
                    false,
                    &serde_json::json!({ "commenter": c.commenter }),
                )
                .await?;
        }
        ChallengeState::Passed => {
            info!(
                "{}#{}: challenge passed by {}",
                c.repo, c.issue_number, c.commenter
            );
            state
                .store
                .log_action(
                    &c.repo,
                    c.issue_number,
                    None,
                    "challenge-passed",
                    false,
                    &serde_json::json!({ "commenter": c.commenter }),
                )
                .await?;
        }
        _ => {}
    }
    Ok(())
}
