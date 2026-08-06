//! Thin GitHub client: App JWT -> installation token -> REST/GraphQL. Kept
//! deliberately small; everything testable lives outside this module.

use anyhow::{Context, Result};
use chrono::Utc;
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde::Serialize;
use serde_json::Value;

const API: &str = "https://api.github.com";
const USER_AGENT: &str = "pullsift/0.1";

#[derive(Serialize)]
struct AppClaims {
    iat: i64,
    exp: i64,
    iss: String,
}

/// A short-lived App JWT for the installation-token endpoint.
pub fn app_jwt(app_id: &str, private_key_pem: &str) -> Result<String> {
    let now = Utc::now().timestamp();
    let claims = AppClaims {
        iat: now - 60,
        exp: now + 9 * 60,
        iss: app_id.to_string(),
    };
    let key = EncodingKey::from_rsa_pem(private_key_pem.as_bytes())
        .context("GitHub App private key must be RSA PEM")?;
    jsonwebtoken::encode(&Header::new(Algorithm::RS256), &claims, &key).context("jwt encode")
}

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
}

impl Client {
    pub fn new() -> Self {
        Self::with_base(API)
    }

    /// Base override for tests against a local server.
    pub fn with_base(base: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .build()
                .expect("reqwest client"),
            base: base.trim_end_matches('/').to_string(),
        }
    }

    pub async fn installation_token(&self, jwt: &str, installation_id: u64) -> Result<String> {
        let url = format!(
            "{}/app/installations/{installation_id}/access_tokens",
            self.base
        );
        let resp: Value = self
            .http
            .post(url)
            .bearer_auth(jwt)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        resp["token"]
            .as_str()
            .map(|s| s.to_string())
            .context("no token in response")
    }

    /// The PR's unified diff.
    pub async fn pr_diff(&self, token: &str, repo: &str, number: u64) -> Result<String> {
        let url = format!("{}/repos/{repo}/pulls/{number}", self.base);
        Ok(self
            .http
            .get(url)
            .bearer_auth(token)
            .header("Accept", "application/vnd.github.diff")
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?)
    }

    /// Changed file paths of a PR (first page, 100 files: enough to type a
    /// campaign; giant PRs are their own signal elsewhere).
    pub async fn pr_files(&self, token: &str, repo: &str, number: u64) -> Result<Vec<String>> {
        let url = format!(
            "{}/repos/{repo}/pulls/{number}/files?per_page=100",
            self.base
        );
        let resp: Value = self
            .http
            .get(url)
            .bearer_auth(token)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp
            .as_array()
            .map(|files| {
                files
                    .iter()
                    .filter_map(|f| f["filename"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Commit author emails and messages for marker scanning.
    pub async fn pr_commits(
        &self,
        token: &str,
        repo: &str,
        number: u64,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let url = format!(
            "{}/repos/{repo}/pulls/{number}/commits?per_page=100",
            self.base
        );
        let resp: Value = self
            .http
            .get(url)
            .bearer_auth(token)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let mut emails = Vec::new();
        let mut messages = Vec::new();
        if let Some(commits) = resp.as_array() {
            for c in commits {
                if let Some(e) = c["commit"]["author"]["email"].as_str() {
                    emails.push(e.to_string());
                }
                if let Some(m) = c["commit"]["message"].as_str() {
                    messages.push(m.to_string());
                }
            }
        }
        Ok((emails, messages))
    }

    pub async fn post_comment(
        &self,
        token: &str,
        repo: &str,
        number: u64,
        body: &str,
    ) -> Result<()> {
        let url = format!("{}/repos/{repo}/issues/{number}/comments", self.base);
        self.http
            .post(url)
            .bearer_auth(token)
            .json(&serde_json::json!({ "body": body }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn add_label(&self, token: &str, repo: &str, number: u64, label: &str) -> Result<()> {
        let url = format!("{}/repos/{repo}/issues/{number}/labels", self.base);
        self.http
            .post(url)
            .bearer_auth(token)
            .json(&serde_json::json!({ "labels": [label] }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn close_pr(&self, token: &str, repo: &str, number: u64) -> Result<()> {
        let url = format!("{}/repos/{repo}/pulls/{number}", self.base);
        self.http
            .patch(url)
            .bearer_auth(token)
            .json(&serde_json::json!({ "state": "closed" }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// Draft conversion is GraphQL-only.
    pub async fn convert_to_draft(&self, token: &str, pr_node_id: &str) -> Result<()> {
        let q = serde_json::json!({
            "query": "mutation($id: ID!) { convertPullRequestToDraft(input: {pullRequestId: $id}) { pullRequest { isDraft } } }",
            "variables": { "id": pr_node_id }
        });
        self.graphql(token, &q).await.map(|_| ())
    }

    pub async fn graphql(&self, token: &str, body: &Value) -> Result<Value> {
        let url = format!("{}/graphql", self.base);
        let resp: Value = self
            .http
            .post(url)
            .bearer_auth(token)
            .json(body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        if !resp["errors"].is_null() {
            anyhow::bail!("graphql errors: {}", resp["errors"]);
        }
        Ok(resp)
    }

    /// Raw file from the default branch; None when absent.
    pub async fn file(&self, token: &str, repo: &str, path: &str) -> Result<Option<String>> {
        let url = format!("{}/repos/{repo}/contents/{path}", self.base);
        let resp = self
            .http
            .get(url)
            .bearer_auth(token)
            .header("Accept", "application/vnd.github.raw+json")
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(resp.error_for_status()?.text().await?))
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new()
    }
}
