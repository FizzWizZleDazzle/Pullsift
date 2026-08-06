//! Webhook intake: HMAC verification and minimal event parsing. The handler
//! stays thin; everything heavy happens in the pipeline.

use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::Sha256;
use subtle::ConstantTimeEq;

/// Verify `X-Hub-Signature-256: sha256=<hex>` over the raw body.
pub fn verify_signature(secret: &str, body: &[u8], header: &str) -> bool {
    let Some(hex_sig) = header.strip_prefix("sha256=") else {
        return false;
    };
    let Ok(given) = hex::decode(hex_sig) else {
        return false;
    };
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("hmac accepts any key");
    mac.update(body);
    let expected = mac.finalize().into_bytes();
    expected.ct_eq(given.as_slice()).into()
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrEvent {
    pub action: String,
    pub repo: String,
    pub number: u64,
    pub author: String,
    pub title: String,
    pub body: String,
    pub additions: u64,
    pub deletions: u64,
    pub changed_files: u64,
    pub commit_count: u64,
    pub author_association: String,
    pub head_is_fork: bool,
    pub head_ref: String,
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommentEvent {
    pub repo: String,
    pub issue_number: u64,
    pub commenter: String,
    pub body: String,
    pub on_pull_request: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    PullRequest(PrEvent),
    Comment(CommentEvent),
    Ping,
    Ignored(String),
}

pub fn parse(event_name: &str, payload: &Value) -> Event {
    match event_name {
        "ping" => Event::Ping,
        "pull_request" => {
            let pr = &payload["pull_request"];
            Event::PullRequest(PrEvent {
                action: payload["action"].as_str().unwrap_or("").to_string(),
                repo: payload["repository"]["full_name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                number: pr["number"].as_u64().unwrap_or(0),
                author: pr["user"]["login"].as_str().unwrap_or("").to_string(),
                title: pr["title"].as_str().unwrap_or("").to_string(),
                body: pr["body"].as_str().unwrap_or("").to_string(),
                additions: pr["additions"].as_u64().unwrap_or(0),
                deletions: pr["deletions"].as_u64().unwrap_or(0),
                changed_files: pr["changed_files"].as_u64().unwrap_or(0),
                commit_count: pr["commits"].as_u64().unwrap_or(0),
                author_association: pr["author_association"].as_str().unwrap_or("").to_string(),
                head_is_fork: pr["head"]["repo"]["fork"].as_bool().unwrap_or(false),
                head_ref: pr["head"]["ref"].as_str().unwrap_or("").to_string(),
                node_id: pr["node_id"].as_str().unwrap_or("").to_string(),
            })
        }
        "issue_comment" => {
            if payload["action"].as_str() != Some("created") {
                return Event::Ignored("issue_comment non-created".into());
            }
            Event::Comment(CommentEvent {
                repo: payload["repository"]["full_name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                issue_number: payload["issue"]["number"].as_u64().unwrap_or(0),
                commenter: payload["comment"]["user"]["login"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                body: payload["comment"]["body"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                on_pull_request: !payload["issue"]["pull_request"].is_null(),
            })
        }
        other => Event::Ignored(other.to_string()),
    }
}

/// PR actions the pipeline scores.
pub fn is_scorable_action(action: &str) -> bool {
    matches!(
        action,
        "opened" | "reopened" | "synchronize" | "ready_for_review"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmac_accepts_valid_and_rejects_invalid() {
        let secret = "s3cret";
        let body = b"payload-bytes";
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        let good = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));
        assert!(verify_signature(secret, body, &good));
        assert!(!verify_signature("wrong", body, &good));
        assert!(!verify_signature(secret, b"other body", &good));
        assert!(!verify_signature(secret, body, "sha256=deadbeef"));
        assert!(!verify_signature(secret, body, "sha1=whatever"));
        assert!(!verify_signature(secret, body, "sha256=nothex!"));
    }

    #[test]
    fn parses_pull_request_event() {
        let payload = serde_json::json!({
            "action": "opened",
            "repository": { "full_name": "octo/repo" },
            "pull_request": {
                "number": 42,
                "node_id": "PR_abc",
                "title": "Update README.md",
                "body": "made it better",
                "additions": 2,
                "changed_files": 1,
                "author_association": "FIRST_TIME_CONTRIBUTOR",
                "user": { "login": "newcomer" },
                "commits": 1,
                "deletions": 1,
                "head": { "ref": "patch-1", "repo": { "fork": true } }
            }
        });
        let Event::PullRequest(pr) = parse("pull_request", &payload) else {
            panic!()
        };
        assert_eq!(pr.repo, "octo/repo");
        assert_eq!(pr.number, 42);
        assert_eq!(pr.author, "newcomer");
        assert!(pr.head_is_fork);
        assert_eq!(pr.author_association, "FIRST_TIME_CONTRIBUTOR");
        assert_eq!(pr.head_ref, "patch-1");
        assert_eq!(pr.commit_count, 1);
        assert_eq!(pr.deletions, 1);
        assert!(is_scorable_action(&pr.action));
    }

    #[test]
    fn parses_comment_event_on_pr() {
        let payload = serde_json::json!({
            "action": "created",
            "repository": { "full_name": "octo/repo" },
            "issue": { "number": 42, "pull_request": { "url": "..." } },
            "comment": { "user": { "login": "newcomer" }, "body": "it fixes the thing" }
        });
        let Event::Comment(c) = parse("issue_comment", &payload) else {
            panic!()
        };
        assert!(c.on_pull_request);
        assert_eq!(c.issue_number, 42);
    }

    #[test]
    fn comment_edits_are_ignored() {
        let payload = serde_json::json!({ "action": "edited" });
        assert!(matches!(
            parse("issue_comment", &payload),
            Event::Ignored(_)
        ));
    }

    #[test]
    fn unknown_events_are_ignored() {
        assert!(matches!(
            parse("workflow_run", &serde_json::json!({})),
            Event::Ignored(_)
        ));
        assert_eq!(parse("ping", &serde_json::json!({})), Event::Ping);
    }

    #[test]
    fn scorable_actions() {
        assert!(is_scorable_action("opened"));
        assert!(is_scorable_action("synchronize"));
        assert!(!is_scorable_action("closed"));
        assert!(!is_scorable_action("labeled"));
    }
}
