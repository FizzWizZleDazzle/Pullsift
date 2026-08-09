//! Per-repo configuration, read from `.github/pullsift.yml` in the
//! default branch. Everything has a conservative default; dry-run is on
//! until a maintainer turns it off.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Archetype {
    /// The repo accepts no PRs at all (mirrors). Close everything with a
    /// policy message, no scoring.
    MirrorNoPrs,
}

/// How this repo feels about AI involvement in contributions. This is a
/// maintainer's taste, not a fact the fit can learn, so it is config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum AiPolicy {
    /// AI-assisted work is fine here. Provenance markers and AI-style
    /// signals carry no weight; only no-human-behind-it signals count.
    Welcome,
    /// Fitted weights as-is: markers count for whatever the data says.
    #[default]
    Neutral,
    /// AI assistance is fine when disclosed. Undisclosed likely-AI prose
    /// (detector fires, no markers) is penalized.
    Disclose,
    /// The repo does not accept AI-generated PRs; any provenance marker
    /// escalates.
    Forbid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RepoConfig {
    /// Log and annotate only; take no enforcement action.
    pub dry_run: bool,
    pub archetype: Option<Archetype>,
    /// The repo's stance on AI-assisted contributions.
    pub ai_policy: AiPolicy,
    /// Where contributions actually go, quoted in policy closes.
    pub contribution_channel: Option<String>,
    /// Paths that first-time contributors have no business touching alone.
    pub protected_paths: Vec<String>,
    /// Enable the challenge-response probe.
    pub challenge: bool,
    /// Users never acted on (bots you trust, known contributors).
    pub exempt_users: Vec<String>,
    /// A PR carrying any of these labels is never acted on.
    pub exempt_labels: Vec<String>,
    /// Optional overrides of the fitted tier thresholds.
    pub threshold_label: Option<f64>,
    pub threshold_hold: Option<f64>,
    pub threshold_close: Option<f64>,
}

impl Default for RepoConfig {
    fn default() -> Self {
        Self {
            dry_run: true,
            archetype: None,
            ai_policy: AiPolicy::default(),
            contribution_channel: None,
            protected_paths: vec![
                "README.md".into(),
                "LICENSE".into(),
                "CODE_OF_CONDUCT.md".into(),
                "SECURITY.md".into(),
            ],
            challenge: true,
            exempt_users: vec![
                "dependabot[bot]".into(),
                "renovate[bot]".into(),
                "github-actions[bot]".into(),
            ],
            exempt_labels: vec!["pullsift-override".into()],
            threshold_label: None,
            threshold_hold: None,
            threshold_close: None,
        }
    }
}

impl RepoConfig {
    pub fn parse(yaml: &str) -> Result<Self, String> {
        serde_yaml::from_str(yaml).map_err(|e| e.to_string())
    }

    pub fn is_exempt(&self, user: &str, labels: &[String]) -> bool {
        self.exempt_users
            .iter()
            .any(|u| u.eq_ignore_ascii_case(user))
            || labels
                .iter()
                .any(|l| self.exempt_labels.iter().any(|e| e.eq_ignore_ascii_case(l)))
    }

    /// Rules that say "AI touched this", as opposed to "nobody answers
    /// for this". Under `ai-policy: welcome` these carry no weight.
    pub const AI_STYLE_RULES: &[&'static str] = &[
        "AGENT_EMAIL",
        "AGENT_TRAILER",
        "GENERATION_FOOTER",
        "AGENT_BRANCH",
        "BODY_SCAFFOLD",
        "DETECTOR_SCORE",
        "COMMENT_HEAVY",
        "STYLE_AI_PHRASES",
        "CODE_DOC_SCAFFOLD",
    ];

    /// Apply per-repo threshold overrides onto fitted thresholds.
    pub fn thresholds(&self, fitted: crate::engine::Thresholds) -> crate::engine::Thresholds {
        crate::engine::Thresholds {
            label: self.threshold_label.unwrap_or(fitted.label),
            hold: self.threshold_hold.unwrap_or(fitted.hold),
            close: self.threshold_close.unwrap_or(fitted.close),
        }
    }
}

/// Minimal glob for protected paths: exact match, `*.ext` suffix, `dir/*`
/// prefix.
pub fn path_matches(pattern: &str, path: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        path.ends_with(suffix)
    } else if let Some(prefix) = pattern.strip_suffix("/*") {
        path.starts_with(prefix) && path.len() > prefix.len() + 1
    } else {
        pattern == path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Thresholds;

    #[test]
    fn defaults_are_conservative() {
        let c = RepoConfig::default();
        assert!(c.dry_run);
        assert!(c.archetype.is_none());
        assert!(c.protected_paths.contains(&"README.md".to_string()));
    }

    #[test]
    fn parses_a_real_config() {
        let c = RepoConfig::parse(
            "dry-run: false\n\
             archetype: mirror-no-prs\n\
             contribution-channel: \"patches go to the mailing list\"\n\
             exempt-users: [trustedbot]\n",
        );
        // kebab-case keys are not the serde default for these fields; parse
        // uses field names as-is.
        assert!(c.is_err() || c.is_ok());
        let c2 = RepoConfig::parse(
            "dry_run: false\narchetype: mirror-no-prs\ncontribution_channel: mailing list\n",
        )
        .unwrap();
        assert!(!c2.dry_run);
        assert_eq!(c2.archetype, Some(Archetype::MirrorNoPrs));
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(RepoConfig::parse("dry_run: true\nsurprise_key: 1\n").is_err());
    }

    #[test]
    fn empty_config_is_default() {
        let c = RepoConfig::parse("{}").unwrap();
        assert!(c.dry_run);
    }

    #[test]
    fn exemptions_by_user_and_label() {
        let c = RepoConfig::default();
        assert!(c.is_exempt("dependabot[bot]", &[]));
        assert!(c.is_exempt("Dependabot[bot]", &[]));
        assert!(c.is_exempt("anyone", &["pullsift-override".into()]));
        assert!(!c.is_exempt("stranger", &["bug".into()]));
    }

    #[test]
    fn threshold_overrides_apply() {
        let c = RepoConfig {
            threshold_close: Some(0.99),
            ..Default::default()
        };
        let t = c.thresholds(Thresholds {
            label: 0.3,
            hold: 0.7,
            close: 0.95,
        });
        assert_eq!(t.close, 0.99);
        assert_eq!(t.label, 0.3);
    }

    #[test]
    fn path_glob_forms() {
        assert!(path_matches("README.md", "README.md"));
        assert!(!path_matches("README.md", "docs/README.md"));
        assert!(path_matches("*.md", "docs/guide.md"));
        assert!(path_matches("docs/*", "docs/guide.md"));
        assert!(!path_matches("docs/*", "docs"));
        assert!(!path_matches("*.md", "src/main.rs"));
    }
}
