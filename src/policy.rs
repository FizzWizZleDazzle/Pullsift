//! Lane C: repo policy. Runs before any scoring; the cheapest lane wins.

use crate::config::{path_matches, Archetype, RepoConfig};
use crate::engine::Fire;

#[derive(Debug, Clone, PartialEq)]
pub enum PolicyOutcome {
    /// Close immediately with a policy message; no scoring.
    CloseByPolicy(String),
    /// Feed extra rules into the engine.
    ExtraRules(Vec<Fire>),
}

pub struct PrMeta<'a> {
    pub author: &'a str,
    pub changed_paths: &'a [String],
    pub is_first_time_contributor: bool,
    pub body: &'a str,
    /// The repo's PR template, when it has one.
    pub template: Option<&'a str>,
}

pub fn evaluate(config: &RepoConfig, pr: &PrMeta) -> PolicyOutcome {
    if config.archetype == Some(Archetype::MirrorNoPrs) {
        let channel = config
            .contribution_channel
            .as_deref()
            .unwrap_or("the project's documented contribution channel");
        return PolicyOutcome::CloseByPolicy(format!(
            "This repository does not accept pull requests. Contributions go \
             through {channel}. This close is repository policy, not a \
             judgment of your change."
        ));
    }

    let mut rules = Vec::new();

    if pr.is_first_time_contributor && !pr.changed_paths.is_empty() {
        let all_protected = pr.changed_paths.iter().all(|p| {
            config
                .protected_paths
                .iter()
                .any(|pat| path_matches(pat, p))
        });
        if all_protected {
            rules.push(Fire::hit("PROTECTED_PATH_ONLY"));
        }
    }

    let docs_only = !pr.changed_paths.is_empty()
        && pr
            .changed_paths
            .iter()
            .all(|p| p.to_lowercase().ends_with(".md") || p.to_lowercase().starts_with("docs/"));
    if docs_only {
        rules.push(Fire::hit("DOCS_ONLY"));
    }

    if pr.body.trim().is_empty() {
        rules.push(Fire::hit("BODY_EMPTY"));
    } else if let Some(template) = pr.template {
        if template_ignored(template, pr.body) {
            rules.push(Fire::hit("TEMPLATE_IGNORED"));
        }
    }

    PolicyOutcome::ExtraRules(rules)
}

/// The template counts as ignored when none of its section headers appear in
/// the body.
fn template_ignored(template: &str, body: &str) -> bool {
    let headers: Vec<&str> = template
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.starts_with('#'))
        .collect();
    if headers.is_empty() {
        return false;
    }
    !headers.iter().any(|h| body.contains(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta<'a>(paths: &'a [String], first_time: bool, body: &'a str) -> PrMeta<'a> {
        PrMeta {
            author: "someone",
            changed_paths: paths,
            is_first_time_contributor: first_time,
            body,
            template: None,
        }
    }

    #[test]
    fn mirror_archetype_closes_everything() {
        let c = RepoConfig {
            archetype: Some(crate::config::Archetype::MirrorNoPrs),
            contribution_channel: Some("the kernel mailing list".into()),
            ..Default::default()
        };
        let paths = vec!["kernel/sched.c".into()];
        let out = evaluate(&c, &meta(&paths, false, "big feature"));
        match out {
            PolicyOutcome::CloseByPolicy(msg) => {
                assert!(msg.contains("kernel mailing list"));
                assert!(msg.contains("not a judgment"));
            }
            _ => panic!("expected policy close"),
        }
    }

    #[test]
    fn readme_only_first_timer_fires_protected() {
        let c = RepoConfig::default();
        let paths = vec!["README.md".into()];
        let PolicyOutcome::ExtraRules(rules) = evaluate(&c, &meta(&paths, true, "update")) else {
            panic!()
        };
        assert!(rules.iter().any(|f| f.rule == "PROTECTED_PATH_ONLY"));
        assert!(rules.iter().any(|f| f.rule == "DOCS_ONLY"));
    }

    #[test]
    fn known_contributor_touching_readme_is_fine() {
        let c = RepoConfig::default();
        let paths = vec!["README.md".into()];
        let PolicyOutcome::ExtraRules(rules) = evaluate(&c, &meta(&paths, false, "update")) else {
            panic!()
        };
        assert!(rules.iter().all(|f| f.rule != "PROTECTED_PATH_ONLY"));
    }

    #[test]
    fn mixed_paths_do_not_fire_protected() {
        let c = RepoConfig::default();
        let paths = vec!["README.md".into(), "src/lib.rs".into()];
        let PolicyOutcome::ExtraRules(rules) = evaluate(&c, &meta(&paths, true, "update")) else {
            panic!()
        };
        assert!(rules.iter().all(|f| f.rule != "PROTECTED_PATH_ONLY"));
        assert!(rules.iter().all(|f| f.rule != "DOCS_ONLY"));
    }

    #[test]
    fn empty_body_fires() {
        let c = RepoConfig::default();
        let paths = vec!["src/lib.rs".into()];
        let PolicyOutcome::ExtraRules(rules) = evaluate(&c, &meta(&paths, false, "  ")) else {
            panic!()
        };
        assert!(rules.iter().any(|f| f.rule == "BODY_EMPTY"));
    }

    #[test]
    fn template_ignored_detection() {
        let template = "## What does this change\n## Why\n## Testing";
        assert!(template_ignored(template, "i made it better"));
        assert!(!template_ignored(
            template,
            "## Why\nbecause the old code broke"
        ));
        assert!(!template_ignored("no headers here", "anything"));
    }

    #[test]
    fn template_flows_through_evaluate() {
        let c = RepoConfig::default();
        let paths = vec!["src/lib.rs".into()];
        let pr = PrMeta {
            author: "x",
            changed_paths: &paths,
            is_first_time_contributor: false,
            body: "did stuff",
            template: Some("## Summary\n## Testing"),
        };
        let PolicyOutcome::ExtraRules(rules) = evaluate(&c, &pr) else {
            panic!()
        };
        assert!(rules.iter().any(|f| f.rule == "TEMPLATE_IGNORED"));
    }
}
