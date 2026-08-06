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
    pub title: &'a str,
    pub head_ref: &'a str,
    pub changed_paths: &'a [String],
    pub is_first_time_contributor: bool,
    pub body: &'a str,
    pub additions: u64,
    pub deletions: u64,
    pub commit_count: u64,
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

    rules.extend(shape_rules(pr));
    PolicyOutcome::ExtraRules(rules)
}

/// PR-shape rules mined from real drive-by and agent PRs. All ship dark
/// (zero prior weight) and get priced by the tuner.
pub fn shape_rules(pr: &PrMeta) -> Vec<Fire> {
    let mut out = Vec::new();
    let title = pr.title.trim();
    let body_lower = pr.body.to_lowercase();

    // GitHub's web editor names branches patch-N; committing straight to
    // main/master in a fork is the same drive-by shape.
    let drive_by_branch = matches!(pr.head_ref, "main" | "master")
        || pr
            .head_ref
            .strip_prefix("patch-")
            .map(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
            .unwrap_or(false);
    if drive_by_branch {
        out.push(Fire::hit("BRANCH_DRIVE_BY"));
    }

    // The web editor's default title: "Update README.md".
    let title_update_file = title_is_web_default(title);
    if title_update_file {
        out.push(Fire::hit("TITLE_UPDATE_FILE"));
    }
    if title_update_file && pr.commit_count == 1 {
        out.push(Fire::hit("SINGLE_WEB_COMMIT"));
    }

    // Unfilled template: most of the body is still inside HTML comments.
    let frac = html_comment_fraction(pr.body);
    if frac >= 0.6 {
        out.push(Fire::new("TEMPLATE_UNFILLED", frac));
    }

    // Body claims tests, diff touches nothing test-shaped.
    let claims_tests = [
        "added tests",
        "added a test",
        "add tests",
        "with tests",
        "unit tests included",
        "wrote tests",
        "includes tests",
    ]
    .iter()
    .any(|p| body_lower.contains(p));
    if claims_tests {
        let touches_tests = pr.changed_paths.iter().any(|p| {
            let p = p.to_lowercase();
            p.contains("test") || p.contains("spec") || p.contains("__tests__")
        });
        if !touches_tests {
            out.push(Fire::hit("BODY_DIFF_MISMATCH"));
        }
    }

    // An essay over a trivial diff.
    let body_words = pr.body.split_whitespace().count() as u64;
    let churn = pr.additions + pr.deletions;
    if churn > 0 && churn < 20 && body_words > 150 {
        let ratio = body_words as f64 / churn as f64;
        out.push(Fire::new("BODY_TO_DIFF_RATIO", (ratio / 100.0).min(1.0)));
    }

    // Please-merge begging.
    let begging_hits = [
        "please merge",
        "plz merge",
        "kindly merge",
        "kindly review",
        "please accept",
        "approve my pr",
        "assign me",
        "please review and merge",
        "hacktoberfest",
    ]
    .iter()
    .filter(|p| body_lower.contains(*p) || title.to_lowercase().contains(*p))
    .count();
    if begging_hits > 0 {
        out.push(Fire::new("BEGGING", (begging_hits as f64 / 2.0).min(1.0)));
    }

    // Generated-looking login shapes.
    if username_generated(pr.author) {
        out.push(Fire::hit("USERNAME_PATTERN"));
    }

    out
}

fn title_is_web_default(title: &str) -> bool {
    let mut words = title.split_whitespace();
    let (Some(verb), Some(file), None) = (words.next(), words.next(), words.next()) else {
        return false;
    };
    let verb_ok = ["update", "create", "add", "delete"].contains(&verb.to_lowercase().as_str());
    let file_lower = file.to_lowercase();
    let file_ok = [".md", ".txt", ".rst"]
        .iter()
        .any(|ext| file_lower.ends_with(ext));
    verb_ok && file_ok
}

/// Fraction of body characters inside `<!-- -->` comments.
fn html_comment_fraction(body: &str) -> f64 {
    let total = body.chars().count();
    if total == 0 {
        return 0.0;
    }
    let mut inside = 0usize;
    let mut rest = body;
    while let Some(start) = rest.find("<!--") {
        let after = &rest[start..];
        match after.find("-->") {
            Some(end) => {
                inside += after[..end + 3].chars().count();
                rest = &after[end + 3..];
            }
            None => {
                inside += after.chars().count();
                break;
            }
        }
    }
    inside as f64 / total as f64
}

fn username_generated(login: &str) -> bool {
    let digits = login.bytes().filter(|b| b.is_ascii_digit()).count();
    if login.is_empty() {
        return false;
    }
    let trailing = login
        .bytes()
        .rev()
        .take_while(|b| b.is_ascii_digit())
        .count();
    trailing >= 4 || digits as f64 / login.len() as f64 > 0.4
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
            title: "fix: a normal change",
            head_ref: "feature/my-fix",
            changed_paths: paths,
            is_first_time_contributor: first_time,
            body,
            additions: 10,
            deletions: 2,
            commit_count: 2,
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
            template: Some("## Summary\n## Testing"),
            ..meta(&paths, false, "did stuff")
        };
        let PolicyOutcome::ExtraRules(rules) = evaluate(&c, &pr) else {
            panic!()
        };
        assert!(rules.iter().any(|f| f.rule == "TEMPLATE_IGNORED"));
    }

    fn fires(pr: &PrMeta) -> Vec<String> {
        shape_rules(pr).into_iter().map(|f| f.rule).collect()
    }

    #[test]
    fn drive_by_branch_shapes() {
        let paths = vec!["README.md".into()];
        for branch in ["patch-1", "patch-42", "main", "master"] {
            let pr = PrMeta {
                head_ref: branch,
                ..meta(&paths, true, "x")
            };
            assert!(fires(&pr).contains(&"BRANCH_DRIVE_BY".into()), "{branch}");
        }
        for branch in ["feature/patch-tool", "patch-", "patchwork", "fix-123"] {
            let pr = PrMeta {
                head_ref: branch,
                ..meta(&paths, true, "x")
            };
            assert!(!fires(&pr).contains(&"BRANCH_DRIVE_BY".into()), "{branch}");
        }
    }

    #[test]
    fn web_default_title_and_single_commit() {
        let paths = vec!["README.md".into()];
        let pr = PrMeta {
            title: "Update README.md",
            commit_count: 1,
            ..meta(&paths, true, "x")
        };
        let f = fires(&pr);
        assert!(f.contains(&"TITLE_UPDATE_FILE".into()));
        assert!(f.contains(&"SINGLE_WEB_COMMIT".into()));
        // Same title, multi-commit: no SINGLE_WEB_COMMIT.
        let pr = PrMeta {
            title: "Update README.md",
            commit_count: 3,
            ..meta(&paths, true, "x")
        };
        assert!(!fires(&pr).contains(&"SINGLE_WEB_COMMIT".into()));
        // Real titles do not fire.
        for t in [
            "fix: update parser",
            "Update the docs for v2",
            "Add feature X",
        ] {
            let pr = PrMeta {
                title: t,
                ..meta(&paths, true, "x")
            };
            assert!(!fires(&pr).contains(&"TITLE_UPDATE_FILE".into()), "{t}");
        }
    }

    #[test]
    fn unfilled_template_by_comment_fraction() {
        let paths = vec!["README.md".into()];
        let unfilled = "<!--\nThank you for your pull request. Please provide \
                        a description and note the Certificate of Origin \
                        below.\n-->\n<!--\nmore template text here that the \
                        author never replaced with anything at all\n-->";
        let pr = PrMeta {
            ..meta(&paths, true, unfilled)
        };
        assert!(fires(&pr).contains(&"TEMPLATE_UNFILLED".into()));
        let filled = "I fixed the retry bug.\n<!-- template note -->\nThe \
                      loop now backs off exponentially and gives up after \
                      five attempts, with a regression test for the 403 case.";
        let pr = PrMeta {
            ..meta(&paths, true, filled)
        };
        assert!(!fires(&pr).contains(&"TEMPLATE_UNFILLED".into()));
    }

    #[test]
    fn body_diff_mismatch_needs_claim_without_test_files() {
        let src_only = vec!["src/lib.rs".into()];
        let with_tests = vec!["src/lib.rs".into(), "tests/regress.rs".into()];
        let pr = PrMeta {
            ..meta(
                &src_only,
                false,
                "Refactors the parser. Added tests for edge cases.",
            )
        };
        assert!(fires(&pr).contains(&"BODY_DIFF_MISMATCH".into()));
        let pr = PrMeta {
            ..meta(
                &with_tests,
                false,
                "Refactors the parser. Added tests for edge cases.",
            )
        };
        assert!(!fires(&pr).contains(&"BODY_DIFF_MISMATCH".into()));
        let pr = PrMeta {
            ..meta(&src_only, false, "Refactors the parser.")
        };
        assert!(!fires(&pr).contains(&"BODY_DIFF_MISMATCH".into()));
    }

    #[test]
    fn essay_over_trivial_diff() {
        let paths = vec!["src/lib.rs".into()];
        let essay = "word ".repeat(300);
        let pr = PrMeta {
            additions: 2,
            deletions: 0,
            ..meta(&paths, false, &essay)
        };
        let rules = shape_rules(&pr);
        let r = rules.iter().find(|f| f.rule == "BODY_TO_DIFF_RATIO");
        assert!(r.is_some_and(|f| f.value > 0.9));
        // Big diff: essay is proportionate, no fire.
        let pr = PrMeta {
            additions: 500,
            ..meta(&paths, false, &essay)
        };
        assert!(!fires(&pr).contains(&"BODY_TO_DIFF_RATIO".into()));
    }

    #[test]
    fn begging_lexicon() {
        let paths = vec!["README.md".into()];
        let pr = PrMeta {
            ..meta(
                &paths,
                true,
                "Please merge this. Kindly review. hacktoberfest",
            )
        };
        let rules = shape_rules(&pr);
        let b = rules.iter().find(|f| f.rule == "BEGGING").unwrap();
        assert_eq!(b.value, 1.0);
        let pr = PrMeta {
            ..meta(&paths, true, "This change fixes the null check.")
        };
        assert!(!fires(&pr).contains(&"BEGGING".into()));
    }

    #[test]
    fn generated_usernames() {
        let paths = vec!["README.md".into()];
        for name in ["user48291734", "a1b2c3d4e5", "dev20260806"] {
            let pr = PrMeta {
                author: name,
                ..meta(&paths, true, "x")
            };
            assert!(fires(&pr).contains(&"USERNAME_PATTERN".into()), "{name}");
        }
        for name in ["musvaage", "torvalds", "agent47", "k9dev"] {
            let pr = PrMeta {
                author: name,
                ..meta(&paths, true, "x")
            };
            assert!(!fires(&pr).contains(&"USERNAME_PATTERN".into()), "{name}");
        }
    }

    #[test]
    fn html_comment_fraction_bounds() {
        assert_eq!(html_comment_fraction(""), 0.0);
        assert_eq!(html_comment_fraction("no comments here"), 0.0);
        assert!(html_comment_fraction("<!-- everything -->") > 0.99);
        // Unterminated comment counts to the end.
        assert!(html_comment_fraction("x<!-- never closed") > 0.9);
    }
}
