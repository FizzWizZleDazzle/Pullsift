//! The scoring pipeline: everything between a parsed webhook event and a
//! planned action. Pure given its inputs; the service assembles the inputs
//! (diff, files, commits, dossier) and executes the plan.

use crate::actions::{self, PlannedAction};
use crate::challenge;
use crate::cluster::{ClusterStore, PrSignature, cluster_rules};
use crate::config::RepoConfig;
use crate::diffsig;
use crate::dossier::DossierFacts;
use crate::engine::{Fire, Tier, Verdict, Weights};
use crate::policy::{self, PolicyOutcome, PrMeta};
use crate::stylometry;
use crate::textsig;
use crate::webhook::PrEvent;
use chrono::{DateTime, Utc};

/// Everything the pipeline needs, already fetched.
pub struct ScoreInputs<'a> {
    pub config: &'a RepoConfig,
    pub event: &'a PrEvent,
    pub diff: &'a str,
    pub changed_paths: Vec<String>,
    pub commit_emails: Vec<String>,
    pub commit_messages: Vec<String>,
    /// Author timestamps of the PR's commits, in whatever order the API
    /// returned them. Empty when unavailable; the rate rule then abstains.
    pub commit_times: Vec<DateTime<Utc>>,
    pub dossier: DossierFacts,
    pub pr_labels: Vec<String>,
    pub template: Option<&'a str>,
    /// Probability from an external/local AI-text detector, when one is
    /// configured (DETECTOR_URL). Fail-open: absence fires nothing.
    pub detector_score: Option<f64>,
}

#[derive(Debug)]
pub enum Outcome {
    /// Exempt user or label: nothing recorded beyond the event.
    Exempt,
    /// Lane C archetype close.
    PolicyClose { comment: String },
    /// Scored: the verdict and the planned action.
    Scored {
        verdict: Verdict,
        planned: PlannedAction,
    },
}

pub fn process(
    inputs: &ScoreInputs,
    weights: &Weights,
    clusters: &mut ClusterStore,
    canary_salt: &str,
    now: DateTime<Utc>,
) -> Outcome {
    let ev = inputs.event;
    let cfg = inputs.config;

    if cfg.is_exempt(&ev.author, &inputs.pr_labels) {
        return Outcome::Exempt;
    }

    // Lane C first: the cheapest lane wins outright.
    let commit_count = if ev.commit_count > 0 {
        ev.commit_count
    } else {
        inputs.commit_messages.len() as u64
    };
    let meta = PrMeta {
        author: &ev.author,
        title: &ev.title,
        head_ref: &ev.head_ref,
        changed_paths: &inputs.changed_paths,
        is_first_time_contributor: ev.author_association == "FIRST_TIME_CONTRIBUTOR"
            || ev.author_association == "NONE",
        body: &ev.body,
        additions: ev.additions,
        deletions: ev.deletions,
        commit_count,
        template: inputs.template,
    };
    let mut fires: Vec<Fire> = match policy::evaluate(cfg, &meta) {
        PolicyOutcome::CloseByPolicy(comment) => return Outcome::PolicyClose { comment },
        PolicyOutcome::ExtraRules(rules) => rules,
    };
    if diffsig::whitespace_only(inputs.diff) {
        fires.push(Fire::hit("WHITESPACE_ONLY"));
    }
    if let Some(density) = diffsig::comment_density(inputs.diff)
        && density >= 0.35
    {
        fires.push(Fire::new("COMMENT_HEAVY", density.min(1.0)));
    }
    if grounding_miss(&ev.body, inputs.diff, &inputs.changed_paths) {
        fires.push(Fire::hit("GROUNDING_MISS"));
    }
    if let Some(rate) = authoring_rate(&inputs.commit_times, ev.additions) {
        fires.push(Fire::new("AUTHORING_RATE", rate));
    }
    fires.extend(crate::codeslop::rules(inputs.diff));
    fires.extend(crate::codestruct::rules(inputs.diff));

    // Lane A: signatures and clustering.
    let prose = format!("{}\n{}", ev.title, ev.body);
    let style = stylometry::analyze(&prose);
    let sig = PrSignature {
        repo: ev.repo.clone(),
        pr_number: ev.number,
        author: ev.author.clone(),
        arrived: now,
        diff_sim: diffsig::simhash(inputs.diff),
        text_min: textsig::minhash(&prose),
        pathset: diffsig::pathset_hash(&inputs.changed_paths),
        style: style.clone(),
    };
    let view = clusters.insert(sig);
    fires.extend(cluster_rules(&view));

    // Lane B: dossier plus current-PR markers and stylometry.
    let mut dossier = inputs.dossier.clone();
    let (email, trailer, footer) =
        crate::dossier::scan_markers(&inputs.commit_emails, &inputs.commit_messages, &ev.body);
    dossier.agent_email |= email;
    dossier.agent_trailer |= trailer;
    dossier.generation_footer |= footer;
    dossier.additions = ev.additions;
    fires.extend(dossier.rules());
    fires.extend(style.rules());

    // Learned token model over the prose; a single rule the fit prices.
    static TOKEN_TABLE: std::sync::OnceLock<crate::tokenscore::TokenTable> =
        std::sync::OnceLock::new();
    let table = TOKEN_TABLE.get_or_init(crate::tokenscore::TokenTable::embedded);
    if let Some(p) = table.score(&prose) {
        fires.push(Fire::new("BODY_TOKEN_SCORE", p));
    }

    if let Some(p) = inputs.detector_score {
        fires.push(Fire::new("DETECTOR_SCORE", p));
    }

    // The repo's AI policy: taste, applied on top of the fitted table.
    let has_marker = dossier.agent_email || dossier.agent_trailer || dossier.generation_footer;
    let mut scoped = weights.clone();
    match cfg.ai_policy {
        crate::config::AiPolicy::Welcome => {
            for rule in crate::config::RepoConfig::AI_STYLE_RULES {
                if let Some(w) = scoped.rules.get_mut(*rule) {
                    *w = 0.0;
                }
            }
        }
        crate::config::AiPolicy::Neutral => {}
        crate::config::AiPolicy::Disclose => {
            if !has_marker && inputs.detector_score.is_some_and(|p| p >= 0.7) {
                fires.push(Fire::hit("UNDISCLOSED_AI"));
            }
        }
        crate::config::AiPolicy::Forbid => {
            if has_marker {
                fires.push(Fire::hit("AI_FORBIDDEN"));
            }
        }
    }

    // Score under this repo's thresholds.
    scoped.thresholds = cfg.thresholds(weights.thresholds);
    let verdict = scoped.score(&fires);

    let challenge_comment = if cfg.challenge && verdict.tier == Tier::Hold {
        let canary = challenge::canary_token(&ev.repo, ev.number, canary_salt);
        Some(challenge::challenge_comment(&canary))
    } else {
        None
    };

    let planned = actions::plan(&verdict, challenge_comment, cfg.dry_run, cfg.score_comments);
    Outcome::Scored { verdict, planned }
}

/// Lines written per hour, across the span of the pull request's own
/// commits. Thousands of lines an hour is a machine's cadence, not a
/// person's, and that is all this measures: it says how the code was
/// produced, not whether it is any good. Plenty of generated work is
/// worth merging, so this carries one fitted weight among many and never
/// reaches a tier alone.
///
/// Needs at least two commits to have a span at all. A single commit
/// carries no information about how long the work took, and guessing from
/// the gap to the pull request would punish people who push promptly.
fn authoring_rate(commit_times: &[DateTime<Utc>], additions: u64) -> Option<f64> {
    const MIN_ADDITIONS: u64 = 200;
    /// Lines per hour at which the rule saturates.
    const FAST: f64 = 4_000.0;

    if commit_times.len() < 2 || additions < MIN_ADDITIONS {
        return None;
    }
    let first = commit_times.iter().min()?;
    let last = commit_times.iter().max()?;
    let hours = (*last - *first).num_seconds() as f64 / 3600.0;
    // Commits sharing a timestamp give no span to divide by; treat the
    // whole batch as one minute of work rather than dividing by zero.
    let hours = hours.max(1.0 / 60.0);
    let rate = additions as f64 / hours;
    (rate >= 500.0).then(|| (rate.ln() / FAST.ln()).clamp(0.0, 1.0))
}

/// The curl tell: a body that name-drops code the PR never touches.
/// Collect backticked identifier-shaped mentions (calls, paths, symbols)
/// from the body; fire only when there are at least two and none of them
/// appears in the diff or the changed paths. Referencing some unchanged
/// code is normal engineering prose; referencing only phantoms is
/// fabrication.
fn grounding_miss(body: &str, diff: &str, changed_paths: &[String]) -> bool {
    let mut mentions: Vec<String> = Vec::new();
    for (i, span) in body.split('`').enumerate() {
        // Odd indexes are inside backticks.
        if i % 2 == 0 {
            continue;
        }
        let s = span.trim();
        if s.len() < 3 || s.len() > 80 || s.contains(char::is_whitespace) {
            continue;
        }
        let identifier_shaped =
            s.contains('(') || s.contains('/') || s.contains('_') || s.contains("::");
        if identifier_shaped {
            let core = s.trim_end_matches("()").trim_matches('`').to_string();
            if core.len() >= 3 {
                mentions.push(core);
            }
        }
    }
    if mentions.len() < 2 {
        return false;
    }
    let paths_lower: Vec<String> = changed_paths.iter().map(|p| p.to_lowercase()).collect();
    let diff_lower = diff.to_lowercase();
    !mentions.iter().any(|m| {
        let m = m.to_lowercase();
        diff_lower.contains(&m) || paths_lower.iter().any(|p| p.contains(&m) || m.contains(p))
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn authoring_rate_reads_machine_cadence_not_size() {
        use chrono::TimeZone;
        let at = |mins: i64| Utc.timestamp_opt(1_750_000_000 + mins * 60, 0).unwrap();

        // Three thousand lines across eleven minutes.
        assert!(super::authoring_rate(&[at(0), at(4), at(11)], 3_000).unwrap() > 0.8);
        // The same change over two days is a person working.
        assert!(super::authoring_rate(&[at(0), at(2880)], 3_000).is_none());
        // A big change alone says nothing without a span to divide by.
        assert!(super::authoring_rate(&[at(0)], 9_000).is_none());
        // Fast, but too small to mean anything.
        assert!(super::authoring_rate(&[at(0), at(1)], 30).is_none());
        // Commits sharing one timestamp must not divide by zero.
        assert!(super::authoring_rate(&[at(5), at(5)], 2_000).unwrap() > 0.0);
    }

    #[test]
    fn enormous_diffs_fire_unless_the_bulk_is_unauthored() {
        use crate::policy::{PolicyOutcome, PrMeta, evaluate};
        let meta = |additions: u64, paths: Vec<String>| PrMeta {
            author: "someone",
            title: "big change",
            head_ref: "feature",
            changed_paths: Box::leak(paths.into_boxed_slice()),
            is_first_time_contributor: false,
            body: "This adds the thing.",
            additions,
            deletions: 0,
            commit_count: 1,
            template: None,
        };
        let fired = |m: PrMeta| match evaluate(&RepoConfig::default(), &m) {
            PolicyOutcome::ExtraRules(r) => r.iter().any(|f| f.rule == "DIFF_ENORMOUS"),
            _ => false,
        };
        assert!(fired(meta(40_000, vec!["src/everything.rs".into()])));
        assert!(!fired(meta(400, vec!["src/small.rs".into()])));
        // A vendored tree is legitimately enormous.
        assert!(!fired(meta(
            40_000,
            vec!["vendor/lib.go".into(), "go.sum".into(), "src/one.go".into()]
        )));
    }

    #[test]
    fn grounding_miss_fires_only_on_wholesale_fabrication() {
        let diff = "+fn unquote(s: &str) -> &str { s }\n";
        let paths = vec!["src/parser.rs".to_string()];
        // Fabricated symbols nowhere in the diff or paths.
        assert!(super::grounding_miss(
            "Fixes an overflow in `curl_inet_ntop()` and hardens `net/resolve_host()`.",
            diff,
            &paths
        ));
        // One real mention grounds the body.
        assert!(!super::grounding_miss(
            "Adds `unquote()` and simplifies `strip_wrapping()`.",
            diff,
            &paths
        ));
        // A single identifier mention is not enough to judge.
        assert!(!super::grounding_miss(
            "See `phantom_fn()` for context.",
            diff,
            &paths
        ));
        // Path mentions ground against changed files.
        assert!(!super::grounding_miss(
            "Touches `src/parser.rs` and `some/other_thing()`.",
            diff,
            &paths
        ));
    }

    use super::*;
    use crate::config::Archetype;
    use chrono::TimeZone;

    fn now(minute: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_750_000_000 + minute * 60, 0).unwrap()
    }

    fn event(author: &str, n: u64) -> PrEvent {
        PrEvent {
            action: "opened".into(),
            repo: "octo/repo".into(),
            number: n,
            author: author.into(),
            title: "Update README.md".into(),
            body: "This PR improves the README documentation and fixes a typo \
                   in the installation instructions to help new users."
                .into(),
            additions: 2,
            deletions: 0,
            changed_files: 1,
            commit_count: 1,
            author_association: "FIRST_TIME_CONTRIBUTOR".into(),
            head_is_fork: true,
            head_ref: "my-branch".into(),
            node_id: "PR_x".into(),
            labels: vec![],
        }
    }

    const README_DIFF: &str = "\
+Express is a fast, unopinionated, minimalist web framework for node.
+It makes building web applications really easy for everyone involved.
";

    fn inputs<'a>(cfg: &'a RepoConfig, ev: &'a PrEvent) -> ScoreInputs<'a> {
        ScoreInputs {
            config: cfg,
            event: ev,
            diff: README_DIFF,
            changed_paths: vec!["README.md".into()],
            commit_emails: vec!["someone@example.com".into()],
            commit_messages: vec!["update readme".into()],
            commit_times: vec![],
            dossier: DossierFacts::default(),
            pr_labels: vec![],
            template: None,
            detector_score: None,
        }
    }

    #[test]
    fn ai_policy_changes_how_markers_score() {
        let ev = event("someuser", 7);
        let with_marker = |cfg: &RepoConfig| {
            let mut inp = inputs(cfg, &ev);
            inp.dossier = DossierFacts {
                agent_trailer: true,
                ..Default::default()
            };
            let mut clusters = ClusterStore::new(0.5);
            match process(&inp, &Weights::default_table(), &mut clusters, "s", now(0)) {
                Outcome::Scored { verdict, .. } => verdict,
                _ => panic!("expected scored"),
            }
        };
        let forbid = with_marker(&RepoConfig {
            ai_policy: crate::config::AiPolicy::Forbid,
            ..Default::default()
        });
        assert!(forbid.evidence.iter().any(|e| e.rule == "AI_FORBIDDEN"));
        let welcome = with_marker(&RepoConfig {
            ai_policy: crate::config::AiPolicy::Welcome,
            ..Default::default()
        });
        assert!(welcome.evidence.iter().all(|e| e.rule != "AI_FORBIDDEN"));
        let trailer_contribution: f64 = welcome
            .evidence
            .iter()
            .filter(|e| e.rule == "AGENT_TRAILER")
            .map(|e| e.contribution)
            .sum();
        assert_eq!(trailer_contribution, 0.0);
        assert!(forbid.probability > welcome.probability);
    }

    #[test]
    fn undisclosed_ai_fires_only_without_markers() {
        let ev = event("someuser", 8);
        let cfg = RepoConfig {
            ai_policy: crate::config::AiPolicy::Disclose,
            ..Default::default()
        };
        let mut inp = inputs(&cfg, &ev);
        inp.detector_score = Some(0.92);
        let mut clusters = ClusterStore::new(0.5);
        let Outcome::Scored { verdict, .. } =
            process(&inp, &Weights::default_table(), &mut clusters, "s", now(0))
        else {
            panic!()
        };
        assert!(verdict.evidence.iter().any(|e| e.rule == "UNDISCLOSED_AI"));

        // Disclosed (marker present): no penalty.
        let mut inp2 = inputs(&cfg, &ev);
        inp2.detector_score = Some(0.92);
        inp2.dossier = DossierFacts {
            generation_footer: true,
            ..Default::default()
        };
        let mut clusters2 = ClusterStore::new(0.5);
        let Outcome::Scored { verdict, .. } = process(
            &inp2,
            &Weights::default_table(),
            &mut clusters2,
            "s",
            now(1),
        ) else {
            panic!()
        };
        assert!(verdict.evidence.iter().all(|e| e.rule != "UNDISCLOSED_AI"));
    }

    #[test]
    fn exempt_user_is_untouched() {
        let cfg = RepoConfig::default();
        let ev = event("dependabot[bot]", 1);
        let mut clusters = ClusterStore::new(0.5);
        let out = process(
            &inputs(&cfg, &ev),
            &Weights::default_table(),
            &mut clusters,
            "s",
            now(0),
        );
        assert!(matches!(out, Outcome::Exempt));
    }

    #[test]
    fn override_label_exempts_the_pr() {
        // A maintainer's pullsift-override label must stop all scoring, even
        // for a PR that would otherwise convict.
        let cfg = RepoConfig {
            dry_run: false,
            ..Default::default()
        };
        let ev = event("ghostbot", 1);
        let mut inp = inputs(&cfg, &ev);
        inp.commit_emails.push("noreply@anthropic.com".into());
        inp.pr_labels = vec!["pullsift-override".into()];
        let mut clusters = ClusterStore::new(0.5);
        let out = process(&inp, &Weights::default_table(), &mut clusters, "s", now(0));
        assert!(matches!(out, Outcome::Exempt));
    }

    #[test]
    fn mirror_repo_closes_by_policy_without_scoring() {
        let cfg = RepoConfig {
            archetype: Some(Archetype::MirrorNoPrs),
            contribution_channel: Some("the mailing list".into()),
            ..Default::default()
        };
        let ev = event("anyone", 1);
        let mut clusters = ClusterStore::new(0.5);
        let out = process(
            &inputs(&cfg, &ev),
            &Weights::default_table(),
            &mut clusters,
            "s",
            now(0),
        );
        let Outcome::PolicyClose { comment } = out else {
            panic!("expected policy close")
        };
        assert!(comment.contains("mailing list"));
        assert_eq!(clusters.len(), 0, "no signature recorded for policy closes");
    }

    #[test]
    fn single_innocent_pr_passes_or_labels_only() {
        let cfg = RepoConfig {
            dry_run: false,
            ..Default::default()
        };
        let ev = event("newcomer", 1);
        let mut clusters = ClusterStore::new(0.5);
        let out = process(
            &inputs(&cfg, &ev),
            &Weights::default_table(),
            &mut clusters,
            "s",
            now(0),
        );
        let Outcome::Scored { verdict, .. } = out else {
            panic!()
        };
        assert!(
            verdict.tier <= Tier::Label,
            "one first-timer README PR must never exceed T1, got {:?} at p={:.3}",
            verdict.tier,
            verdict.probability
        );
    }

    #[test]
    fn tutorial_wave_escalates_and_first_pr_did_not() {
        let cfg = RepoConfig {
            dry_run: false,
            ..Default::default()
        };
        let mut clusters = ClusterStore::new(0.5);
        let weights = Weights::default_table();
        let mut tiers = Vec::new();
        for i in 0..40 {
            let ev = event(&format!("student{i}"), i);
            let out = process(
                &inputs(&cfg, &ev),
                &weights,
                &mut clusters,
                "s",
                now(i as i64 * 2),
            );
            let Outcome::Scored { verdict, .. } = out else {
                panic!()
            };
            tiers.push(verdict.tier);
        }
        assert!(tiers[0] <= Tier::Label, "first arrival is innocent");
        assert!(
            *tiers.last().unwrap() >= Tier::Hold,
            "the wave must escalate, got {:?}",
            tiers.last().unwrap()
        );
        assert!(tiers.windows(2).all(|w| w[1] >= w[0] || w[1] >= Tier::Hold));
    }

    #[test]
    fn agent_pr_with_damning_dossier_reaches_enforcement() {
        // Dossier plus provenance markers reaches at least Hold. It does
        // not reach Close on its own: the fitted corpus carries merged
        // agent PRs as ham, so markers and a bad trail are strong but not
        // near-certain evidence; Close is reserved for campaign or
        // challenge-failure signals on top.
        let cfg = RepoConfig {
            dry_run: false,
            ..Default::default()
        };
        let ev = event("ghostbot", 1);
        let mut inp = inputs(&cfg, &ev);
        inp.commit_emails.push("noreply@anthropic.com".into());
        inp.commit_messages
            .push("feat: x\n\nCo-Authored-By: Claude <noreply@anthropic.com>".into());
        inp.dossier = DossierFacts {
            login: "ghostbot".into(),
            prior: (0..10)
                .map(|i| crate::dossier::PriorPr {
                    merged: false,
                    closed_unmerged: true,
                    spam_labeled: i < 3,
                    received_review: true,
                    author_followed_up: false,
                    repo_key: format!("r{i}/x"),
                    title: format!("feat: thing {i}"),
                })
                .collect(),
            restricted_contributions: 10,
            ..Default::default()
        };
        let mut clusters = ClusterStore::new(0.5);
        let out = process(&inp, &Weights::default_table(), &mut clusters, "s", now(0));
        let Outcome::Scored { verdict, planned } = out else {
            panic!()
        };
        assert!(
            verdict.tier >= Tier::Hold,
            "p={:.4}, tier {:?}",
            verdict.probability,
            verdict.tier
        );
        assert!(matches!(
            planned,
            PlannedAction::Hold { .. } | PlannedAction::Close { .. }
        ));
    }

    #[test]
    fn dry_run_never_closes() {
        let cfg = RepoConfig::default(); // dry_run: true
        let ev = event("ghostbot", 1);
        let mut inp = inputs(&cfg, &ev);
        inp.commit_emails.push("noreply@anthropic.com".into());
        inp.dossier.agent_trailer = true;
        inp.dossier.generation_footer = true;
        let mut clusters = ClusterStore::new(0.5);
        let out = process(&inp, &Weights::default_table(), &mut clusters, "s", now(0));
        let Outcome::Scored { planned, .. } = out else {
            panic!()
        };
        assert!(matches!(
            planned,
            PlannedAction::Label { .. } | PlannedAction::Comment { .. } | PlannedAction::None
        ));
    }

    #[test]
    fn hold_carries_a_challenge_when_enabled() {
        // Overrides force the Hold band around a mid probability.
        let cfg = RepoConfig {
            dry_run: false,
            threshold_label: Some(0.01),
            threshold_hold: Some(0.02),
            threshold_close: Some(0.9999),
            ..Default::default()
        };
        let ev = event("suspicious", 1);
        let mut inp = inputs(&cfg, &ev);
        inp.dossier.agent_trailer = true;
        let mut clusters = ClusterStore::new(0.5);
        let out = process(&inp, &Weights::default_table(), &mut clusters, "s", now(0));
        let Outcome::Scored { planned, verdict } = out else {
            panic!()
        };
        assert_eq!(verdict.tier, Tier::Hold);
        let PlannedAction::Hold {
            challenge_comment, ..
        } = planned
        else {
            panic!("expected hold")
        };
        let c = challenge_comment.expect("challenge enabled by default");
        assert!(c.contains("CANARY-"));
    }
}
