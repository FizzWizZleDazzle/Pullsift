//! The scoring pipeline: everything between a parsed webhook event and a
//! planned action. Pure given its inputs; the service assembles the inputs
//! (diff, files, commits, dossier) and executes the plan.

use crate::actions::{self, PlannedAction};
use crate::challenge;
use crate::cluster::{cluster_rules, ClusterStore, PrSignature};
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
    pub dossier: DossierFacts,
    pub pr_labels: Vec<String>,
    pub template: Option<&'a str>,
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
    let meta = PrMeta {
        author: &ev.author,
        changed_paths: &inputs.changed_paths,
        is_first_time_contributor: ev.author_association == "FIRST_TIME_CONTRIBUTOR"
            || ev.author_association == "NONE",
        body: &ev.body,
        template: inputs.template,
    };
    let mut fires: Vec<Fire> = match policy::evaluate(cfg, &meta) {
        PolicyOutcome::CloseByPolicy(comment) => return Outcome::PolicyClose { comment },
        PolicyOutcome::ExtraRules(rules) => rules,
    };

    // Lane A: signatures and clustering.
    let prose = format!("{}\n{}", ev.title, ev.body);
    let style = stylometry::analyze(&prose);
    let sig = PrSignature {
        repo: ev.repo.clone(),
        pr_number: ev.number,
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

    // Score under this repo's thresholds.
    let mut scoped = weights.clone();
    scoped.thresholds = cfg.thresholds(weights.thresholds);
    let verdict = scoped.score(&fires);

    let challenge_comment = if cfg.challenge && verdict.tier == Tier::Hold {
        let canary = challenge::canary_token(&ev.repo, ev.number, canary_salt);
        Some(challenge::challenge_comment(&canary))
    } else {
        None
    };

    let planned = actions::plan(&verdict, challenge_comment, cfg.dry_run);
    Outcome::Scored { verdict, planned }
}

#[cfg(test)]
mod tests {
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
            changed_files: 1,
            author_association: "FIRST_TIME_CONTRIBUTOR".into(),
            head_is_fork: true,
            node_id: "PR_x".into(),
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
            dossier: DossierFacts::default(),
            pr_labels: vec![],
            template: None,
        }
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
    fn agent_pr_with_damning_dossier_closes() {
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
        assert_eq!(verdict.tier, Tier::Close, "p={:.4}", verdict.probability);
        assert!(matches!(planned, PlannedAction::Close { .. }));
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
            PlannedAction::Label { .. } | PlannedAction::None
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
