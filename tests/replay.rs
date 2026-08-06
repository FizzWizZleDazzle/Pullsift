//! Replay acceptance tests: the three reference cases, driven by real
//! fixtures captured from the GitHub API (see fixtures/). These must pass
//! before any live install.
//!
//! 1. express README flood: near-identical drive-by PRs cluster and escalate.
//! 2. linguist#8074 author: a GitHub-flagged, opaque account scores Hold+
//!    on arrival with no repo-local history.
//! 3. torvalds/linux: a mirror repo closes every PR by policy, unscored.

use chrono::{DateTime, TimeZone, Utc};
use slopcatcher::actions::PlannedAction;
use slopcatcher::cluster::ClusterStore;
use slopcatcher::config::{Archetype, RepoConfig};
use slopcatcher::dossier::{parse_dossier, DossierFacts};
use slopcatcher::engine::{Tier, Weights};
use slopcatcher::fit::{auc, Example};
use slopcatcher::pipeline::{process, Outcome, ScoreInputs};
use slopcatcher::webhook::PrEvent;

fn fixture(path: &str) -> String {
    let p = format!("{}/fixtures/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("fixture {p}: {e}"))
}

#[derive(serde::Deserialize)]
struct PrFixture {
    number: u64,
    title: String,
    body: Option<String>,
    user: String,
    additions: u64,
    changed_files: u64,
    author_association: String,
}

fn load_pr(dir: &str, n: u64) -> (PrEvent, String) {
    let meta: PrFixture = serde_json::from_str(&fixture(&format!("{dir}/{n}.json"))).unwrap();
    let diff = fixture(&format!("{dir}/{n}.diff"));
    let ev = PrEvent {
        action: "opened".into(),
        repo: "expressjs/express".into(),
        number: meta.number,
        author: meta.user,
        title: meta.title,
        body: meta.body.unwrap_or_default(),
        additions: meta.additions,
        changed_files: meta.changed_files,
        author_association: meta.author_association,
        head_is_fork: true,
        node_id: format!("PR_{n}"),
    };
    (ev, diff)
}

fn changed_paths(diff: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            out.push(rest.to_string());
        }
    }
    out
}

fn live_config() -> RepoConfig {
    RepoConfig {
        dry_run: false,
        ..Default::default()
    }
}

fn t(minute: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(1_750_000_000 + minute * 60, 0).unwrap()
}

fn score(
    cfg: &RepoConfig,
    ev: &PrEvent,
    diff: &str,
    dossier: DossierFacts,
    clusters: &mut ClusterStore,
    minute: i64,
) -> Outcome {
    let inputs = ScoreInputs {
        config: cfg,
        event: ev,
        diff,
        changed_paths: changed_paths(diff),
        commit_emails: vec![],
        commit_messages: vec![],
        dossier,
        pr_labels: vec![],
        template: None,
    };
    process(
        &inputs,
        &Weights::default_table(),
        clusters,
        "replay-salt",
        t(minute),
    )
}

const WAVE: &[u64] = &[7352, 7319, 7279, 7028];
const HAM: &[u64] = &[7305, 7316, 7345, 7353, 7366, 7377];

#[test]
fn express_wave_clusters_and_escalates() {
    let cfg = live_config();
    let mut clusters = ClusterStore::new(0.5);
    let mut tiers = Vec::new();
    for (i, &n) in WAVE.iter().enumerate() {
        let (ev, diff) = load_pr("express_wave", n);
        let out = score(
            &cfg,
            &ev,
            &diff,
            DossierFacts::default(),
            &mut clusters,
            i as i64 * 3,
        );
        let Outcome::Scored { verdict, .. } = out else {
            panic!("wave PR {n} must be scored")
        };
        tiers.push((n, verdict.tier, verdict.probability));
    }
    // The wave escalates: by the last arrival the tier is Hold or Close.
    let last = tiers.last().unwrap();
    assert!(
        last.1 >= Tier::Hold,
        "wave must escalate to Hold+, got {tiers:?}"
    );
}

#[test]
fn express_ham_passes_untouched() {
    let cfg = live_config();
    for &n in HAM {
        // Fresh cluster store per PR: these are independent arrivals.
        let mut clusters = ClusterStore::new(0.5);
        let (ev, diff) = load_pr("express_ham", n);
        let out = score(&cfg, &ev, &diff, DossierFacts::default(), &mut clusters, 0);
        let (verdict, planned) = match out {
            // Trusted bots (dependabot) exit before scoring: a pass.
            Outcome::Exempt => continue,
            Outcome::Scored { verdict, planned } => (verdict, planned),
            other => panic!("ham PR {n}: unexpected {other:?}"),
        };
        assert!(
            verdict.tier <= Tier::Label,
            "merged PR {n} must never exceed Label, got {:?} at p={:.3}",
            verdict.tier,
            verdict.probability
        );
        assert!(!matches!(
            planned,
            PlannedAction::Close { .. } | PlannedAction::Hold { .. }
        ));
    }
}

#[test]
fn default_weights_separate_wave_from_ham() {
    // AUC of the shipped weight table over the replayed fires: the wave and
    // the merged PRs must be fully separable.
    let cfg = live_config();
    let weights = Weights::default_table();
    let mut examples = Vec::new();

    let mut clusters = ClusterStore::new(0.5);
    for (i, &n) in WAVE.iter().enumerate() {
        let (ev, diff) = load_pr("express_wave", n);
        let out = score(
            &cfg,
            &ev,
            &diff,
            DossierFacts::default(),
            &mut clusters,
            i as i64 * 3,
        );
        let Outcome::Scored { verdict, .. } = out else {
            panic!()
        };
        let fires = verdict
            .evidence
            .iter()
            .map(|e| slopcatcher::engine::Fire::new(&e.rule, e.value))
            .collect();
        examples.push(Example::new(fires, true));
    }
    for &n in HAM {
        let mut clusters = ClusterStore::new(0.5);
        let (ev, diff) = load_pr("express_ham", n);
        let out = score(&cfg, &ev, &diff, DossierFacts::default(), &mut clusters, 0);
        let verdict = match out {
            Outcome::Exempt => continue, // trusted bot, no score to compare
            Outcome::Scored { verdict, .. } => verdict,
            other => panic!("ham PR {n}: unexpected {other:?}"),
        };
        let fires = verdict
            .evidence
            .iter()
            .map(|e| slopcatcher::engine::Fire::new(&e.rule, e.value))
            .collect();
        examples.push(Example::new(fires, false));
    }

    let a = auc(&weights, &examples);
    assert!(a > 0.95, "default weights AUC on replay corpus: {a}");
}

#[test]
fn linguist_author_scores_hold_on_arrival() {
    // The dossier fixture is what the bot actually sees for this account:
    // GitHub returns an empty PR connection and blocks search ("users cannot
    // be searched"), which sets search_blocked.
    let resp: serde_json::Value = serde_json::from_str(&fixture("musvaage_dossier.json")).unwrap();
    let mut facts = parse_dossier("musvaage", &resp, t(0));
    facts.search_blocked = true;

    assert_eq!(facts.prior.len(), 0, "flagged account: history is hidden");
    assert!(!facts.has_bio);

    let cfg = live_config();
    let ev = PrEvent {
        action: "opened".into(),
        repo: "github-linguist/linguist".into(),
        number: 8074,
        author: "musvaage".into(),
        title: "add claire support".into(),
        body: fixture_linguist_body(),
        additions: 120,
        changed_files: 4,
        author_association: "NONE".into(),
        head_is_fork: true,
        node_id: "PR_8074".into(),
    };
    let diff = "+Claire:\n+  type: programming\n+  color: \"#009688\"\n\
                +  extensions:\n+    - \".cl\"\n+  tm_scope: none\n\
                +  ace_mode: text\n+  language_id: 890123456\n";
    let mut clusters = ClusterStore::new(0.5);
    let inputs = ScoreInputs {
        config: &cfg,
        event: &ev,
        diff,
        changed_paths: vec!["lib/linguist/languages.yml".into()],
        commit_emails: vec![],
        commit_messages: vec![],
        dossier: facts,
        pr_labels: vec![],
        template: None,
    };
    let out = process(&inputs, &Weights::default_table(), &mut clusters, "s", t(0));
    let Outcome::Scored { verdict, .. } = out else {
        panic!()
    };
    assert!(
        verdict.tier >= Tier::Hold,
        "flagged author must arrive at Hold+, got {:?} at p={:.3}",
        verdict.tier,
        verdict.probability
    );
    assert!(verdict.evidence.iter().any(|e| e.rule == "GH_FLAGGED"));
}

fn fixture_linguist_body() -> String {
    "## Description\r\n\r\nAdd support for the **Claire** language; Claire 3.x, \
     its fork as XL Claire, and Claire 4.\r\n\r\nCommon Lisp and Cool use the \
     same .cl file extension.\r\n"
        .to_string()
}

#[test]
fn linux_mirror_closes_every_pr_by_policy() {
    let cfg = RepoConfig {
        dry_run: false,
        archetype: Some(Archetype::MirrorNoPrs),
        contribution_channel: Some("the kernel mailing lists (see Documentation/process)".into()),
        ..Default::default()
    };
    // Real shapes seen on torvalds/linux: meme PRs and earnest mistakes alike.
    let samples = [
        ("Update README", "+Linux is the best os\n"),
        (
            "fix typo in comment",
            "+// fixed a typo here in the scheduler\n",
        ),
        ("Add my name to credits", "+John Doe <john@example.com>\n"),
    ];
    for (i, (title, diff)) in samples.iter().enumerate() {
        let ev = PrEvent {
            action: "opened".into(),
            repo: "torvalds/linux".into(),
            number: i as u64 + 1,
            author: format!("user{i}"),
            title: title.to_string(),
            body: "please merge".into(),
            additions: 1,
            changed_files: 1,
            author_association: "NONE".into(),
            head_is_fork: true,
            node_id: format!("PR_{i}"),
        };
        let mut clusters = ClusterStore::new(0.5);
        let inputs = ScoreInputs {
            config: &cfg,
            event: &ev,
            diff,
            changed_paths: vec!["README".into()],
            commit_emails: vec![],
            commit_messages: vec![],
            dossier: DossierFacts::default(),
            pr_labels: vec![],
            template: None,
        };
        let out = process(&inputs, &Weights::default_table(), &mut clusters, "s", t(0));
        let Outcome::PolicyClose { comment } = out else {
            panic!("mirror repo must close '{title}' by policy")
        };
        assert!(comment.contains("kernel mailing lists"));
        assert!(comment.contains("not a judgment"));
    }
}

#[test]
fn dry_run_on_the_wave_only_annotates() {
    // Same wave, dry-run config (the install default): nothing stronger
    // than a label may be planned.
    let cfg = RepoConfig::default();
    assert!(cfg.dry_run);
    let mut clusters = ClusterStore::new(0.5);
    for (i, &n) in WAVE.iter().enumerate() {
        let (ev, diff) = load_pr("express_wave", n);
        let out = score(
            &cfg,
            &ev,
            &diff,
            DossierFacts::default(),
            &mut clusters,
            i as i64 * 3,
        );
        let Outcome::Scored { planned, .. } = out else {
            panic!()
        };
        assert!(
            matches!(planned, PlannedAction::None | PlannedAction::Label { .. }),
            "dry run planned {planned:?}"
        );
    }
}
