//! Offline tuner: replay the mined corpus through the production pipeline,
//! cross-validate a fitted weight table against the incumbent, and write
//! the new table with provenance.
//!
//! Usage: tune [--dry] [corpus-dir]
//!
//! Folds are grouped by author (an author never appears in both train and
//! eval); a second pass groups by repo, which is the number that predicts
//! how the table behaves on an install it has never seen. The fit is
//! anchored to the incumbent table: rules that never fired, or fired too
//! rarely to price, keep their incumbent weight, so rare-but-designed
//! signals (network verdicts, challenge outcomes) are not silently zeroed.
//! Each spam source is also held out in turn, because a table that only
//! recognizes the campaigns it was trained on is the failure this tool
//! exists to catch.

use chrono::{DateTime, Utc};
use pullsift::cluster::ClusterStore;
use pullsift::config::RepoConfig;
use pullsift::dossier::{DossierFacts, parse_dossier, scan_markers};
use pullsift::engine::{Fire, Weights};
use pullsift::fit::{
    Example, FitOptions, auc, auc_of, certifiable, fire_counts, fit, observed_fpr,
    thresholds_at_fpr, thresholds_from_negatives,
};
use pullsift::hashing::fnv1a64;
use pullsift::pipeline::{Outcome, ScoreInputs, process};
use pullsift::webhook::PrEvent;
use serde::Deserialize;
use std::collections::BTreeMap;

const FOLDS: u64 = 5;

#[derive(Deserialize)]
struct Record {
    label: String,
    source: String,
    repo: String,
    number: u64,
    title: String,
    #[serde(default)]
    body: String,
    author: String,
    #[serde(default)]
    head_ref: String,
    #[serde(default)]
    additions: u64,
    #[serde(default)]
    deletions: u64,
    #[serde(default)]
    changed_files: u64,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    commits: Vec<CommitRec>,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default)]
    diff: String,
    #[serde(default)]
    dossier: serde_json::Value,
    #[serde(default)]
    search_blocked: bool,
}

#[derive(Deserialize)]
struct CommitRec {
    #[serde(default)]
    email: String,
    #[serde(default)]
    message: String,
    /// Author timestamp. Absent on records mined before the field existed;
    /// the authoring-rate rule abstains for those rather than guessing.
    #[serde(default)]
    date: String,
}

struct Scored {
    fires: Vec<Fire>,
    is_slop: bool,
    author: String,
    repo: String,
    source: String,
    id: String,
    /// Title + body, for training the token model per fold.
    prose: String,
}

/// Examples with a fold-local token-model fire appended. The replayed
/// evidence is stripped of any BODY_TOKEN_SCORE first (the embedded table
/// may be non-empty on re-runs), so the fold's own table is the only token
/// signal and cross-validation stays leak-free.
fn token_examples(scored: &[&Scored], table: &pullsift::tokenscore::TokenTable) -> Vec<Example> {
    scored
        .iter()
        .map(|s| {
            let mut fires: Vec<Fire> = s
                .fires
                .iter()
                .filter(|f| f.rule != "BODY_TOKEN_SCORE")
                .cloned()
                .collect();
            if let Some(p) = table.score(&s.prose) {
                fires.push(Fire::new("BODY_TOKEN_SCORE", p));
            }
            Example::new(fires, s.is_slop).in_group(&s.author)
        })
        .collect()
}

/// Training examples whose token-model fire is cross-fitted: the table
/// that scores a record was trained on the other inner folds, never on
/// the record itself. Pricing BODY_TOKEN_SCORE on in-sample token scores
/// made the fit treat a weak signal as a strong one, because a table
/// scores the documents it was built from far better than the next ones.
fn crossfit_token_examples(scored: &[&Scored]) -> Vec<Example> {
    let inner = |s: &Scored| fold_of(&format!("inner:{}", s.author));
    let mut out: Vec<Option<Example>> = vec![None; scored.len()];
    for k in 0..FOLDS {
        let table_s: Vec<&Scored> = scored.iter().copied().filter(|s| inner(s) != k).collect();
        let table = train_table(&table_s);
        for (i, s) in scored.iter().enumerate() {
            if inner(s) == k {
                out[i] = token_examples(&[*s], &table).pop();
            }
        }
    }
    out.into_iter()
        .map(|e| e.expect("every record scored"))
        .collect()
}

fn train_table(scored: &[&Scored]) -> pullsift::tokenscore::TokenTable {
    let docs: Vec<(String, bool, String)> = scored
        .iter()
        .map(|s| (s.prose.clone(), s.is_slop, s.repo.clone()))
        .collect();
    pullsift::tokenscore::TokenTable::train(&docs)
}

fn options(incumbent: &Weights) -> FitOptions {
    FitOptions {
        prior: Some(incumbent.clone()),
        ..FitOptions::default()
    }
}

/// One out-of-fold prediction: the example, its probability under the
/// fold's candidate, and the record id.
type OutOfFold = Vec<(Example, f64, String)>;

/// Grouped cross-validation: fit on the other folds, score the held-out
/// fold. Returns per-fold candidate and incumbent AUCs and the pooled
/// out-of-fold predictions.
fn cross_validate(
    scored: &[Scored],
    group: impl Fn(&Scored) -> u64,
    incumbent: &Weights,
) -> (Vec<f64>, Vec<f64>, OutOfFold) {
    let mut cv_candidate = Vec::new();
    let mut cv_incumbent = Vec::new();
    let mut oof = Vec::new();
    for fold in 0..FOLDS {
        let train_s: Vec<&Scored> = scored.iter().filter(|s| group(s) != fold).collect();
        let eval_s: Vec<&Scored> = scored.iter().filter(|s| group(s) == fold).collect();
        // Token model trained on this fold's training records only; the
        // training records themselves carry cross-fitted token scores.
        let table = train_table(&train_s);
        let train = crossfit_token_examples(&train_s);
        let eval = token_examples(&eval_s, &table);
        if eval.is_empty() || train.is_empty() {
            continue;
        }
        let cand = fit(&train, &options(incumbent));
        if eval.iter().any(|e| e.is_slop) && eval.iter().any(|e| !e.is_slop) {
            cv_candidate.push(auc(&cand, &eval));
            cv_incumbent.push(auc(incumbent, &eval));
        } else {
            println!("fold {fold}: single-class eval, AUC skipped");
        }
        for (e, s) in eval.into_iter().zip(&eval_s) {
            let p = cand.score(&e.fires).probability;
            oof.push((e, p, s.id.clone()));
        }
    }
    (cv_candidate, cv_incumbent, oof)
}

/// Wilson 95 percent interval on a proportion: what a recall of k out of
/// n is consistent with. Nine misses out of nine is not "zero recall", it
/// is "below about a third".
fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.96f64;
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

fn recall_at_fpr(scored: &[(f64, bool)], target: f64) -> f64 {
    let neg: Vec<f64> = scored.iter().filter(|s| !s.1).map(|s| s.0).collect();
    let th = thresholds_from_negatives(neg, target, target, target);
    let pos: Vec<f64> = scored.iter().filter(|s| s.1).map(|s| s.0).collect();
    if pos.is_empty() {
        return f64::NAN;
    }
    pos.iter().filter(|p| **p >= th.label).count() as f64 / pos.len() as f64
}

fn load(dir: &std::path::Path) -> Vec<Record> {
    let mut out = Vec::new();
    for name in ["slop.jsonl", "ham.jsonl"] {
        let path = dir.join(name);
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("missing {}", path.display());
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            match serde_json::from_str::<Record>(line) {
                Ok(r) => out.push(r),
                Err(e) => eprintln!("{name}:{}: skipped ({e})", i + 1),
            }
        }
    }
    out
}

/// "NONE" when the author had no visible PR to this repo before this one,
/// otherwise "CONTRIBUTOR". The mined `author_association` cannot be used:
/// merging promotes the author, so it encodes the outcome.
fn point_in_time_association(dossier: &serde_json::Value, repo: &str, created_at: &str) -> String {
    let nodes = dossier["data"]["user"]["pullRequests"]["nodes"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let prior_here = nodes.iter().any(|n| {
        let created = n["createdAt"].as_str().unwrap_or("");
        let node_repo = n["repository"]["nameWithOwner"].as_str().unwrap_or("");
        !created.is_empty()
            && !created_at.is_empty()
            && created < created_at
            && node_repo.eq_ignore_ascii_case(repo)
    });
    if prior_here { "CONTRIBUTOR" } else { "NONE" }.to_string()
}

/// Load the optional detector sidecar (`detector.jsonl`): offline scores
/// from the self-hosted AI-text detector, keyed by `repo#number`.
fn load_detector(dir: &std::path::Path) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(dir.join("detector.jsonl")) else {
        return out;
    };
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line)
            && let (Some(id), Some(p)) = (v["id"].as_str(), v["probability"].as_f64())
        {
            out.insert(id.to_string(), p);
        }
    }
    out
}

/// Replay all records in arrival order per repo, collecting fires exactly as
/// production would see them. Records the pipeline decides without scoring
/// come back separately with their effective probability: exempt is 0,
/// policy close is 1. Benchmark emission needs a score for every record.
fn replay(
    records: &[Record],
    detector: &BTreeMap<String, f64>,
) -> (Vec<Scored>, Vec<(String, bool, f64)>) {
    let cfg = RepoConfig {
        dry_run: false,
        ..Default::default()
    };
    let weights = Weights::default_table();

    // Order per repo by created_at so clustering and burst see real arrivals.
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by_key(|&i| (records[i].repo.clone(), records[i].created_at.clone()));

    let mut stores: BTreeMap<String, ClusterStore> = BTreeMap::new();
    let mut out = Vec::new();
    let mut decided = Vec::new();
    for &i in &order {
        let r = &records[i];
        let now = r
            .created_at
            .parse::<DateTime<Utc>>()
            .unwrap_or_else(|_| DateTime::<Utc>::from_timestamp(1_750_000_000, 0).unwrap());

        // Recorded author_association is an outcome field: GitHub computes
        // it at read time, so a merged PR's author reads CONTRIBUTOR even
        // though they were a stranger when it opened. Reconstruct the
        // arrival-time value from history predating this PR, the way a live
        // webhook would have seen it.
        let association = point_in_time_association(&r.dossier, &r.repo, &r.created_at);
        let ev = PrEvent {
            action: "opened".into(),
            repo: r.repo.clone(),
            number: r.number,
            author: r.author.clone(),
            title: r.title.clone(),
            body: r.body.clone(),
            additions: r.additions,
            deletions: r.deletions,
            changed_files: r.changed_files,
            commit_count: r.commits.len() as u64,
            author_association: association,
            head_is_fork: true,
            head_ref: r.head_ref.clone(),
            node_id: String::new(),
            labels: vec![],
        };
        let mut facts: DossierFacts = if r.dossier.is_null() {
            DossierFacts::default()
        } else {
            parse_dossier(&r.author, &r.dossier, now)
        };
        facts.search_blocked = r.search_blocked;
        let commit_emails: Vec<String> = r.commits.iter().map(|c| c.email.clone()).collect();
        let commit_messages: Vec<String> = r.commits.iter().map(|c| c.message.clone()).collect();
        let commit_times: Vec<chrono::DateTime<Utc>> = r
            .commits
            .iter()
            .filter_map(|c| chrono::DateTime::parse_from_rfc3339(&c.date).ok())
            .map(|t| t.with_timezone(&Utc))
            .collect();
        let (e, t, f) = scan_markers(&commit_emails, &commit_messages, &r.body);
        facts.agent_email |= e;
        facts.agent_trailer |= t;
        facts.generation_footer |= f;

        let inputs = ScoreInputs {
            config: &cfg,
            event: &ev,
            diff: &r.diff,
            changed_paths: r.files.clone(),
            commit_emails,
            commit_messages,
            commit_times,
            dossier: facts,
            pr_labels: vec![],
            template: None,
            detector_score: detector.get(&format!("{}#{}", r.repo, r.number)).copied(),
        };
        let store = stores
            .entry(r.repo.clone())
            .or_insert_with(|| ClusterStore::new(0.5));
        match process(&inputs, &weights, store, "tune-salt", now) {
            Outcome::Scored { verdict, .. } => out.push(Scored {
                fires: verdict
                    .evidence
                    .iter()
                    .map(|e| Fire::new(&e.rule, e.value))
                    .collect(),
                is_slop: r.label == "slop",
                author: r.author.clone(),
                repo: r.repo.clone(),
                source: r.source.clone(),
                id: format!("{}#{}", r.repo, r.number),
                prose: format!("{}\n{}", r.title, r.body),
            }),
            Outcome::Exempt => {
                decided.push((format!("{}#{}", r.repo, r.number), r.label == "slop", 0.0))
            }
            Outcome::PolicyClose { .. } => {
                decided.push((format!("{}#{}", r.repo, r.number), r.label == "slop", 1.0))
            }
        }
    }
    (out, decided)
}

fn fold_of(key: &str) -> u64 {
    fnv1a64(key.to_lowercase().as_bytes()) % FOLDS
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dry = args.iter().any(|a| a == "--dry");
    let score_only = args.iter().any(|a| a == "--score-only");
    let mut emit: Option<String> = None;
    let mut fires_path: Option<String> = None;
    let mut dir: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--emit" {
            emit = it.next().cloned();
        } else if a == "--fires" {
            fires_path = it.next().cloned();
        } else if !a.starts_with("--") {
            dir = Some(a.clone());
        }
    }
    let dir = dir.unwrap_or_else(|| format!("{}/bench/corpus/archive", env!("CARGO_MANIFEST_DIR")));

    let records = load(std::path::Path::new(&dir));
    let n_slop = records.iter().filter(|r| r.label == "slop").count();
    println!(
        "corpus: {} records ({} slop, {} ham)",
        records.len(),
        n_slop,
        records.len() - n_slop
    );
    let mut by_source: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &records {
        *by_source.entry(r.source.as_str()).or_default() += 1;
    }
    for (s, c) in &by_source {
        println!("  {s}: {c}");
    }

    let detector = load_detector(std::path::Path::new(&dir));
    println!("detector sidecar: {} scores", detector.len());
    let (scored, decided) = replay(&records, &detector);
    println!(
        "replayed: {} scored, {} decided without scoring",
        scored.len(),
        decided.len()
    );

    // --score-only: evaluate the shipped weight table on this corpus and
    // emit predictions; fit nothing. For held-out secondary corpora.
    if score_only {
        let table = Weights::default_table();
        let refs: Vec<&Scored> = scored.iter().collect();
        let examples = token_examples(&refs, &pullsift::tokenscore::TokenTable::embedded());
        let a = auc(&table, &examples);
        println!("score-only AUC with shipped weights: {a:.4}");
        if let Some(path) = &emit {
            let mut lines = String::new();
            for (ex, s) in examples.iter().zip(&refs) {
                let p = table.score(&ex.fires).probability;
                lines.push_str(&format!(
                    "{}\n",
                    serde_json::json!({ "id": s.id, "score": p })
                ));
            }
            for (id, _, p) in &decided {
                lines.push_str(&format!(
                    "{}\n",
                    serde_json::json!({ "id": id, "score": p })
                ));
            }
            std::fs::write(path, lines).unwrap();
            println!("wrote predictions to {path}");
        }
        return;
    }

    // --fires PATH: every record's raw fires, with its label and source.
    // Aggregate coverage answers "does this rule separate on this corpus";
    // it cannot answer "does it separate on this slice of it", which is
    // the question whenever a lane only applies to part of the corpus.
    if let Some(path) = &fires_path {
        let table = pullsift::tokenscore::TokenTable::embedded();
        let refs: Vec<&Scored> = scored.iter().collect();
        let mut lines = String::new();
        for (ex, s) in token_examples(&refs, &table).iter().zip(&refs) {
            let fired: BTreeMap<&str, f64> = ex
                .fires
                .iter()
                .filter(|f| f.value > 0.0)
                .map(|f| (f.rule.as_str(), f.value))
                .collect();
            lines.push_str(&format!(
                "{}\n",
                serde_json::json!({
                    "id": s.id, "slop": s.is_slop, "source": s.source, "fires": fired,
                })
            ));
        }
        std::fs::write(path, lines).unwrap();
        println!("wrote fires to {path}");
    }

    // Cross-validation, author-grouped: the benchmark's split. Then
    // repo-grouped: the install-you-have-never-seen number.
    let incumbent = Weights::default_table();
    let (cv_candidate, cv_incumbent, oof) =
        cross_validate(&scored, |s| fold_of(&s.author), &incumbent);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let round3 = |v: &[f64]| {
        v.iter()
            .map(|a| (a * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    };
    let pooled: Vec<(f64, bool)> = oof.iter().map(|(e, p, _)| (*p, e.is_slop)).collect();
    println!(
        "cv AUC (author-grouped): candidate {:.4} (folds {:?}), incumbent {:.4}; pooled OOF AUC {:.4}, recall at 1% FPR {:.3}",
        mean(&cv_candidate),
        round3(&cv_candidate),
        mean(&cv_incumbent),
        auc_of(&pooled),
        recall_at_fpr(&pooled, 0.01),
    );
    let (cv_repo, _, oof_repo) = cross_validate(&scored, |s| fold_of(&s.repo), &incumbent);
    let pooled_repo: Vec<(f64, bool)> = oof_repo.iter().map(|(e, p, _)| (*p, e.is_slop)).collect();
    println!(
        "cv AUC (repo-grouped):   candidate {:.4} (folds {:?}); pooled OOF AUC {:.4}, recall at 1% FPR {:.3}",
        mean(&cv_repo),
        round3(&cv_repo),
        auc_of(&pooled_repo),
        recall_at_fpr(&pooled_repo, 0.01),
    );

    // Leave-one-source-out: hold out every slop source with enough
    // records, fit on the rest, and ask how much of it the table still
    // catches at the 1% cut set on the training negatives. A source at
    // zero is a campaign shape the other sources do not teach.
    let mut sources: BTreeMap<&str, usize> = BTreeMap::new();
    for s in scored.iter().filter(|s| s.is_slop) {
        *sources.entry(s.source.as_str()).or_default() += 1;
    }
    println!("\nheld-out recall by source (1% FPR cut from training negatives):");
    let mut loso: BTreeMap<String, f64> = BTreeMap::new();
    for (src, n) in sources.iter().filter(|(_, n)| **n >= 15) {
        let train_s: Vec<&Scored> = scored
            .iter()
            .filter(|s| !(s.is_slop && s.source == *src))
            .collect();
        let held_s: Vec<&Scored> = scored
            .iter()
            .filter(|s| s.is_slop && s.source == *src)
            .collect();
        let table = train_table(&train_s);
        let train = crossfit_token_examples(&train_s);
        let held = token_examples(&held_s, &table);
        let w = fit(&train, &options(&incumbent));
        let cut = thresholds_at_fpr(&w, &train, 0.01, 0.01, 0.01).label;
        let caught = held
            .iter()
            .filter(|e| w.score(&e.fires).probability >= cut)
            .count();
        let r = caught as f64 / held.len() as f64;
        let (lo, hi) = wilson(caught, held.len());
        println!("  {src:28} {caught:3}/{n:<3} {r:.2}  [{lo:.2}, {hi:.2}]");
        loso.insert(src.to_string(), r);
    }

    // Final token table and fit on everything; the prior anchors what the
    // corpus cannot price.
    let all_refs: Vec<&Scored> = scored.iter().collect();
    let final_table = train_table(&all_refs);
    println!("\ntoken table: {} tokens", final_table.llr.len());
    let all = crossfit_token_examples(&all_refs);
    let mut final_w = fit(&all, &options(&incumbent));

    // Thresholds from pooled out-of-fold predictions: in-sample selection
    // was measurably optimistic (OOF FPR blew the targets), so the cuts
    // come from probabilities the models did not train on. A tier whose
    // target the negative count cannot certify sits a full logit above
    // the highest negative instead of on the empirical quantile.
    let in_sample = thresholds_at_fpr(&final_w, &all, 0.05, 0.01, 0.001);
    let oof_neg: Vec<f64> = oof
        .iter()
        .filter(|(e, _, _)| !e.is_slop)
        .map(|(_, p, _)| *p)
        .collect();
    let n_neg_oof = oof_neg.len();
    final_w.thresholds = thresholds_from_negatives(oof_neg, 0.05, 0.01, 0.001);
    let certified = (
        certifiable(n_neg_oof, 0.05),
        certifiable(n_neg_oof, 0.01),
        certifiable(n_neg_oof, 0.001),
    );
    println!(
        "in-sample thresholds would have been: label {:.4} hold {:.4} close {:.4}",
        in_sample.label, in_sample.hold, in_sample.close
    );

    // Out-of-fold FPR at the final thresholds: the honesty check on
    // in-sample threshold selection.
    let oof_fpr = |t: f64| {
        let neg: Vec<&(Example, f64, String)> = oof.iter().filter(|(e, _, _)| !e.is_slop).collect();
        if neg.is_empty() {
            return 0.0;
        }
        neg.iter().filter(|(_, p, _)| *p >= t).count() as f64 / neg.len() as f64
    };
    println!(
        "thresholds: label {:.4} hold {:.4} close {:.4} (certified on {n_neg_oof} negatives: label {} hold {} close {})",
        final_w.thresholds.label,
        final_w.thresholds.hold,
        final_w.thresholds.close,
        certified.0,
        certified.1,
        certified.2,
    );
    println!(
        "in-sample FPR: label {:.4} hold {:.4} close {:.4}",
        observed_fpr(&final_w, &all, final_w.thresholds.label),
        observed_fpr(&final_w, &all, final_w.thresholds.hold),
        observed_fpr(&final_w, &all, final_w.thresholds.close),
    );
    println!(
        "out-of-fold FPR at those thresholds: label {:.4} hold {:.4} close {:.4}",
        oof_fpr(final_w.thresholds.label),
        oof_fpr(final_w.thresholds.hold),
        oof_fpr(final_w.thresholds.close),
    );

    // Weight report.
    let mut ranked: Vec<(&String, &f64)> = final_w.rules.iter().collect();
    ranked.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    println!("\ntop rules by |weight|:");
    for (rule, w) in ranked.iter().take(20) {
        println!("  {rule:24} {w:+.3}");
    }
    let dead: Vec<&str> = ranked
        .iter()
        .filter(|(_, w)| w.abs() < 0.05)
        .map(|(r, _)| r.as_str())
        .collect();
    println!("near-zero rules: {dead:?}");

    // Coverage. A weight only means something where the rule fires, so a
    // near-zero weight has two very different causes: the rule fires and
    // does not separate, or it never fires and was never priced at all.
    // The empirical log-ratio of fire rates says which.
    let n_pos = all.iter().filter(|e| e.is_slop).count();
    let n_neg = all.len() - n_pos;
    let cov = fire_counts(&all);
    let mut rows: Vec<(&str, usize, usize, f64)> = cov
        .iter()
        .map(|(r, (p, n))| {
            let rate_p = (*p as f64 + 0.5) / (n_pos as f64 + 1.0);
            let rate_n = (*n as f64 + 0.5) / (n_neg as f64 + 1.0);
            (r.as_str(), *p, *n, (rate_p / rate_n).ln())
        })
        .collect();
    rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nrule coverage ({n_pos} slop, {n_neg} ham):");
    for (r, p, n, llr) in &rows {
        let w = final_w.rules.get(*r).copied().unwrap_or(0.0);
        println!("  {r:24} slop {p:4}  ham {n:5}  llr {llr:+.2}  weight {w:+.3}");
    }

    // Ham false positives at each tier, for eyeballing. `all` is aligned
    // with `scored` and carries the token-model fire.
    println!("\nham false positives:");
    for (s, ex) in scored.iter().zip(&all) {
        if s.is_slop {
            continue;
        }
        let v = final_w.score(&ex.fires);
        if v.probability >= final_w.thresholds.label {
            let top: Vec<&str> = v
                .evidence
                .iter()
                .filter(|e| e.contribution > 0.2)
                .take(4)
                .map(|e| e.rule.as_str())
                .collect();
            let tier = format!("{:?}", v.tier);
            println!(
                "  {:40} p={:.3} {:6} {:?} [{}]",
                s.id, v.probability, tier, top, s.source
            );
        }
    }

    // Per-source recall at hold.
    println!("\nper-source recall at hold threshold:");
    let mut per: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (s, ex) in scored.iter().zip(&all) {
        if !s.is_slop {
            continue;
        }
        let e = per.entry(s.source.as_str()).or_default();
        e.1 += 1;
        if final_w.score(&ex.fires).probability >= final_w.thresholds.hold {
            e.0 += 1;
        }
    }
    for (src, (hit, total)) in &per {
        let (lo, hi) = wilson(*hit, *total);
        println!("  {src:20} {hit}/{total}  [{lo:.2}, {hi:.2}]");
    }

    // Benchmark emission: one out-of-fold prediction per scored record
    // (each from a model that never saw the record's author), plus the
    // records the pipeline decided without scoring.
    if let Some(path) = &emit {
        let mut lines = String::new();
        for (_, p, id) in &oof {
            lines.push_str(&format!(
                "{}\n",
                serde_json::json!({ "id": id, "score": p })
            ));
        }
        for (id, _, p) in &decided {
            lines.push_str(&format!(
                "{}\n",
                serde_json::json!({ "id": id, "score": p })
            ));
        }
        std::fs::write(path, lines).unwrap();
        println!("\nwrote predictions to {path}");
    }

    if dry {
        println!("\n--dry: not writing weights");
        return;
    }
    // The weights file doubles as the model card: what it was fitted on,
    // how it scored out of fold under both groupings, which tiers the
    // negative count certifies, and every rule's fire counts beside its
    // weight.
    let card: BTreeMap<&str, serde_json::Value> = cov
        .iter()
        .map(|(r, (p, n))| (r.as_str(), serde_json::json!({ "slop": p, "ham": n })))
        .collect();
    final_w.meta = Some(serde_json::json!({
        "fitted_at": Utc::now().to_rfc3339(),
        "corpus": { "total": records.len(), "slop": n_slop },
        "cv_auc": mean(&cv_candidate),
        "cv_auc_repo_grouped": mean(&cv_repo),
        "incumbent_cv_auc": mean(&cv_incumbent),
        "oof_recall_at_1pct_fpr": recall_at_fpr(&pooled, 0.01),
        "held_out_recall_by_source": loso,
        "thresholds_certified": {
            "label": certified.0, "hold": certified.1, "close": certified.2,
            "negatives": n_neg_oof,
        },
        "fires": card,
    }));
    let path = format!("{}/weights/default.json", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(&path, serde_json::to_string_pretty(&final_w).unwrap()).unwrap();
    let tok_path = format!("{}/weights/tokens.json", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(
        &tok_path,
        serde_json::to_string_pretty(&final_table).unwrap(),
    )
    .unwrap();
    println!("\nwrote {path} and {tok_path}");
}
