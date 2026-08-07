//! Offline tuner: replay the mined corpus through the production pipeline,
//! cross-validate a fitted weight table against the incumbent, and write
//! the new table with provenance.
//!
//! Usage: tune [--dry] [corpus-dir]
//!
//! Folds are grouped by author (an author never appears in both train and
//! eval). Rules that never fired in the corpus keep their incumbent weight:
//! no data means the prior stands, so rare-but-designed signals (network
//! verdicts, challenge outcomes) are not silently zeroed.

use chrono::{DateTime, Utc};
use pullsift::cluster::ClusterStore;
use pullsift::config::RepoConfig;
use pullsift::dossier::{parse_dossier, scan_markers, DossierFacts};
use pullsift::engine::{Fire, Weights};
use pullsift::fit::{auc, fit, observed_fpr, thresholds_at_fpr, Example, FitOptions};
use pullsift::hashing::fnv1a64;
use pullsift::pipeline::{process, Outcome, ScoreInputs};
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
    author_association: String,
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
}

struct Scored {
    fires: Vec<Fire>,
    is_slop: bool,
    author: String,
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
            Example::new(fires, s.is_slop)
        })
        .collect()
}

fn train_table(scored: &[&Scored]) -> pullsift::tokenscore::TokenTable {
    let docs: Vec<(String, bool)> = scored
        .iter()
        .map(|s| (s.prose.clone(), s.is_slop))
        .collect();
    pullsift::tokenscore::TokenTable::train(&docs)
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
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(id), Some(p)) = (v["id"].as_str(), v["probability"].as_f64()) {
                out.insert(id.to_string(), p);
            }
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

fn fold_of(author: &str) -> u64 {
    fnv1a64(author.to_lowercase().as_bytes()) % FOLDS
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dry = args.iter().any(|a| a == "--dry");
    let score_only = args.iter().any(|a| a == "--score-only");
    let mut emit: Option<String> = None;
    let mut dir: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--emit" {
            emit = it.next().cloned();
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

    // Cross-validation, author-grouped.
    let incumbent = Weights::default_table();
    let mut cv_candidate = Vec::new();
    let mut cv_incumbent = Vec::new();
    let mut oof: Vec<(Example, f64)> = Vec::new(); // out-of-fold: example + candidate prob
    let mut oof_ids: Vec<String> = Vec::new(); // aligned with oof, for --emit
    for fold in 0..FOLDS {
        let train_s: Vec<&Scored> = scored
            .iter()
            .filter(|s| fold_of(&s.author) != fold)
            .collect();
        let eval_s: Vec<&Scored> = scored
            .iter()
            .filter(|s| fold_of(&s.author) == fold)
            .collect();
        // Token model trained on this fold's training authors only.
        let table = train_table(&train_s);
        let train = token_examples(&train_s, &table);
        let eval = token_examples(&eval_s, &table);
        if eval.is_empty() || train.is_empty() {
            continue;
        }
        let cand = fit(&train, &FitOptions::default());
        // AUC is undefined on a single-class fold, but the fold's records
        // still get out-of-fold predictions so benchmark emission covers
        // every record.
        if eval.iter().any(|e| e.is_slop) && eval.iter().any(|e| !e.is_slop) {
            cv_candidate.push(auc(&cand, &eval));
            cv_incumbent.push(auc(&incumbent, &eval));
        } else {
            println!("fold {fold}: single-class eval, AUC skipped");
        }
        for (e, s) in eval.into_iter().zip(&eval_s) {
            let p = cand.score(&e.fires).probability;
            oof.push((e, p));
            oof_ids.push(s.id.clone());
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "cv AUC: candidate {:.4} (folds {:?}), incumbent {:.4}",
        mean(&cv_candidate),
        cv_candidate
            .iter()
            .map(|a| (a * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>(),
        mean(&cv_incumbent),
    );

    // Final token table and fit on everything; unfired rules keep incumbent
    // weights.
    let all_refs: Vec<&Scored> = scored.iter().collect();
    let final_table = train_table(&all_refs);
    println!("token table: {} tokens", final_table.llr.len());
    let all = token_examples(&all_refs, &final_table);
    let mut final_w = fit(&all, &FitOptions::default());
    for (rule, w) in &incumbent.rules {
        final_w.rules.entry(rule.clone()).or_insert(*w);
    }
    // Under-sampled rules keep at least their prior. Cluster rules: the
    // corpus holds few genuine multi-account waves, whose members are also
    // caught by token and title rules, so correlation starves the cluster
    // weights; the corpus cannot yet price wave mechanics. The provenance
    // markers (AGENT_*) lost their floors once the corpus gained merged
    // agent PRs on the ham side: the fit prices them now, and a floor
    // there forced false positives on accepted agent work.
    for rule in [
        "CLUSTER_BURST",
        "CLUSTER_SIZE_LOG",
        "CLUSTER_STYLE_COHESION",
    ] {
        if let (Some(prior), Some(fitted)) =
            (incumbent.rules.get(rule), final_w.rules.get_mut(rule))
        {
            if *fitted < *prior {
                *fitted = *prior;
            }
        }
    }

    // Thresholds from pooled out-of-fold predictions: in-sample selection
    // was measurably optimistic (OOF FPR blew the targets), so the cuts
    // come from probabilities the models did not train on.
    let in_sample = thresholds_at_fpr(&final_w, &all, 0.05, 0.01, 0.001);
    let mut oof_neg: Vec<f64> = oof
        .iter()
        .filter(|(e, _)| !e.is_slop)
        .map(|(_, p)| *p)
        .collect();
    oof_neg.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cut = |target: f64| -> f64 {
        let n = oof_neg.len();
        let allowed = (target * n as f64).floor() as usize;
        if allowed == 0 || n == 0 {
            oof_neg
                .first()
                .map(|m| (m + 1e-9).min(1.0))
                .unwrap_or(0.999)
        } else {
            (oof_neg[allowed] + 1e-9).min(1.0)
        }
    };
    let label = cut(0.05);
    let hold = cut(0.01).max(label + 1e-9);
    let close = cut(0.001).max(hold + 1e-9);
    final_w.thresholds = pullsift::engine::Thresholds { label, hold, close };
    println!(
        "in-sample thresholds would have been: label {:.4} hold {:.4} close {:.4}",
        in_sample.label, in_sample.hold, in_sample.close
    );

    // Out-of-fold FPR at the final thresholds: the honesty check on
    // in-sample threshold selection.
    let oof_fpr = |t: f64| {
        let neg: Vec<&(Example, f64)> = oof.iter().filter(|(e, _)| !e.is_slop).collect();
        if neg.is_empty() {
            return 0.0;
        }
        neg.iter().filter(|(_, p)| *p >= t).count() as f64 / neg.len() as f64
    };
    println!(
        "thresholds: label {:.4} hold {:.4} close {:.4}",
        final_w.thresholds.label, final_w.thresholds.hold, final_w.thresholds.close
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
        println!("  {src:20} {hit}/{total}");
    }

    // Benchmark emission: one out-of-fold prediction per scored record
    // (each from a model that never saw the record's author), plus the
    // records the pipeline decided without scoring.
    if let Some(path) = &emit {
        let mut lines = String::new();
        for ((_, p), id) in oof.iter().zip(&oof_ids) {
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
    final_w.meta = Some(serde_json::json!({
        "fitted_at": Utc::now().to_rfc3339(),
        "corpus": { "total": records.len(), "slop": n_slop },
        "cv_auc": mean(&cv_candidate),
        "incumbent_cv_auc": mean(&cv_incumbent),
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
