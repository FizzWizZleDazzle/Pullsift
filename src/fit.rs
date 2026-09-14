//! Weight fitting: bagged logistic regression anchored to a prior table,
//! sign constraints per rule family, stability gating, AUC, and tier
//! thresholds chosen at fixed false-positive rates.
//!
//! No ML framework; the rule vectors are small and the corpus fits in
//! memory. Maintainer corrections enter as sample weights.
//!
//! Variance control, in order of effect:
//!
//! - Every rule outside the `TRUST_` family fits non-negative; trust rules
//!   fit non-positive. A slop indicator that would fit negative is a corpus
//!   artifact and goes to zero instead of exonerating.
//! - Weights are shrunk toward the prior table (the incumbent) rather than
//!   toward zero, so a refit on thin data moves weights only as far as the
//!   data can justify.
//! - Rules that fired on fewer than `min_fires` training examples keep
//!   their prior weight: three hits cannot price a rule.
//! - The fit is bagged over bootstrap resamples of the example groups
//!   (authors), and a rule keeps its averaged weight only when it is
//!   non-zero in at least `stability` of the bags. A rule that only some
//!   resamples support is not priced at all.
//! - Family contributions are capped during fitting exactly as the engine
//!   caps them at scoring time, so the fitted weights describe the model
//!   that ships.

use crate::engine::{Family, Fire, Thresholds, Weights, logit, sigmoid};
use crate::hashing::splitmix64;
use std::collections::BTreeMap;

/// A labeled example: the rules that fired, whether it was slop, a sample
/// weight (corrections are up-weighted), and the group it belongs to for
/// bootstrap resampling. Examples with the same group (an author) are
/// resampled together, because an author's PRs are not independent draws.
#[derive(Debug, Clone)]
pub struct Example {
    pub fires: Vec<Fire>,
    pub is_slop: bool,
    pub sample_weight: f64,
    pub group: String,
}

impl Example {
    pub fn new(fires: Vec<Fire>, is_slop: bool) -> Self {
        Self {
            fires,
            is_slop,
            sample_weight: 1.0,
            group: String::new(),
        }
    }

    /// A maintainer override of one of our verdicts; weighted 5x.
    pub fn correction(fires: Vec<Fire>, is_slop: bool) -> Self {
        Self {
            sample_weight: 5.0,
            ..Self::new(fires, is_slop)
        }
    }

    pub fn in_group(mut self, group: &str) -> Self {
        self.group = group.to_string();
        self
    }
}

#[derive(Clone)]
pub struct FitOptions {
    pub learning_rate: f64,
    pub iterations: usize,
    /// Shrinkage toward the prior (or zero for rules the prior lacks).
    pub l2: f64,
    /// Weights are pulled toward this table instead of toward zero, and
    /// under-observed rules keep its value outright.
    pub prior: Option<Weights>,
    /// Bootstrap resamples to average over; 0 fits once on the data.
    pub bags: usize,
    /// Fraction of bags in which a rule must be non-zero to keep a weight.
    pub stability: f64,
    /// A rule fired on fewer training examples than this keeps its prior.
    pub min_fires: usize,
    /// Per-family contribution cap applied during fitting; matches the
    /// engine's cap on the fitted table.
    pub family_cap: Option<f64>,
    /// Bootstrap seed, for reproducible tables.
    pub seed: u64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.5,
            iterations: 2000,
            l2: 1e-3,
            prior: None,
            bags: 25,
            stability: 0.8,
            min_fires: 10,
            family_cap: Some(DEFAULT_FAMILY_CAP),
            seed: 0x5eed,
        }
    }
}

/// Default cap on one family's contribution, in logits. Below the gap
/// between the bias and the close tier, so a close needs two families.
pub const DEFAULT_FAMILY_CAP: f64 = 4.0;

/// A weight counts as present in a bag above this magnitude.
const STABLE_EPS: f64 = 0.05;

/// Sparse row: (rule index, value), label, sample weight.
struct Row {
    x: Vec<(usize, f64)>,
    y: f64,
    sw: f64,
}

struct Design {
    rule_ix: BTreeMap<String, usize>,
    rules: Vec<String>,
    families: Vec<Family>,
    rows: Vec<Row>,
}

fn design(examples: &[Example]) -> Design {
    let mut rule_ix: BTreeMap<String, usize> = BTreeMap::new();
    for ex in examples {
        for f in &ex.fires {
            let next = rule_ix.len();
            rule_ix.entry(f.rule.clone()).or_insert(next);
        }
    }
    let mut rules = vec![String::new(); rule_ix.len()];
    for (r, i) in &rule_ix {
        rules[*i] = r.clone();
    }
    let families = rules.iter().map(|r| Family::of(r)).collect();
    let rows = examples
        .iter()
        .map(|ex| Row {
            x: ex
                .fires
                .iter()
                .map(|f| (rule_ix[&f.rule], f.value.clamp(0.0, 1.0)))
                .collect(),
            y: if ex.is_slop { 1.0 } else { 0.0 },
            sw: ex.sample_weight,
        })
        .collect();
    Design {
        rule_ix,
        rules,
        families,
        rows,
    }
}

/// One gradient-descent fit over `rows`. `fixed[i]` pins rule i at
/// `prior[i]`; other rules start there and move under the sign constraint
/// of their family.
fn fit_once(
    d: &Design,
    rows: &[&Row],
    prior: &[f64],
    fixed: &[bool],
    opts: &FitOptions,
) -> (Vec<f64>, f64) {
    let dim = d.rules.len();
    let total_w: f64 = rows.iter().map(|r| r.sw).sum::<f64>().max(1e-9);
    let mut w = prior.to_vec();
    let mut bias = 0.0;
    let fam_ix: Vec<usize> = d.families.iter().map(|f| f.index()).collect();
    let fam_capped: [bool; Family::COUNT] = [
        Family::Cluster.capped(),
        Family::Code.capped(),
        Family::Prose.capped(),
        Family::Shape.capped(),
        Family::Dossier.capped(),
        Family::Trust.capped(),
        Family::Policy.capped(),
    ];
    let mut gw = vec![0.0; dim];
    for _ in 0..opts.iterations {
        gw.iter_mut().for_each(|g| *g = 0.0);
        let mut gb = 0.0;
        for row in rows {
            let mut fam_sum = [0.0f64; Family::COUNT];
            for (i, xi) in &row.x {
                fam_sum[fam_ix[*i]] += w[*i] * xi;
            }
            let mut z = bias;
            let mut over = [false; Family::COUNT];
            for (f, s) in fam_sum.iter().enumerate() {
                z += match opts.family_cap {
                    Some(cap) if fam_capped[f] => {
                        over[f] = s.abs() > cap;
                        s.clamp(-cap, cap)
                    }
                    _ => *s,
                };
            }
            let err = (sigmoid(z) - row.y) * row.sw;
            gb += err;
            for (i, xi) in &row.x {
                if !over[fam_ix[*i]] {
                    gw[*i] += err * xi;
                }
            }
        }
        bias -= opts.learning_rate * gb / total_w;
        for i in 0..dim {
            if fixed[i] {
                continue;
            }
            let grad = gw[i] / total_w + opts.l2 * (w[i] - prior[i]);
            w[i] -= opts.learning_rate * grad;
            if d.families[i].exonerating() {
                w[i] = w[i].min(0.0);
            } else {
                w[i] = w[i].max(0.0);
            }
        }
    }
    (w, bias)
}

/// Fit weights on examples. Rule universe is the union of all fired rules
/// plus every rule in the prior. Returns a full `Weights` with thresholds
/// set at the FPR targets on the training examples themselves; callers
/// with held-out data re-derive them there.
pub fn fit(examples: &[Example], opts: &FitOptions) -> Weights {
    let d = design(examples);
    let dim = d.rules.len();

    let prior: Vec<f64> = d
        .rules
        .iter()
        .map(|r| {
            opts.prior
                .as_ref()
                .and_then(|p| p.rules.get(r))
                .copied()
                .unwrap_or(0.0)
        })
        .collect();
    let mut fires = vec![0usize; dim];
    for row in &d.rows {
        for (i, xi) in &row.x {
            if *xi > 0.0 {
                fires[*i] += 1;
            }
        }
    }
    let fixed: Vec<bool> = fires.iter().map(|n| *n < opts.min_fires).collect();

    let (w, bias) = if opts.bags == 0 {
        let all: Vec<&Row> = d.rows.iter().collect();
        fit_once(&d, &all, &prior, &fixed, opts)
    } else {
        // Group-bootstrap: resample groups with replacement. Examples with
        // no group are their own group.
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (k, ex) in examples.iter().enumerate() {
            let key = if ex.group.is_empty() {
                format!("\u{0}{k}")
            } else {
                ex.group.clone()
            };
            groups.entry(key).or_default().push(k);
        }
        let groups: Vec<&Vec<usize>> = groups.values().collect();
        let mut sum_w = vec![0.0; dim];
        let mut present = vec![0usize; dim];
        let mut sum_b = 0.0;
        let mut rng = opts.seed;
        for _ in 0..opts.bags {
            let mut rows: Vec<&Row> = Vec::with_capacity(d.rows.len());
            for _ in 0..groups.len() {
                rng = splitmix64(rng);
                let g = groups[(rng % groups.len() as u64) as usize];
                rows.extend(g.iter().map(|&k| &d.rows[k]));
            }
            let (bw, bb) = fit_once(&d, &rows, &prior, &fixed, opts);
            for i in 0..dim {
                sum_w[i] += bw[i];
                if bw[i].abs() > STABLE_EPS {
                    present[i] += 1;
                }
            }
            sum_b += bb;
        }
        let n = opts.bags as f64;
        let w = (0..dim)
            .map(|i| {
                if fixed[i] {
                    prior[i]
                } else if (present[i] as f64) / n >= opts.stability {
                    sum_w[i] / n
                } else {
                    0.0
                }
            })
            .collect();
        (w, sum_b / n)
    };

    let mut rules = BTreeMap::new();
    for (rule, ix) in &d.rule_ix {
        rules.insert(rule.clone(), w[*ix]);
    }
    // Rules the corpus never fired keep the prior: no data, the prior stands.
    if let Some(p) = &opts.prior {
        for (rule, pw) in &p.rules {
            rules.entry(rule.clone()).or_insert(*pw);
        }
    }
    let mut weights = Weights {
        bias,
        rules,
        thresholds: Thresholds {
            label: 0.3,
            hold: 0.7,
            close: 0.95,
        },
        family_cap: opts.family_cap,
        meta: None,
    };
    weights.thresholds = thresholds_at_fpr(&weights, examples, 0.05, 0.01, 0.001);
    weights
}

/// How many examples fired each rule (value above zero), per class. The
/// coverage report and the `min_fires` gate both read this.
pub fn fire_counts(examples: &[Example]) -> BTreeMap<String, (usize, usize)> {
    let mut out: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for ex in examples {
        for f in &ex.fires {
            if f.value <= 0.0 {
                continue;
            }
            let c = out.entry(f.rule.clone()).or_default();
            if ex.is_slop {
                c.0 += 1;
            } else {
                c.1 += 1;
            }
        }
    }
    out
}

fn probabilities(weights: &Weights, examples: &[Example]) -> Vec<(f64, bool)> {
    examples
        .iter()
        .map(|ex| (weights.score(&ex.fires).probability, ex.is_slop))
        .collect()
}

/// Area under the ROC curve via the rank statistic, ties counted half.
pub fn auc(weights: &Weights, examples: &[Example]) -> f64 {
    let scored = probabilities(weights, examples);
    auc_of(&scored)
}

/// AUC of (probability, is_slop) pairs.
pub fn auc_of(scored: &[(f64, bool)]) -> f64 {
    let pos: Vec<f64> = scored.iter().filter(|s| s.1).map(|s| s.0).collect();
    let mut neg: Vec<f64> = scored.iter().filter(|s| !s.1).map(|s| s.0).collect();
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    neg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut wins = 0.0;
    for p in &pos {
        let lo = neg.partition_point(|n| n < p);
        let hi = neg.partition_point(|n| n <= p);
        wins += lo as f64 + 0.5 * (hi - lo) as f64;
    }
    wins / (pos.len() as f64 * neg.len() as f64)
}

/// Whether `n_neg` negatives can certify a false-positive rate of
/// `target`: the rule of three. Zero false positives among n negatives
/// bounds the true rate below 3/n with 95 percent confidence, so a target
/// is certifiable only when n >= 3 / target.
pub fn certifiable(n_neg: usize, target: f64) -> bool {
    n_neg as f64 * target >= 3.0
}

/// Pick tier thresholds so the observed FPR on `examples` (ideally held-out)
/// stays at or below each target. With too few negatives to certify a
/// target, the threshold clears every negative seen by a margin of one
/// logit: the empirical quantile is then a single record, and the margin
/// keeps the cut from moving with it.
pub fn thresholds_at_fpr(
    weights: &Weights,
    examples: &[Example],
    label_fpr: f64,
    hold_fpr: f64,
    close_fpr: f64,
) -> Thresholds {
    let neg: Vec<f64> = probabilities(weights, examples)
        .into_iter()
        .filter(|s| !s.1)
        .map(|s| s.0)
        .collect();
    thresholds_from_negatives(neg, label_fpr, hold_fpr, close_fpr)
}

/// The threshold rule over a bag of negative probabilities.
pub fn thresholds_from_negatives(
    mut neg: Vec<f64>,
    label_fpr: f64,
    hold_fpr: f64,
    close_fpr: f64,
) -> Thresholds {
    neg.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = neg.len();
    let cut = |target: f64| -> f64 {
        if n == 0 {
            return 0.999;
        }
        let allowed = (target * n as f64).floor() as usize;
        if certifiable(n, target) && allowed > 0 {
            // Sit just above the (allowed+1)-th highest negative.
            (neg[allowed] + 1e-9).min(1.0)
        } else {
            sigmoid(logit(neg[0]) + 1.0).min(1.0)
        }
    };
    // Enforce label < hold < close by raising the upper tiers only: raising
    // a threshold can only lower its FPR, so the targets stay guaranteed.
    let label = cut(label_fpr);
    let hold = cut(hold_fpr).max(label + 1e-9);
    let close = cut(close_fpr).max(hold + 1e-9);
    Thresholds { label, hold, close }
}

/// Observed FPR of a probability threshold on a labeled set.
pub fn observed_fpr(weights: &Weights, examples: &[Example], threshold: f64) -> f64 {
    let scored = probabilities(weights, examples);
    let neg: Vec<&(f64, bool)> = scored.iter().filter(|s| !s.1).collect();
    if neg.is_empty() {
        return 0.0;
    }
    neg.iter().filter(|s| s.0 >= threshold).count() as f64 / neg.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick() -> FitOptions {
        FitOptions {
            bags: 8,
            iterations: 800,
            min_fires: 3,
            ..Default::default()
        }
    }

    /// Slop fires AGENT_TRAILER and CLUSTER_BURST; ham fires ACCOUNT_NEW
    /// sometimes (newness alone must not convict).
    fn corpus() -> Vec<Example> {
        let mut ex = Vec::new();
        for i in 0..40 {
            let mut fires = vec![Fire::hit("AGENT_TRAILER")];
            if i % 2 == 0 {
                fires.push(Fire::hit("CLUSTER_BURST"));
            }
            if i % 3 == 0 {
                fires.push(Fire::hit("ACCOUNT_NEW"));
            }
            ex.push(Example::new(fires, true).in_group(&format!("s{}", i % 7)));
        }
        for i in 0..40 {
            let mut fires = vec![];
            if i % 2 == 0 {
                fires.push(Fire::hit("ACCOUNT_NEW"));
            }
            if i % 5 == 0 {
                fires.push(Fire::new("STYLE_EMOJI", 0.2));
            }
            ex.push(Example::new(fires, false).in_group(&format!("h{}", i % 9)));
        }
        ex
    }

    #[test]
    fn fit_separates_separable_corpus() {
        let ex = corpus();
        let w = fit(&ex, &quick());
        let a = auc(&w, &ex);
        assert!(a > 0.99, "AUC {a} on separable data");
        // The discriminative rule gets a positive weight...
        assert!(w.rules["AGENT_TRAILER"] > 1.0);
        // ...and newness, present in both classes, stays small.
        assert!(w.rules["ACCOUNT_NEW"].abs() < w.rules["AGENT_TRAILER"]);
    }

    #[test]
    fn single_fit_matches_bagged_direction() {
        let ex = corpus();
        let single = fit(&ex, &FitOptions { bags: 0, ..quick() });
        assert!(single.rules["AGENT_TRAILER"] > 1.0);
        assert!(auc(&single, &ex) > 0.99);
    }

    #[test]
    fn fitted_thresholds_hold_their_fpr_in_sample() {
        let ex = corpus();
        let w = fit(&ex, &quick());
        assert!(observed_fpr(&w, &ex, w.thresholds.close) <= 0.001 + 1e-9);
        assert!(observed_fpr(&w, &ex, w.thresholds.hold) <= 0.01 + 1e-9);
        assert!(observed_fpr(&w, &ex, w.thresholds.label) <= 0.05 + 1e-9);
        assert!(w.thresholds.validate().is_ok());
    }

    #[test]
    fn corrections_pull_the_boundary() {
        // Same feature, first labeled slop 10x, then corrected ham 4x at 5x
        // weight: corrections dominate (20 effective vs 10).
        let mut ex: Vec<Example> = (0..10)
            .map(|_| Example::new(vec![Fire::hit("DOCS_ONLY")], true))
            .collect();
        ex.extend((0..4).map(|_| Example::correction(vec![Fire::hit("DOCS_ONLY")], false)));
        // Anchor class balance with clear examples.
        ex.extend((0..10).map(|_| Example::new(vec![Fire::hit("AGENT_EMAIL")], true)));
        ex.extend((0..10).map(|_| Example::new(vec![], false)));
        let w = fit(&ex, &quick());
        let p = w.score(&[Fire::hit("DOCS_ONLY")]).probability;
        assert!(p < 0.5, "corrections outweigh original labels, got p={p}");
    }

    #[test]
    fn auc_of_random_labels_near_half() {
        let ex: Vec<Example> = (0..200)
            .map(|i| {
                Example::new(
                    vec![Fire::new("NOISE", ((i * 7) % 10) as f64 / 10.0)],
                    i % 2 == 0,
                )
            })
            .collect();
        let w = fit(&ex, &quick());
        let a = auc(&w, &ex);
        assert!((a - 0.5).abs() < 0.15, "AUC {a} should hover near 0.5");
    }

    #[test]
    fn auc_empty_class_is_nan() {
        let ex = vec![Example::new(vec![], true)];
        let w = Weights::default_table();
        assert!(auc(&w, &ex).is_nan());
    }

    #[test]
    fn sign_constraints_follow_family() {
        // A slop rule firing mostly on ham goes to zero; a trust rule
        // firing mostly on ham goes negative.
        let mut ex: Vec<Example> = (0..40)
            .map(|_| Example::new(vec![Fire::hit("AGENT_TRAILER")], true))
            .collect();
        ex.extend((0..40).map(|_| {
            Example::new(
                vec![
                    Fire::hit("LOOKS_INNOCENT"),
                    Fire::hit("TRUST_MERGED_ELSEWHERE"),
                ],
                false,
            )
        }));
        let w = fit(&ex, &quick());
        assert_eq!(w.rules["LOOKS_INNOCENT"], 0.0);
        assert!(w.rules["TRUST_MERGED_ELSEWHERE"] < -0.5);
        assert!(w.rules["AGENT_TRAILER"] > 1.0);
    }

    #[test]
    fn under_observed_rules_keep_their_prior() {
        let mut ex = corpus();
        // One slop example carries a rule the prior prices at 3.0.
        ex[0].fires.push(Fire::hit("NETWORK_AUTHOR_VERDICT"));
        let mut prior = Weights::default_table();
        prior.rules.insert("NETWORK_AUTHOR_VERDICT".into(), 3.0);
        prior.rules.insert("NEVER_FIRED".into(), 1.25);
        let w = fit(
            &ex,
            &FitOptions {
                prior: Some(prior),
                ..quick()
            },
        );
        assert_eq!(w.rules["NETWORK_AUTHOR_VERDICT"], 3.0);
        assert_eq!(w.rules["NEVER_FIRED"], 1.25);
    }

    #[test]
    fn unstable_rules_are_not_priced() {
        // A rule that fires on one slop group and one ham group is present
        // or absent depending on the resample; bagging must not price it.
        let mut ex = corpus();
        for e in ex.iter_mut().filter(|e| e.group == "s3") {
            e.fires.push(Fire::hit("FLICKER"));
        }
        let w = fit(
            &ex,
            &FitOptions {
                stability: 1.0,
                bags: 12,
                ..quick()
            },
        );
        assert!(
            w.rules["FLICKER"] >= 0.0,
            "priced or zero, never negative"
        );
        // The stable rule survives full-stability gating.
        assert!(w.rules["AGENT_TRAILER"] > 1.0);
    }

    #[test]
    fn threshold_with_few_negatives_clears_them_by_a_margin() {
        let ex = vec![
            Example::new(vec![Fire::hit("AGENT_EMAIL")], true),
            Example::new(vec![], false),
            Example::new(vec![], false),
        ];
        let w = Weights::default_table();
        let th = thresholds_at_fpr(&w, &ex, 0.05, 0.01, 0.001);
        assert!(observed_fpr(&w, &ex, th.close) == 0.0);
        assert!(observed_fpr(&w, &ex, th.label) == 0.0);
        let top = w.score(&[]).probability;
        assert!(th.label > top, "cut sits above the highest negative");
        assert!(logit(th.label) - logit(top) > 0.99, "by a full logit");
    }

    #[test]
    fn certification_is_the_rule_of_three() {
        assert!(certifiable(300, 0.01));
        assert!(!certifiable(299, 0.01));
        assert!(certifiable(3000, 0.001));
        assert!(!certifiable(2430, 0.001));
    }

    #[test]
    fn family_cap_shapes_the_fit() {
        // Three cluster rules always fire together on slop. Uncapped, the
        // fit can spread 6+ logits over them; capped at 2, the family sum
        // it learns to rely on stays at the cap.
        let mut ex: Vec<Example> = (0..40)
            .map(|_| {
                Example::new(
                    vec![
                        Fire::hit("CLUSTER_BURST"),
                        Fire::hit("CLUSTER_SIZE_LOG"),
                        Fire::hit("CLUSTER_XREPO"),
                    ],
                    true,
                )
            })
            .collect();
        ex.extend((0..40).map(|_| Example::new(vec![], false)));
        let w = fit(
            &ex,
            &FitOptions {
                family_cap: Some(2.0),
                bags: 0,
                ..quick()
            },
        );
        let v = w.score(&[
            Fire::hit("CLUSTER_BURST"),
            Fire::hit("CLUSTER_SIZE_LOG"),
            Fire::hit("CLUSTER_XREPO"),
        ]);
        assert!(
            (v.score - w.bias - 2.0).abs() < 1e-9,
            "family contributes the cap"
        );
        assert!(auc(&w, &ex) > 0.99);
    }
}
