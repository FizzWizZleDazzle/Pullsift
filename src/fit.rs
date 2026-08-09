//! Weight fitting: plain logistic regression by gradient descent, AUC, and
//! tier thresholds chosen at fixed false-positive rates.
//!
//! No ML framework; the rule vectors are small and the corpus fits in
//! memory. Maintainer corrections enter as sample weights.

use crate::engine::{Fire, Thresholds, Weights, sigmoid};
use std::collections::BTreeMap;

/// A labeled example: the rules that fired, whether it was slop, and a
/// sample weight (corrections are up-weighted).
#[derive(Debug, Clone)]
pub struct Example {
    pub fires: Vec<Fire>,
    pub is_slop: bool,
    pub sample_weight: f64,
}

impl Example {
    pub fn new(fires: Vec<Fire>, is_slop: bool) -> Self {
        Self {
            fires,
            is_slop,
            sample_weight: 1.0,
        }
    }

    /// A maintainer override of one of our verdicts; weighted 5x.
    pub fn correction(fires: Vec<Fire>, is_slop: bool) -> Self {
        Self {
            fires,
            is_slop,
            sample_weight: 5.0,
        }
    }
}

pub struct FitOptions {
    pub learning_rate: f64,
    pub iterations: usize,
    pub l2: f64,
    /// Constrain rule weights to be non-negative (projected gradient).
    /// Every rule is designed as a slop indicator; a negative fitted weight
    /// means a corpus artifact, and the constraint sends it to zero instead
    /// of letting it exonerate.
    pub non_negative: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.5,
            iterations: 4000,
            l2: 1e-3,
            non_negative: true,
        }
    }
}

/// Fit weights on examples. Rule universe is the union of all fired rules.
/// Returns a full `Weights` with thresholds set at the FPR targets below.
pub fn fit(examples: &[Example], opts: &FitOptions) -> Weights {
    let mut rule_ix: BTreeMap<String, usize> = BTreeMap::new();
    for ex in examples {
        for f in &ex.fires {
            let next = rule_ix.len();
            rule_ix.entry(f.rule.clone()).or_insert(next);
        }
    }
    let dim = rule_ix.len();

    // Dense rows.
    let rows: Vec<(Vec<f64>, f64, f64)> = examples
        .iter()
        .map(|ex| {
            let mut x = vec![0.0; dim];
            for f in &ex.fires {
                x[rule_ix[&f.rule]] = f.value.clamp(0.0, 1.0);
            }
            (x, if ex.is_slop { 1.0 } else { 0.0 }, ex.sample_weight)
        })
        .collect();
    let total_w: f64 = rows.iter().map(|r| r.2).sum::<f64>().max(1e-9);

    let mut w = vec![0.0; dim];
    let mut bias = 0.0;
    for _ in 0..opts.iterations {
        let mut gw = vec![0.0; dim];
        let mut gb = 0.0;
        for (x, y, sw) in &rows {
            let z = bias + dot(&w, x);
            let err = (sigmoid(z) - y) * sw;
            gb += err;
            for (gi, xi) in gw.iter_mut().zip(x) {
                *gi += err * xi;
            }
        }
        bias -= opts.learning_rate * gb / total_w;
        for i in 0..dim {
            let grad = gw[i] / total_w + opts.l2 * w[i];
            w[i] -= opts.learning_rate * grad;
            if opts.non_negative && w[i] < 0.0 {
                w[i] = 0.0;
            }
        }
    }

    let mut rules = BTreeMap::new();
    for (rule, ix) in &rule_ix {
        rules.insert(rule.clone(), w[*ix]);
    }
    let mut weights = Weights {
        bias,
        rules,
        thresholds: Thresholds {
            label: 0.3,
            hold: 0.7,
            close: 0.95,
        },
        meta: None,
    };
    weights.thresholds = thresholds_at_fpr(&weights, examples, 0.05, 0.01, 0.001);
    weights
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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
    let pos: Vec<f64> = scored.iter().filter(|s| s.1).map(|s| s.0).collect();
    let neg: Vec<f64> = scored.iter().filter(|s| !s.1).map(|s| s.0).collect();
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let mut wins = 0.0;
    for p in &pos {
        for n in &neg {
            if p > n {
                wins += 1.0;
            } else if p == n {
                wins += 0.5;
            }
        }
    }
    wins / (pos.len() as f64 * neg.len() as f64)
}

/// Pick tier thresholds so the observed FPR on `examples` (ideally held-out)
/// stays at or below each target. With too few negatives to certify a rate,
/// the threshold clears every negative.
pub fn thresholds_at_fpr(
    weights: &Weights,
    examples: &[Example],
    label_fpr: f64,
    hold_fpr: f64,
    close_fpr: f64,
) -> Thresholds {
    let mut neg: Vec<f64> = probabilities(weights, examples)
        .into_iter()
        .filter(|s| !s.1)
        .map(|s| s.0)
        .collect();
    neg.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let cut = |target: f64| -> f64 {
        // Highest threshold t such that (# negatives >= t) / n <= target.
        let n = neg.len();
        let allowed = (target * n as f64).floor() as usize;
        if allowed == 0 || n == 0 {
            // Clear every negative seen.
            neg.first().map(|m| (m + 1e-9).min(1.0)).unwrap_or(0.999)
        } else {
            // Sit just above the (allowed+1)-th highest negative.
            (neg[allowed] + 1e-9).min(1.0)
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
            ex.push(Example::new(fires, true));
        }
        for i in 0..40 {
            let mut fires = vec![];
            if i % 2 == 0 {
                fires.push(Fire::hit("ACCOUNT_NEW"));
            }
            if i % 5 == 0 {
                fires.push(Fire::new("STYLE_EMOJI", 0.2));
            }
            ex.push(Example::new(fires, false));
        }
        ex
    }

    #[test]
    fn fit_separates_separable_corpus() {
        let ex = corpus();
        let w = fit(&ex, &FitOptions::default());
        let a = auc(&w, &ex);
        assert!(a > 0.99, "AUC {a} on separable data");
        // The discriminative rule gets a positive weight...
        assert!(w.rules["AGENT_TRAILER"] > 1.0);
        // ...and newness, present in both classes, stays small.
        assert!(w.rules["ACCOUNT_NEW"].abs() < w.rules["AGENT_TRAILER"]);
    }

    #[test]
    fn fitted_thresholds_hold_their_fpr_in_sample() {
        let ex = corpus();
        let w = fit(&ex, &FitOptions::default());
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
        let w = fit(&ex, &FitOptions::default());
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
        let w = fit(&ex, &FitOptions::default());
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
    fn non_negative_fit_zeroes_ham_correlated_rules() {
        // A rule firing mostly on ham would fit negative; the constraint
        // sends it to zero instead.
        let mut ex: Vec<Example> = (0..40)
            .map(|_| Example::new(vec![Fire::hit("AGENT_TRAILER")], true))
            .collect();
        ex.extend((0..40).map(|_| Example::new(vec![Fire::hit("LOOKS_INNOCENT")], false)));
        let w = fit(&ex, &FitOptions::default());
        assert_eq!(w.rules["LOOKS_INNOCENT"], 0.0);
        assert!(w.rules["AGENT_TRAILER"] > 1.0);
        // Unconstrained: the same rule goes negative.
        let w2 = fit(
            &ex,
            &FitOptions {
                non_negative: false,
                ..Default::default()
            },
        );
        assert!(w2.rules["LOOKS_INNOCENT"] < -0.5);
    }

    #[test]
    fn threshold_with_few_negatives_clears_them_all() {
        let ex = vec![
            Example::new(vec![Fire::hit("AGENT_EMAIL")], true),
            Example::new(vec![], false),
            Example::new(vec![], false),
        ];
        let w = Weights::default_table();
        let th = thresholds_at_fpr(&w, &ex, 0.05, 0.01, 0.001);
        assert!(observed_fpr(&w, &ex, th.close) == 0.0);
        assert!(observed_fpr(&w, &ex, th.label) == 0.0);
    }
}
