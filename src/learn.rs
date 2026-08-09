//! Automatic weight learning.
//!
//! A batch job (never live mutation) that turns stored outcomes into a
//! candidate weight table and promotes it only behind guardrails:
//!
//! - enough data, in both classes, in both splits;
//! - held-out AUC must not regress against the incumbent;
//! - tier thresholds are re-derived on held-out data at the fixed FPR
//!   targets.
//!
//! Labels come from the store: maintainer merges of PRs we flagged and
//! reopens of PRs we closed are corrections (5x weight); confirmations and
//! spam-labeled closes are ordinary examples. Every promoted table is a new
//! version row in the database; rollback is flipping the active row.

use crate::engine::Weights;
use crate::fit::{Example, FitOptions, auc, fit, thresholds_at_fpr};

/// Refuse to learn from less than this many examples.
pub const MIN_EXAMPLES: usize = 200;
/// Each class must have at least this many members overall.
pub const MIN_CLASS: usize = 40;
/// Candidate may not lose more than this much held-out AUC.
pub const AUC_SLACK: f64 = 0.005;

/// Every 5th example is held out. Callers must pass examples in a stable
/// order (the store orders by id) so the split is reproducible.
const HOLDOUT_MOD: usize = 5;

#[derive(Debug)]
pub struct LearnOutcome {
    pub promoted: bool,
    pub reason: String,
    pub candidate_auc: f64,
    pub incumbent_auc: f64,
    /// Present only when promoted.
    pub weights: Option<Weights>,
}

fn refuse(reason: &str) -> LearnOutcome {
    LearnOutcome {
        promoted: false,
        reason: reason.to_string(),
        candidate_auc: f64::NAN,
        incumbent_auc: f64::NAN,
        weights: None,
    }
}

pub fn learn(examples: &[Example], incumbent: &Weights) -> LearnOutcome {
    if examples.len() < MIN_EXAMPLES {
        return refuse("too few examples");
    }
    let pos = examples.iter().filter(|e| e.is_slop).count();
    let neg = examples.len() - pos;
    if pos < MIN_CLASS || neg < MIN_CLASS {
        return refuse("class imbalance: not enough of one label");
    }

    let (train, holdout): (Vec<Example>, Vec<Example>) = {
        let mut tr = Vec::new();
        let mut ho = Vec::new();
        for (i, ex) in examples.iter().enumerate() {
            if i % HOLDOUT_MOD == 0 {
                ho.push(ex.clone());
            } else {
                tr.push(ex.clone());
            }
        }
        (tr, ho)
    };
    let ho_pos = holdout.iter().filter(|e| e.is_slop).count();
    if ho_pos == 0 || ho_pos == holdout.len() {
        return refuse("holdout split lost a class");
    }

    let mut candidate = fit(&train, &FitOptions::default());
    candidate.thresholds = thresholds_at_fpr(&candidate, &holdout, 0.05, 0.01, 0.001);
    if candidate.thresholds.validate().is_err() {
        return refuse("candidate thresholds failed validation");
    }

    let candidate_auc = auc(&candidate, &holdout);
    let incumbent_auc = auc(incumbent, &holdout);
    if !candidate_auc.is_finite() {
        return refuse("candidate AUC not computable");
    }
    if incumbent_auc.is_finite() && candidate_auc < incumbent_auc - AUC_SLACK {
        return LearnOutcome {
            promoted: false,
            reason: format!(
                "held-out AUC regressed: candidate {candidate_auc:.4} vs incumbent {incumbent_auc:.4}"
            ),
            candidate_auc,
            incumbent_auc,
            weights: None,
        };
    }

    LearnOutcome {
        promoted: true,
        reason: format!("promoted: held-out AUC {candidate_auc:.4} (incumbent {incumbent_auc:.4})"),
        candidate_auc,
        incumbent_auc,
        weights: Some(candidate),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Fire;

    fn corpus(n: usize) -> Vec<Example> {
        let mut ex = Vec::new();
        for i in 0..n {
            if i % 2 == 0 {
                let mut fires = vec![Fire::hit("AGENT_TRAILER")];
                if i % 4 == 0 {
                    fires.push(Fire::hit("CLUSTER_BURST"));
                }
                ex.push(Example::new(fires, true));
            } else {
                let mut fires = vec![];
                if i % 3 == 0 {
                    fires.push(Fire::hit("ACCOUNT_NEW"));
                }
                ex.push(Example::new(fires, false));
            }
        }
        ex
    }

    #[test]
    fn promotes_on_good_corpus() {
        let out = learn(&corpus(400), &Weights::default_table());
        assert!(out.promoted, "{}", out.reason);
        let w = out.weights.unwrap();
        assert!(w.thresholds.validate().is_ok());
        assert!(out.candidate_auc > 0.99);
    }

    #[test]
    fn refuses_small_corpus() {
        let out = learn(&corpus(50), &Weights::default_table());
        assert!(!out.promoted);
        assert!(out.reason.contains("too few"));
    }

    #[test]
    fn refuses_single_class() {
        let ex: Vec<Example> = (0..300)
            .map(|_| Example::new(vec![Fire::hit("AGENT_TRAILER")], true))
            .collect();
        let out = learn(&ex, &Weights::default_table());
        assert!(!out.promoted);
        assert!(out.reason.contains("class imbalance"));
    }

    #[test]
    fn refuses_when_candidate_regresses() {
        // Labels carry no signal; the incumbent (a good hand table judged on
        // the same noise) can't be beaten meaningfully, and if the fitted
        // noise model comes out behind by more than the slack it must not be
        // promoted. Either way the promoted table must never lose AUC.
        let ex: Vec<Example> = (0..400)
            .map(|i| {
                Example::new(
                    vec![Fire::new("NOISE", ((i * 13) % 17) as f64 / 17.0)],
                    (i * 7) % 3 == 0,
                )
            })
            .collect();
        let out = learn(&ex, &Weights::default_table());
        if out.promoted {
            assert!(out.candidate_auc >= out.incumbent_auc - AUC_SLACK);
        } else {
            assert!(out.reason.contains("regressed") || out.reason.contains("thresholds"));
        }
    }

    #[test]
    fn split_is_deterministic() {
        let a = learn(&corpus(400), &Weights::default_table());
        let b = learn(&corpus(400), &Weights::default_table());
        assert_eq!(a.candidate_auc, b.candidate_auc);
        assert_eq!(a.weights.unwrap().rules, b.weights.unwrap().rules);
    }
}
