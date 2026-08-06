//! Calibrated log-odds rule engine.
//!
//! Every detection signal is a rule that emits a value in [0,1]. Each rule
//! has a weight interpreted as a log-likelihood ratio. A PR's raw score is
//! `bias + sum(w_i * x_i)`; its slop probability is `sigmoid(score)`. Tiers
//! are probability thresholds chosen at fixed false-positive rates on a
//! held-out corpus (see `fit`).
//!
//! Rules unknown to the weight table score zero but are still logged, so new
//! rules can ship dark and get priced at the next fit.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A rule that fired for a PR, with its value in [0,1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fire {
    pub rule: String,
    pub value: f64,
}

impl Fire {
    pub fn new(rule: &str, value: f64) -> Self {
        Self {
            rule: rule.to_string(),
            value: value.clamp(0.0, 1.0),
        }
    }

    /// A binary rule at full strength.
    pub fn hit(rule: &str) -> Self {
        Self::new(rule, 1.0)
    }
}

/// Enforcement tier, ordered by severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tier {
    Pass,
    /// T1: label and annotate.
    Label,
    /// T2: hold as draft, digest instead of notifications.
    Hold,
    /// T3: close with evidence and an appeal path.
    Close,
}

/// Probability thresholds per tier. `label < hold < close` must hold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    pub label: f64,
    pub hold: f64,
    pub close: f64,
}

impl Thresholds {
    pub fn validate(&self) -> Result<(), String> {
        let ordered = self.label < self.hold && self.hold < self.close;
        let in_range = [self.label, self.hold, self.close]
            .iter()
            .all(|p| (0.0..=1.0).contains(p));
        if ordered && in_range {
            Ok(())
        } else {
            Err(format!("invalid thresholds: {self:?}"))
        }
    }

    pub fn tier(&self, probability: f64) -> Tier {
        if probability >= self.close {
            Tier::Close
        } else if probability >= self.hold {
            Tier::Hold
        } else if probability >= self.label {
            Tier::Label
        } else {
            Tier::Pass
        }
    }
}

/// The weight table: bias, per-rule weights, tier thresholds. Serialized as
/// JSON and shipped as data, not code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weights {
    pub bias: f64,
    pub rules: BTreeMap<String, f64>,
    pub thresholds: Thresholds,
}

/// One line of evidence in a verdict: what fired, at what weight, and its
/// contribution to the score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub rule: String,
    pub value: f64,
    pub weight: f64,
    pub contribution: f64,
}

/// A scored PR. Every verdict carries its full evidence; nothing is ever
/// unexplained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub score: f64,
    pub probability: f64,
    pub tier: Tier,
    pub evidence: Vec<EvidenceItem>,
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Weights {
    pub fn score(&self, fires: &[Fire]) -> Verdict {
        let mut score = self.bias;
        let mut evidence = Vec::with_capacity(fires.len());
        for f in fires {
            let value = f.value.clamp(0.0, 1.0);
            let weight = self.rules.get(&f.rule).copied().unwrap_or(0.0);
            let contribution = weight * value;
            score += contribution;
            evidence.push(EvidenceItem {
                rule: f.rule.clone(),
                value,
                weight,
                contribution,
            });
        }
        // Largest contributions first: the evidence list reads as "why".
        evidence.sort_by(|a, b| {
            b.contribution
                .abs()
                .partial_cmp(&a.contribution.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let probability = sigmoid(score);
        Verdict {
            score,
            probability,
            tier: self.thresholds.tier(probability),
            evidence,
        }
    }

    /// The default weight table embedded in the binary. Hand-set priors;
    /// replaced by fitted weights once the corpus exists.
    pub fn default_table() -> Self {
        serde_json::from_str(include_str!("../weights/default.json"))
            .expect("embedded default weights must parse")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Weights {
        Weights {
            bias: -4.0,
            rules: BTreeMap::from([
                ("AGENT_TRAILER".into(), 3.0),
                ("CLUSTER_BURST".into(), 4.0),
                ("ACCOUNT_NEW".into(), 0.5),
            ]),
            thresholds: Thresholds {
                label: 0.30,
                hold: 0.70,
                close: 0.95,
            },
        }
    }

    #[test]
    fn empty_fires_score_bias_only() {
        let v = table().score(&[]);
        assert_eq!(v.score, -4.0);
        assert!(v.probability < 0.02);
        assert_eq!(v.tier, Tier::Pass);
    }

    #[test]
    fn score_is_additive_and_monotonic() {
        let t = table();
        let one = t.score(&[Fire::hit("AGENT_TRAILER")]);
        let two = t.score(&[Fire::hit("AGENT_TRAILER"), Fire::hit("CLUSTER_BURST")]);
        assert!(two.score > one.score);
        assert!(two.probability > one.probability);
        assert_eq!(two.score, -4.0 + 3.0 + 4.0);
    }

    #[test]
    fn values_are_clamped() {
        let t = table();
        let v = t.score(&[Fire::new("AGENT_TRAILER", 7.0)]);
        assert_eq!(v.evidence[0].value, 1.0);
        let v = t.score(&[Fire::new("AGENT_TRAILER", -3.0)]);
        assert_eq!(v.evidence[0].value, 0.0);
    }

    #[test]
    fn unknown_rules_score_zero_but_are_logged() {
        let t = table();
        let v = t.score(&[Fire::hit("SOME_FUTURE_RULE")]);
        assert_eq!(v.score, -4.0);
        assert_eq!(v.evidence.len(), 1);
        assert_eq!(v.evidence[0].weight, 0.0);
    }

    #[test]
    fn tier_boundaries() {
        let th = Thresholds {
            label: 0.3,
            hold: 0.7,
            close: 0.95,
        };
        assert_eq!(th.tier(0.29), Tier::Pass);
        assert_eq!(th.tier(0.30), Tier::Label);
        assert_eq!(th.tier(0.70), Tier::Hold);
        assert_eq!(th.tier(0.95), Tier::Close);
        assert_eq!(th.tier(1.0), Tier::Close);
    }

    #[test]
    fn thresholds_validate_ordering() {
        assert!(Thresholds {
            label: 0.3,
            hold: 0.7,
            close: 0.95
        }
        .validate()
        .is_ok());
        assert!(Thresholds {
            label: 0.8,
            hold: 0.7,
            close: 0.95
        }
        .validate()
        .is_err());
    }

    #[test]
    fn evidence_sorted_by_contribution() {
        let t = table();
        let v = t.score(&[
            Fire::new("ACCOUNT_NEW", 1.0),
            Fire::hit("CLUSTER_BURST"),
            Fire::hit("AGENT_TRAILER"),
        ]);
        assert_eq!(v.evidence[0].rule, "CLUSTER_BURST");
        assert_eq!(v.evidence[1].rule, "AGENT_TRAILER");
        assert_eq!(v.evidence[2].rule, "ACCOUNT_NEW");
    }

    #[test]
    fn sigmoid_sanity() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(10.0) > 0.9999);
        assert!(sigmoid(-10.0) < 0.0001);
    }

    #[test]
    fn default_table_parses_and_validates() {
        let t = Weights::default_table();
        assert!(t.thresholds.validate().is_ok());
        assert!(!t.rules.is_empty());
        assert!(t.bias < 0.0, "prior must favor pass");
    }

    #[test]
    fn weights_serde_roundtrip() {
        let t = table();
        let s = serde_json::to_string(&t).unwrap();
        let back: Weights = serde_json::from_str(&s).unwrap();
        assert_eq!(back.rules, t.rules);
        assert_eq!(back.bias, t.bias);
    }
}
