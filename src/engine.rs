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
//!
//! Rules belong to families (cluster, code, prose, shape, dossier, trust,
//! policy) named by prefix. A family's total contribution is capped, so no
//! single lane can carry a verdict to the close tier alone: a close needs
//! corroboration from at least two families. Policy rules (challenge
//! outcomes, network verdicts, the repo's AI stance) are decisive by
//! design and are never capped. `TRUST_` rules are the one family whose
//! weights are negative: they exonerate, and they are capped the same way
//! so a trusted account cannot launder an obvious campaign.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The lane a rule belongs to, by name prefix. Families are the unit of
/// the contribution cap in `Weights::score` and of the sign constraint in
/// the fit: `Trust` rules fit non-positive, every other family fits
/// non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    Cluster,
    Code,
    Prose,
    Shape,
    Dossier,
    Trust,
    /// Decisive by design: challenge outcomes, corroborated network
    /// verdicts, and the repo's own AI policy. Never capped.
    Policy,
}

impl Family {
    pub fn of(rule: &str) -> Family {
        const POLICY: &[&str] = &[
            "NETWORK_AUTHOR_VERDICT",
            "CANARY_EATEN",
            "CHALLENGE_TIMEOUT",
            "AI_FORBIDDEN",
            "UNDISCLOSED_AI",
        ];
        const CODE: &[&str] = &[
            "COMMENT_HEAVY",
            "DIFF_ENORMOUS",
            "WHITESPACE_ONLY",
            "AUTHORING_RATE",
        ];
        const PROSE: &[&str] = &["BODY_TOKEN_SCORE", "DETECTOR_SCORE"];
        if POLICY.contains(&rule) {
            Family::Policy
        } else if rule.starts_with("TRUST_") {
            Family::Trust
        } else if rule.starts_with("CLUSTER_") {
            Family::Cluster
        } else if rule.starts_with("CODE_") || CODE.contains(&rule) {
            Family::Code
        } else if rule.starts_with("STYLE_") || PROSE.contains(&rule) {
            Family::Prose
        } else if rule.starts_with("DOSSIER_")
            || rule.starts_with("ACCOUNT_")
            || rule.starts_with("AGENT_")
            || matches!(
                rule,
                "GENERATION_FOOTER"
                    | "VELOCITY_FORK_TO_PR"
                    | "REPLY_INSTANT"
                    | "HOUR_ENTROPY_FLAT"
                    | "UNRELATED_SPREAD"
            )
        {
            Family::Dossier
        } else {
            Family::Shape
        }
    }

    /// Number of families, for fixed-size accumulators.
    pub const COUNT: usize = 7;

    /// Dense index, for fixed-size accumulators.
    pub fn index(self) -> usize {
        match self {
            Family::Cluster => 0,
            Family::Code => 1,
            Family::Prose => 2,
            Family::Shape => 3,
            Family::Dossier => 4,
            Family::Trust => 5,
            Family::Policy => 6,
        }
    }

    /// Whether the family's weights are exonerating (fit non-positive).
    pub fn exonerating(self) -> bool {
        self == Family::Trust
    }

    /// Whether the family's contribution is subject to the cap.
    pub fn capped(self) -> bool {
        self != Family::Policy
    }
}

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Weights {
    pub bias: f64,
    pub rules: BTreeMap<String, f64>,
    pub thresholds: Thresholds,
    /// Largest absolute contribution one rule family may make to a score.
    /// None leaves families uncapped (tables fitted before the cap
    /// existed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_cap: Option<f64>,
    /// Provenance: corpus size, AUC, fit date. Informational only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
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

pub fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    (p / (1.0 - p)).ln()
}

impl Weights {
    pub fn score(&self, fires: &[Fire]) -> Verdict {
        let mut evidence: Vec<EvidenceItem> = fires
            .iter()
            .map(|f| {
                let value = f.value.clamp(0.0, 1.0);
                let weight = self.rules.get(&f.rule).copied().unwrap_or(0.0);
                EvidenceItem {
                    rule: f.rule.clone(),
                    value,
                    weight,
                    contribution: weight * value,
                }
            })
            .collect();
        // Family cap: scale a family's contributions down proportionally
        // when their sum exceeds the cap, so the evidence table still sums
        // to the score.
        if let Some(cap) = self.family_cap {
            let mut sums: BTreeMap<Family, f64> = BTreeMap::new();
            for e in &evidence {
                *sums.entry(Family::of(&e.rule)).or_default() += e.contribution;
            }
            for e in &mut evidence {
                let fam = Family::of(&e.rule);
                let sum = sums[&fam];
                if fam.capped() && sum.abs() > cap {
                    e.contribution *= cap / sum.abs();
                }
            }
        }
        let score = self.bias + evidence.iter().map(|e| e.contribution).sum::<f64>();
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

    /// Fit timestamp from the provenance meta; None when the table
    /// carries none.
    pub fn fitted_at(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.meta
            .as_ref()?
            .get("fitted_at")?
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|t| t.with_timezone(&chrono::Utc))
    }

    /// Whether this table is a strictly newer fit than `other`. A table
    /// without provenance is never newer, and any fitted table is newer
    /// than one without provenance.
    pub fn newer_fit_than(&self, other: &Weights) -> bool {
        match (self.fitted_at(), other.fitted_at()) {
            (Some(a), Some(b)) => a > b,
            (Some(_), None) => true,
            (None, _) => false,
        }
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
            family_cap: None,
            meta: None,
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
        assert!(
            Thresholds {
                label: 0.3,
                hold: 0.7,
                close: 0.95
            }
            .validate()
            .is_ok()
        );
        assert!(
            Thresholds {
                label: 0.8,
                hold: 0.7,
                close: 0.95
            }
            .validate()
            .is_err()
        );
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
    fn family_cap_bounds_one_lane_and_spares_policy() {
        let mut t = table();
        t.rules.insert("CLUSTER_SIZE_LOG".into(), 3.0);
        t.rules.insert("CANARY_EATEN".into(), 8.0);
        t.rules.insert("TRUST_MERGED_ELSEWHERE".into(), -6.0);
        t.family_cap = Some(4.0);
        // Two cluster rules sum to 7 but the family contributes 4.
        let v = t.score(&[Fire::hit("CLUSTER_BURST"), Fire::hit("CLUSTER_SIZE_LOG")]);
        assert!(
            (v.score - (-4.0 + 4.0)).abs() < 1e-9,
            "capped score {}",
            v.score
        );
        let shown: f64 = v.evidence.iter().map(|e| e.contribution).sum();
        assert!(
            (shown - 4.0).abs() < 1e-9,
            "evidence sums to the capped score"
        );
        // Policy rules are never capped.
        let v = t.score(&[Fire::hit("CANARY_EATEN")]);
        assert!((v.score - 4.0).abs() < 1e-9);
        // Trust is capped symmetrically.
        let v = t.score(&[Fire::hit("TRUST_MERGED_ELSEWHERE")]);
        assert!(
            (v.score - (-8.0)).abs() < 1e-9,
            "trust capped at -4: {}",
            v.score
        );
        // Without a cap, nothing changes.
        t.family_cap = None;
        let v = t.score(&[Fire::hit("CLUSTER_BURST"), Fire::hit("CLUSTER_SIZE_LOG")]);
        assert!((v.score - 3.0).abs() < 1e-9);
    }

    #[test]
    fn families_by_prefix() {
        assert_eq!(Family::of("CLUSTER_BURST"), Family::Cluster);
        assert_eq!(Family::of("CODE_DUP_BLOCK"), Family::Code);
        assert_eq!(Family::of("DIFF_ENORMOUS"), Family::Code);
        assert_eq!(Family::of("STYLE_EMOJI"), Family::Prose);
        assert_eq!(Family::of("BODY_TOKEN_SCORE"), Family::Prose);
        assert_eq!(Family::of("BODY_SCAFFOLD"), Family::Shape);
        assert_eq!(Family::of("TITLE_UPDATE_FILE"), Family::Shape);
        assert_eq!(Family::of("ACCOUNT_NEW"), Family::Dossier);
        assert_eq!(Family::of("AGENT_EMAIL"), Family::Dossier);
        assert_eq!(Family::of("TRUST_ACCOUNT_AGE"), Family::Trust);
        assert!(Family::of("TRUST_ACCOUNT_AGE").exonerating());
        assert_eq!(Family::of("CANARY_EATEN"), Family::Policy);
        assert!(!Family::of("CANARY_EATEN").capped());
        assert_eq!(Family::of("SOME_FUTURE_RULE"), Family::Shape);
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
    fn newer_fit_wins_and_missing_provenance_never_does() {
        let stamped = |at: &str| Weights {
            meta: Some(serde_json::json!({ "fitted_at": at })),
            ..table()
        };
        let old = stamped("2026-08-06T21:24:28Z");
        let new = stamped("2026-08-10T00:00:25Z");
        assert!(new.newer_fit_than(&old));
        assert!(!old.newer_fit_than(&new));
        assert!(!old.newer_fit_than(&old));
        let bare = table();
        assert!(old.newer_fit_than(&bare));
        assert!(!bare.newer_fit_than(&old));
        assert!(!bare.newer_fit_than(&bare));
        assert!(Weights::default_table().fitted_at().is_some());
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
