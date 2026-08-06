//! Enforcement: a verdict becomes a planned action; dry-run turns any plan
//! into an annotation. Comment templates live here, ASCII only.

use crate::engine::{EvidenceItem, Tier, Verdict};
use serde::{Deserialize, Serialize};

pub const SUSPECT_LABEL: &str = "slop-suspect";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlannedAction {
    /// Below every threshold, or exempt: do nothing at all.
    None,
    /// T1: label plus a collapsed evidence comment.
    Label { evidence_comment: String },
    /// T2: convert to draft, label, add to the daily digest.
    Hold {
        evidence_comment: String,
        challenge_comment: Option<String>,
    },
    /// T3: close with evidence and the appeal path.
    Close { comment: String },
    /// Lane C: close with a policy message, no slop accusation.
    CloseByPolicy { comment: String },
}

/// Decide the action for a scored PR. `challenge_comment` is included when
/// the repo runs challenges and the verdict landed in Hold.
pub fn plan(verdict: &Verdict, challenge_comment: Option<String>, dry_run: bool) -> PlannedAction {
    let action = match verdict.tier {
        Tier::Pass => PlannedAction::None,
        Tier::Label => PlannedAction::Label {
            evidence_comment: evidence_comment(verdict),
        },
        Tier::Hold => PlannedAction::Hold {
            evidence_comment: evidence_comment(verdict),
            challenge_comment,
        },
        Tier::Close => PlannedAction::Close {
            comment: close_comment(verdict),
        },
    };
    if dry_run {
        match action {
            PlannedAction::None => PlannedAction::None,
            other => PlannedAction::Label {
                evidence_comment: format!(
                    "Dry run: slopcatcher would have taken action \
                     `{}`.\n\n{}",
                    action_name(&other),
                    evidence_comment(verdict)
                ),
            },
        }
    } else {
        action
    }
}

pub fn action_name(a: &PlannedAction) -> &'static str {
    match a {
        PlannedAction::None => "none",
        PlannedAction::Label { .. } => "label",
        PlannedAction::Hold { .. } => "hold",
        PlannedAction::Close { .. } => "close",
        PlannedAction::CloseByPolicy { .. } => "close-by-policy",
    }
}

/// Collapsed evidence table: rule, value, weight, contribution.
pub fn evidence_comment(verdict: &Verdict) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "<details><summary>slopcatcher evidence (probability {:.3})</summary>\n\n",
        verdict.probability
    ));
    s.push_str("| rule | value | weight | contribution |\n|---|---|---|---|\n");
    for e in &verdict.evidence {
        s.push_str(&format!(
            "| {} | {:.2} | {:+.2} | {:+.2} |\n",
            e.rule, e.value, e.weight, e.contribution
        ));
    }
    s.push_str("\n</details>\n");
    s
}

fn close_comment(verdict: &Verdict) -> String {
    let top: Vec<&EvidenceItem> = verdict.evidence.iter().take(3).collect();
    let mut reasons = String::new();
    for e in top {
        reasons.push_str(&format!("- {}\n", e.rule));
    }
    format!(
        "This pull request is closed by slopcatcher because its signals put \
it past the review-worthiness threshold this repository configured.\n\n\
Main signals:\n{reasons}\n\
If this is a mistake, reply here and a maintainer can reopen it; a reply \
in your own words about what the change does is enough.\n\n\
{evidence}",
        evidence = evidence_comment(verdict)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Fire, Thresholds, Weights};
    use std::collections::BTreeMap;

    fn verdict(p_target: Tier) -> Verdict {
        let w = Weights {
            bias: -4.0,
            rules: BTreeMap::from([("AGENT_EMAIL".into(), 12.0)]),
            thresholds: Thresholds {
                label: 0.3,
                hold: 0.7,
                close: 0.95,
            },
        };
        match p_target {
            Tier::Pass => w.score(&[]),
            Tier::Label => w.score(&[Fire::new("AGENT_EMAIL", 0.31)]),
            Tier::Hold => w.score(&[Fire::new("AGENT_EMAIL", 0.42)]),
            Tier::Close => w.score(&[Fire::hit("AGENT_EMAIL")]),
        }
    }

    #[test]
    fn tiers_map_to_actions() {
        assert_eq!(plan(&verdict(Tier::Pass), None, false), PlannedAction::None);
        assert!(matches!(
            plan(&verdict(Tier::Label), None, false),
            PlannedAction::Label { .. }
        ));
        assert!(matches!(
            plan(&verdict(Tier::Hold), None, false),
            PlannedAction::Hold { .. }
        ));
        assert!(matches!(
            plan(&verdict(Tier::Close), None, false),
            PlannedAction::Close { .. }
        ));
    }

    #[test]
    fn dry_run_downgrades_everything_to_annotation() {
        let a = plan(&verdict(Tier::Close), None, true);
        let PlannedAction::Label { evidence_comment } = a else {
            panic!("dry run must annotate, not act")
        };
        assert!(evidence_comment.contains("Dry run"));
        assert!(evidence_comment.contains("`close`"));
        // Pass stays silent even in dry run.
        assert_eq!(plan(&verdict(Tier::Pass), None, true), PlannedAction::None);
    }

    #[test]
    fn hold_carries_challenge_when_provided() {
        let a = plan(&verdict(Tier::Hold), Some("challenge text".into()), false);
        let PlannedAction::Hold {
            challenge_comment, ..
        } = a
        else {
            panic!()
        };
        assert_eq!(challenge_comment.as_deref(), Some("challenge text"));
    }

    #[test]
    fn close_comment_names_signals_and_appeal() {
        let PlannedAction::Close { comment } = plan(&verdict(Tier::Close), None, false) else {
            panic!()
        };
        assert!(comment.contains("AGENT_EMAIL"));
        assert!(comment.contains("reopen"));
        assert!(comment.contains("<details>"));
    }

    #[test]
    fn evidence_comment_lists_every_fire() {
        let v = verdict(Tier::Close);
        let c = evidence_comment(&v);
        assert!(c.contains("| AGENT_EMAIL |"));
        assert!(c.contains("probability"));
    }
}
