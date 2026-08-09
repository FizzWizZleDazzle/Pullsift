//! Enforcement: a verdict becomes a planned action; dry-run turns any plan
//! into an annotation. Comment templates live here, ASCII only.

use crate::engine::{EvidenceItem, Tier, Verdict};
use serde::{Deserialize, Serialize};

pub const SUSPECT_LABEL: &str = "slop-suspect";

/// Hidden marker in the score comment; rescores find and edit the existing
/// comment instead of stacking a new one per push.
pub const SCORE_COMMENT_MARKER: &str = "<!-- pullsift-score -->";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlannedAction {
    /// Exempt, or Pass with score comments off: do nothing at all.
    None,
    /// Pass: the score comment alone, no label, no enforcement.
    Comment { evidence_comment: String },
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
/// `score_comments` gives a passing PR the score comment instead of silence.
pub fn plan(
    verdict: &Verdict,
    challenge_comment: Option<String>,
    dry_run: bool,
    score_comments: bool,
) -> PlannedAction {
    let action = match verdict.tier {
        Tier::Pass if score_comments => PlannedAction::Comment {
            evidence_comment: pass_comment(verdict),
        },
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
            // A pass has no action to hold back; the score comment stands.
            PlannedAction::None => PlannedAction::None,
            PlannedAction::Comment { .. } => action,
            other => PlannedAction::Label {
                evidence_comment: format!(
                    "Dry run: Pullsift would have taken action \
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
        PlannedAction::Comment { .. } => "comment",
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
        "<details><summary>Pullsift evidence (probability {:.3})</summary>\n\n",
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

/// The score comment for a passing PR: verdict first, evidence collapsed.
fn pass_comment(verdict: &Verdict) -> String {
    format!(
        "{SCORE_COMMENT_MARKER}\nPullsift scored this pull request \
         {:.3}, below every action threshold. No action taken.\n\n{}",
        verdict.probability,
        evidence_comment(verdict)
    )
}

fn close_comment(verdict: &Verdict) -> String {
    let top: Vec<&EvidenceItem> = verdict.evidence.iter().take(3).collect();
    let mut reasons = String::new();
    for e in top {
        reasons.push_str(&format!("- {}\n", e.rule));
    }
    format!(
        "This pull request is closed by Pullsift because its signals put \
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
            meta: None,
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
        assert!(matches!(
            plan(&verdict(Tier::Pass), None, false, true),
            PlannedAction::Comment { .. }
        ));
        assert!(matches!(
            plan(&verdict(Tier::Label), None, false, true),
            PlannedAction::Label { .. }
        ));
        assert!(matches!(
            plan(&verdict(Tier::Hold), None, false, true),
            PlannedAction::Hold { .. }
        ));
        assert!(matches!(
            plan(&verdict(Tier::Close), None, false, true),
            PlannedAction::Close { .. }
        ));
    }

    #[test]
    fn pass_comments_by_default_and_opting_out_silences_it() {
        let PlannedAction::Comment { evidence_comment } =
            plan(&verdict(Tier::Pass), None, false, true)
        else {
            panic!("pass must carry the score comment")
        };
        assert!(evidence_comment.contains(SCORE_COMMENT_MARKER));
        assert!(evidence_comment.contains("No action taken"));
        assert!(evidence_comment.contains("probability"));
        assert_eq!(
            plan(&verdict(Tier::Pass), None, false, false),
            PlannedAction::None
        );
    }

    #[test]
    fn dry_run_downgrades_everything_to_annotation() {
        let a = plan(&verdict(Tier::Close), None, true, true);
        let PlannedAction::Label { evidence_comment } = a else {
            panic!("dry run must annotate, not act")
        };
        assert!(evidence_comment.contains("Dry run"));
        assert!(evidence_comment.contains("`close`"));
        // The pass score comment survives dry run; silence survives opt-out.
        assert!(matches!(
            plan(&verdict(Tier::Pass), None, true, true),
            PlannedAction::Comment { .. }
        ));
        assert_eq!(
            plan(&verdict(Tier::Pass), None, true, false),
            PlannedAction::None
        );
    }

    #[test]
    fn hold_carries_challenge_when_provided() {
        let a = plan(
            &verdict(Tier::Hold),
            Some("challenge text".into()),
            false,
            true,
        );
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
        let PlannedAction::Close { comment } = plan(&verdict(Tier::Close), None, false, true)
        else {
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
