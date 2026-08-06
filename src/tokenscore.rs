//! Learned token model over PR prose: naive Bayes log-likelihood ratios per
//! token, trained on the mined corpus by the tuner and shipped as data
//! (`weights/tokens.json`). This replaces hand-written phrase lists with a
//! lexicon the corpus actually supports, and re-learns at every refit.
//!
//! The whole model surfaces as a single engine rule (BODY_TOKEN_SCORE) whose
//! weight the fit prices like any other.

use crate::hashing::tokens;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Per-token log-likelihood ratio clamp: one token can never say more than
/// this on its own.
const LLR_CLAMP: f64 = 3.0;
/// Total evidence clamp before the sigmoid.
const SUM_CLAMP: f64 = 12.0;
/// A token must appear in this many documents overall to be kept.
const MIN_DOCS: usize = 3;
/// Table size cap: the strongest tokens by |llr|.
const MAX_TOKENS: usize = 400;
/// Score only when at least this many known tokens are present.
const MIN_MATCHES: usize = 3;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenTable {
    pub llr: BTreeMap<String, f64>,
}

impl TokenTable {
    pub fn embedded() -> Self {
        serde_json::from_str(include_str!("../weights/tokens.json"))
            .expect("embedded token table must parse")
    }

    pub fn is_empty(&self) -> bool {
        self.llr.is_empty()
    }

    /// Train from (text, is_slop) documents. Presence-based (a token counts
    /// once per document), Laplace-smoothed, clamped.
    pub fn train(docs: &[(String, bool)]) -> Self {
        let mut slop_docs = 0usize;
        let mut ham_docs = 0usize;
        let mut slop_count: BTreeMap<String, usize> = BTreeMap::new();
        let mut ham_count: BTreeMap<String, usize> = BTreeMap::new();
        for (text, is_slop) in docs {
            let unique: BTreeSet<String> = tokens(text).into_iter().collect();
            if *is_slop {
                slop_docs += 1;
            } else {
                ham_docs += 1;
            }
            for t in unique {
                let map = if *is_slop {
                    &mut slop_count
                } else {
                    &mut ham_count
                };
                *map.entry(t).or_default() += 1;
            }
        }
        if slop_docs == 0 || ham_docs == 0 {
            return Self::default();
        }

        let mut entries: Vec<(String, f64, usize)> = Vec::new();
        let all: BTreeSet<&String> = slop_count.keys().chain(ham_count.keys()).collect();
        for token in all {
            let s = slop_count.get(token).copied().unwrap_or(0);
            let h = ham_count.get(token).copied().unwrap_or(0);
            if s + h < MIN_DOCS {
                continue;
            }
            let p_s = (s as f64 + 1.0) / (slop_docs as f64 + 2.0);
            let p_h = (h as f64 + 1.0) / (ham_docs as f64 + 2.0);
            let llr = (p_s / p_h).ln().clamp(-LLR_CLAMP, LLR_CLAMP);
            entries.push((token.clone(), llr, s + h));
        }
        entries.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        entries.truncate(MAX_TOKENS);
        Self {
            llr: entries.into_iter().map(|(t, l, _)| (t, l)).collect(),
        }
    }

    /// Slop probability of a text under the token model, or None when the
    /// table is empty or too few known tokens appear.
    pub fn score(&self, text: &str) -> Option<f64> {
        if self.llr.is_empty() {
            return None;
        }
        let unique: BTreeSet<String> = tokens(text).into_iter().collect();
        let mut sum = 0.0;
        let mut matches = 0usize;
        for t in &unique {
            if let Some(l) = self.llr.get(t) {
                sum += l;
                matches += 1;
            }
        }
        if matches < MIN_MATCHES {
            return None;
        }
        Some(crate::engine::sigmoid(sum.clamp(-SUM_CLAMP, SUM_CLAMP)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<(String, bool)> {
        let mut docs = Vec::new();
        for i in 0..20 {
            docs.push((
                format!("this pr delivers seamless comprehensive robust enhancement number {i} kindly merge"),
                true,
            ));
            docs.push((
                format!("fix null check in parser regression test added case {i} covers the panic"),
                false,
            ));
        }
        docs
    }

    #[test]
    fn learns_separating_tokens() {
        let t = TokenTable::train(&corpus());
        assert!(!t.is_empty());
        assert!(t.llr["seamless"] > 1.0, "slop token positive llr");
        assert!(t.llr["regression"] < -1.0, "ham token negative llr");
        let slop_p = t
            .score("a seamless robust comprehensive change kindly")
            .unwrap();
        let ham_p = t
            .score("fix the parser panic with a regression test")
            .unwrap();
        assert!(slop_p > 0.9, "slop {slop_p}");
        assert!(ham_p < 0.1, "ham {ham_p}");
    }

    #[test]
    fn one_class_corpus_trains_empty() {
        let docs: Vec<(String, bool)> = (0..10)
            .map(|i| (format!("text {i} words here now"), true))
            .collect();
        assert!(TokenTable::train(&docs).is_empty());
    }

    #[test]
    fn empty_table_and_thin_text_score_none() {
        let empty = TokenTable::default();
        assert!(empty.score("anything at all").is_none());
        let t = TokenTable::train(&corpus());
        assert!(t.score("").is_none());
        assert!(t.score("unrelated vocabulary entirely").is_none());
    }

    #[test]
    fn rare_tokens_are_dropped() {
        let mut docs = corpus();
        docs.push(("supercalifragilistic".into(), true));
        let t = TokenTable::train(&docs);
        assert!(!t.llr.contains_key("supercalifragilistic"));
    }

    #[test]
    fn llr_is_clamped() {
        let t = TokenTable::train(&corpus());
        assert!(t.llr.values().all(|l| l.abs() <= LLR_CLAMP + 1e-9));
    }

    #[test]
    fn serde_roundtrip_and_embedded_parse() {
        let t = TokenTable::train(&corpus());
        let s = serde_json::to_string(&t).unwrap();
        let back: TokenTable = serde_json::from_str(&s).unwrap();
        assert_eq!(back.llr, t.llr);
        let _ = TokenTable::embedded(); // must parse
    }
}
