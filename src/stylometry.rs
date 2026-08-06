//! Stylometry: a fixed-length feature vector over PR prose.
//!
//! Used two ways: per-PR rules fed to the engine at full weight (owner
//! decision), and pairwise as a cluster cohesion feature, where shared
//! unusual phrasing binds a campaign cluster tighter.

use crate::engine::Fire;
use serde::{Deserialize, Serialize};

pub const DIM: usize = 7;

/// Phrases statistically overrepresented in generated PR prose.
const AI_PHRASES: &[&str] = &[
    "this pr introduces",
    "this pull request introduces",
    "enhances the overall",
    "comprehensive",
    "seamless",
    "leverage",
    "leverages",
    "streamline",
    "streamlines",
    "delve",
    "furthermore",
    "moreover",
    "it is worth noting",
    "it's worth noting",
    "in conclusion",
    "dive into",
    "robust and maintainable",
    "best practices",
    "i hope this helps",
    "as an ai",
    "happy to iterate",
    "let me know if",
];

/// Unicode punctuation substituted by generators and word processors.
const UNICODE_PUNCT: &[char] = &[
    '\u{2018}', '\u{2019}', '\u{201c}', '\u{201d}', '\u{2026}', '\u{2192}', '\u{21d2}',
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StyleVector {
    /// Em dashes (U+2014) plus " -- " substitutes, per 100 words.
    pub em_dash: f64,
    /// Curly quotes, ellipsis, arrows, per 100 chars.
    pub unicode_punct: f64,
    /// Emoji per 100 chars.
    pub emoji: f64,
    /// Non-ASCII chars (excluding the above) per 100 chars.
    pub nonascii: f64,
    /// AI-phrase lexicon hits per 100 words.
    pub ai_phrases: f64,
    /// Markdown structure density: headers + bold-runs + bullet lines per line.
    pub structure: f64,
    /// Words per sentence, normalized.
    pub sentence_len: f64,
}

fn is_emoji(c: char) -> bool {
    matches!(u32::from(c),
        0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x1F1E6..=0x1F1FF | 0xFE0F | 0x2B00..=0x2BFF)
}

fn rate(count: usize, denom: usize, per: f64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        count as f64 * per / denom as f64
    }
}

pub fn analyze(text: &str) -> StyleVector {
    let chars: Vec<char> = text.chars().collect();
    let nchars = chars.len();
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    let nwords = words.len();

    let em = chars.iter().filter(|&&c| c == '\u{2014}').count() + lower.matches(" -- ").count();
    let upunct = chars.iter().filter(|c| UNICODE_PUNCT.contains(c)).count();
    let emoji = chars.iter().filter(|&&c| is_emoji(c)).count();
    let nonascii = chars
        .iter()
        .filter(|&&c| {
            !c.is_ascii() && c != '\u{2014}' && !UNICODE_PUNCT.contains(&c) && !is_emoji(c)
        })
        .count();
    let phrases: usize = AI_PHRASES.iter().map(|p| lower.matches(p).count()).sum();

    let lines: Vec<&str> = text.lines().collect();
    let structured = lines
        .iter()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with('#') || t.starts_with("- ") || t.starts_with("* ") || t.starts_with("**")
        })
        .count();
    let structure = if lines.is_empty() {
        0.0
    } else {
        structured as f64 / lines.len() as f64
    };

    let sentences = chars
        .iter()
        .filter(|&&c| c == '.' || c == '!' || c == '?')
        .count()
        .max(1);
    let sentence_len = (nwords as f64 / sentences as f64 / 40.0).min(1.0);

    StyleVector {
        em_dash: rate(em, nwords, 100.0).min(10.0),
        unicode_punct: rate(upunct, nchars, 100.0).min(10.0),
        emoji: rate(emoji, nchars, 100.0).min(10.0),
        nonascii: rate(nonascii, nchars, 100.0).min(100.0),
        ai_phrases: rate(phrases, nwords, 100.0).min(10.0),
        structure,
        sentence_len,
    }
}

impl StyleVector {
    pub fn to_array(&self) -> [f64; DIM] {
        [
            self.em_dash,
            self.unicode_punct,
            self.emoji,
            self.nonascii,
            self.ai_phrases,
            self.structure,
            self.sentence_len,
        ]
    }

    pub fn cosine(&self, other: &StyleVector) -> f64 {
        let a = self.to_array();
        let b = other.to_array();
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    pub fn mean(vectors: &[StyleVector]) -> Option<StyleVector> {
        if vectors.is_empty() {
            return None;
        }
        let mut acc = [0.0; DIM];
        for v in vectors {
            for (a, x) in acc.iter_mut().zip(v.to_array()) {
                *a += x;
            }
        }
        let n = vectors.len() as f64;
        Some(StyleVector {
            em_dash: acc[0] / n,
            unicode_punct: acc[1] / n,
            emoji: acc[2] / n,
            nonascii: acc[3] / n,
            ai_phrases: acc[4] / n,
            structure: acc[5] / n,
            sentence_len: acc[6] / n,
        })
    }

    /// Per-PR rules for the engine. Rates squashed so a value of 1.0 means
    /// "saturated", not "one occurrence".
    pub fn rules(&self) -> Vec<Fire> {
        let squash = |x: f64, scale: f64| (x / scale).min(1.0);
        let mut out = Vec::new();
        let mut push = |rule: &str, v: f64| {
            if v > 0.0 {
                out.push(Fire::new(rule, v));
            }
        };
        push("STYLE_EM_DASH", squash(self.em_dash, 2.0));
        push("STYLE_UNICODE_PUNCT", squash(self.unicode_punct, 2.0));
        push("STYLE_EMOJI", squash(self.emoji, 2.0));
        push("STYLE_NONASCII", squash(self.nonascii, 20.0));
        push("STYLE_AI_PHRASES", squash(self.ai_phrases, 3.0));
        push("STYLE_STRUCTURE", self.structure);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: &str = "Fixes the null check in parse_config. The old code \
        dereferenced the pointer before checking it. Added a regression test.";

    const SLOPPY: &str = "\u{1F680} This PR introduces a comprehensive \
        enhancement \u{2014} it leverages best practices to streamline the \
        overall experience! \u{2728} Furthermore, it is worth noting that \
        this delivers seamless integration. I hope this helps \u{2014} let \
        me know if you'd like changes! \u{201c}Robust\u{201d} \u{2026}";

    #[test]
    fn plain_text_scores_near_zero() {
        let v = analyze(PLAIN);
        assert_eq!(v.em_dash, 0.0);
        assert_eq!(v.emoji, 0.0);
        assert_eq!(v.ai_phrases, 0.0);
        assert!(v.rules().len() <= 2);
    }

    #[test]
    fn slop_text_lights_up() {
        let v = analyze(SLOPPY);
        assert!(v.em_dash > 0.0, "em dashes");
        assert!(v.emoji > 0.0, "emoji");
        assert!(v.unicode_punct > 0.0, "curly quotes/ellipsis");
        assert!(v.ai_phrases > 0.0, "lexicon");
        let rules = v.rules();
        assert!(rules.iter().any(|f| f.rule == "STYLE_AI_PHRASES"));
        assert!(rules.iter().any(|f| f.rule == "STYLE_EM_DASH"));
    }

    #[test]
    fn double_hyphen_counts_as_em_dash() {
        let v = analyze("this is fine -- or is it -- who knows");
        assert!(v.em_dash > 0.0);
    }

    #[test]
    fn nonascii_counts_exclude_typography_and_emoji() {
        // Cyrillic text: nonascii high, but not emoji/punct.
        let v =
            analyze("\u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043c}\u{0438}\u{0440}");
        assert!(v.nonascii > 0.0);
        assert_eq!(v.emoji, 0.0);
        assert_eq!(v.unicode_punct, 0.0);
    }

    #[test]
    fn cosine_of_same_style_is_high() {
        let a = analyze(SLOPPY);
        let b = analyze(&SLOPPY.replace("comprehensive", "holistic"));
        assert!(a.cosine(&b) > 0.95);
        let c = analyze(PLAIN);
        assert!(a.cosine(&c) < a.cosine(&b));
    }

    #[test]
    fn mean_of_identical_vectors_is_identity() {
        let v = analyze(SLOPPY);
        let m = StyleVector::mean(&[v.clone(), v.clone()]).unwrap();
        assert!((m.em_dash - v.em_dash).abs() < 1e-12);
        assert!(StyleVector::mean(&[]).is_none());
    }

    #[test]
    fn empty_text_is_all_zero() {
        let v = analyze("");
        assert_eq!(v.to_array().iter().filter(|x| **x != 0.0).count(), 0);
        assert!(v.rules().is_empty());
    }

    #[test]
    fn markdown_structure_density() {
        let v = analyze("## Summary\n- one\n- two\n**Bold claim**\ntext");
        assert!(v.structure >= 0.6);
    }
}
