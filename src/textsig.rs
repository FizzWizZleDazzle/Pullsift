//! Text signatures: minhash over title+body shingles, with banded LSH keys,
//! for near-duplicate prose even when the diffs differ.

use crate::hashing::{fnv1a64, shingle_hashes, splitmix64, tokens};

pub const PERMS: usize = 128;
pub const BANDS: usize = 16;
pub const ROWS: usize = PERMS / BANDS; // 8

const SHINGLE: usize = 3;
const FAMILY_SEED: u64 = 0x5107_CA7C_4E12_0001;

/// Estimated Jaccard similarity at which two texts count as near-duplicates.
pub const NEAR_JACCARD: f64 = 0.7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHash {
    pub mins: [u64; PERMS],
}

/// Minhash sketch of a text. None when the text is too short to shingle.
pub fn minhash(text: &str) -> Option<MinHash> {
    let toks = tokens(text);
    let shingles = shingle_hashes(&toks, SHINGLE);
    if shingles.len() < 3 {
        return None;
    }
    let mut mins = [u64::MAX; PERMS];
    let mut seed = FAMILY_SEED;
    for slot in mins.iter_mut() {
        seed = splitmix64(seed);
        for &sh in &shingles {
            let h = splitmix64(sh ^ seed);
            if h < *slot {
                *slot = h;
            }
        }
    }
    Some(MinHash { mins })
}

impl MinHash {
    /// Estimated Jaccard similarity: fraction of matching permutation slots.
    pub fn jaccard(&self, other: &MinHash) -> f64 {
        let eq = self
            .mins
            .iter()
            .zip(other.mins.iter())
            .filter(|(a, b)| a == b)
            .count();
        eq as f64 / PERMS as f64
    }

    /// 16 banded LSH keys; two texts sharing any key are candidates.
    pub fn band_keys(&self) -> [u64; BANDS] {
        let mut keys = [0u64; BANDS];
        for (band, key) in keys.iter_mut().enumerate() {
            let start = band * ROWS;
            let mut buf = Vec::with_capacity(ROWS * 8 + 1);
            buf.push(band as u8);
            for m in &self.mins[start..start + ROWS] {
                buf.extend_from_slice(&m.to_le_bytes());
            }
            *key = fnv1a64(&buf);
        }
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: &str = "This PR updates the README to improve the documentation \
        and fix a small typo in the installation instructions for new users.";

    #[test]
    fn identical_texts_have_jaccard_one() {
        let a = minhash(BODY).unwrap();
        let b = minhash(BODY).unwrap();
        assert_eq!(a.jaccard(&b), 1.0);
        assert_eq!(a.band_keys(), b.band_keys());
    }

    #[test]
    fn near_duplicates_score_high() {
        let tweaked = BODY.replace("small typo", "minor typo");
        let a = minhash(BODY).unwrap();
        let b = minhash(&tweaked).unwrap();
        let j = a.jaccard(&b);
        assert!(j >= NEAR_JACCARD, "jaccard {j}");
        // Banding: near-duplicates share at least one band key.
        let shared = a
            .band_keys()
            .iter()
            .filter(|k| b.band_keys().contains(k))
            .count();
        assert!(shared >= 1, "no shared bands at jaccard {j}");
    }

    #[test]
    fn unrelated_texts_score_low() {
        let other = "Refactor the scheduler to use a priority queue and add \
            benchmarks for the worst case path under sustained load.";
        let a = minhash(BODY).unwrap();
        let b = minhash(other).unwrap();
        assert!(a.jaccard(&b) < 0.2);
    }

    #[test]
    fn short_text_has_no_sketch() {
        assert!(minhash("fix typo").is_none());
        assert!(minhash("").is_none());
    }

    #[test]
    fn sketch_is_deterministic() {
        let a = minhash(BODY).unwrap();
        let b = minhash(BODY).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn jaccard_estimate_tracks_overlap() {
        // Two texts sharing roughly half their sentences should land in a
        // middling band, distinguishing them from both extremes.
        let half = "This PR updates the README to improve the documentation \
            and add a completely new section about deployment strategies today.";
        let a = minhash(BODY).unwrap();
        let b = minhash(half).unwrap();
        let j = a.jaccard(&b);
        assert!(j > 0.15 && j < 0.85, "jaccard {j}");
    }
}
