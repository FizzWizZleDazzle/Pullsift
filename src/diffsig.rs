//! Diff signatures: normalize a unified patch, shingle the added lines, and
//! produce a weighted 64-bit simhash. Near-duplicate diffs sit within a small
//! Hamming distance of each other.
//!
//! Candidate retrieval uses multi-index hashing over four 16-bit bands: by
//! pigeonhole, any pair within Hamming distance 3 shares at least one exact
//! band, so recall is guaranteed for d <= 3 and probabilistic up to the
//! verification cutoff `NEAR_HAMMING`.

use crate::hashing::{fnv1a64, shingle_hashes, tokens};
use std::collections::HashMap;

/// Verification cutoff: signatures at or below this Hamming distance are
/// near-duplicates.
pub const NEAR_HAMMING: u32 = 6;

const SHINGLE: usize = 4;

/// Added lines of a unified diff, normalized: no hunk headers, no file
/// markers, whitespace collapsed, lowercased.
pub fn normalize_patch(patch: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix('+') {
            if rest.starts_with("++") {
                continue; // "+++ b/file" marker
            }
            let collapsed = rest.split_whitespace().collect::<Vec<_>>().join(" ");
            if !collapsed.is_empty() {
                out.push(collapsed.to_lowercase());
            }
        }
    }
    out
}

/// Weighted 64-bit simhash over 4-token shingles of the normalized added
/// lines. Returns None when the diff is too small to fingerprint (a handful
/// of tokens would collide everything).
pub fn simhash(patch: &str) -> Option<u64> {
    let text = normalize_patch(patch).join("\n");
    let toks = tokens(&text);
    let hashes = shingle_hashes(&toks, SHINGLE);
    if hashes.len() < 3 {
        return None;
    }
    let mut counts: HashMap<u64, i64> = HashMap::new();
    for h in hashes {
        *counts.entry(h).or_insert(0) += 1;
    }
    let mut acc = [0i64; 64];
    for (h, w) in counts {
        for (bit, slot) in acc.iter_mut().enumerate() {
            if (h >> bit) & 1 == 1 {
                *slot += w;
            } else {
                *slot -= w;
            }
        }
    }
    let mut sig = 0u64;
    for (bit, v) in acc.iter().enumerate() {
        if *v > 0 {
            sig |= 1 << bit;
        }
    }
    Some(sig)
}

pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// True when the patch adds lines but none of them carry an alphanumeric
/// token: pure whitespace or punctuation churn.
pub fn whitespace_only(patch: &str) -> bool {
    let mut saw_added = false;
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix('+') {
            if rest.starts_with("++") {
                continue;
            }
            saw_added = true;
            if !crate::hashing::tokens(rest).is_empty() {
                return false;
            }
        }
    }
    saw_added
}

/// Comment share of the added lines, or None below a minimum of ten
/// nonempty added lines. Generated code over-comments: restate-the-code
/// comments and section dividers push this ratio up (ICSE 2025 finding;
/// the fit prices it).
pub fn comment_density(patch: &str) -> Option<f64> {
    let mut code = 0u32;
    let mut comments = 0u32;
    for line in patch.lines() {
        let Some(rest) = line.strip_prefix('+') else {
            continue;
        };
        if rest.starts_with("++") {
            continue;
        }
        let t = rest.trim_start();
        if t.is_empty() {
            continue;
        }
        if t.starts_with("//")
            || t.starts_with('#')
            || t.starts_with("/*")
            || t.starts_with('*')
            || t.starts_with("--")
            || t.starts_with("\"\"\"")
        {
            comments += 1;
        } else {
            code += 1;
        }
    }
    let total = code + comments;
    if total < 10 {
        return None;
    }
    Some(comments as f64 / total as f64)
}

/// The four 16-bit bands of a signature, keyed for multi-index lookup.
pub fn bands(sig: u64) -> [(u8, u16); 4] {
    [
        (0, (sig >> 48) as u16),
        (1, (sig >> 32) as u16),
        (2, (sig >> 16) as u16),
        (3, sig as u16),
    ]
}

/// Hash of the sorted set of changed file paths. Identical path-sets are a
/// strong campaign tell on their own (every tutorial PR touches README.md).
pub fn pathset_hash(paths: &[String]) -> u64 {
    let mut sorted: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    sorted.sort_unstable();
    fnv1a64(sorted.join("\u{1f}").as_bytes())
}

/// Multi-index over simhashes: O(1) candidate retrieval per PR.
#[derive(Debug, Default)]
pub struct SimhashIndex {
    buckets: HashMap<(u8, u16), Vec<usize>>,
    sigs: Vec<u64>,
}

impl SimhashIndex {
    /// Insert a signature, returning its id.
    pub fn insert(&mut self, sig: u64) -> usize {
        let id = self.sigs.len();
        self.sigs.push(sig);
        for key in bands(sig) {
            self.buckets.entry(key).or_default().push(id);
        }
        id
    }

    /// Ids of stored signatures within `NEAR_HAMMING` of `sig`.
    pub fn near(&self, sig: u64) -> Vec<usize> {
        let mut seen = Vec::new();
        for key in bands(sig) {
            if let Some(ids) = self.buckets.get(&key) {
                for &id in ids {
                    if !seen.contains(&id) && hamming(self.sigs[id], sig) <= NEAR_HAMMING {
                        seen.push(id);
                    }
                }
            }
        }
        seen
    }

    pub fn get(&self, id: usize) -> Option<u64> {
        self.sigs.get(id).copied()
    }

    pub fn len(&self) -> usize {
        self.sigs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sigs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn comment_density_measures_added_lines() {
        let mut patch = String::from("--- a/x.rs\n+++ b/x.rs\n");
        for i in 0..8 {
            patch.push_str(&format!("+// step {i}: initialize the value\n"));
            patch.push_str(&format!("+let v{i} = {i};\n"));
        }
        let d = super::comment_density(&patch).unwrap();
        assert!((d - 0.5).abs() < 1e-9, "density {d}");
        // Too few added lines: abstain.
        assert!(super::comment_density("+// one\n+let a = 1;\n").is_none());
    }

    use super::*;

    const PATCH_A: &str = "\
--- a/README.md
+++ b/README.md
@@ -1,3 +1,4 @@
 # Express
+Express is a fast, unopinionated, minimalist web framework for node.
+It makes building web applications really easy for everyone involved.
 body";

    // Same content, different whitespace and case: must normalize identical.
    const PATCH_A2: &str = "\
--- a/README.md
+++ b/README.md
@@ -1,3 +1,4 @@
 # Express
+Express   is a FAST, unopinionated,   minimalist web framework for node.
+It makes  building web applications really easy for everyone  involved.
 body";

    const PATCH_B: &str = "\
--- a/src/router.rs
+++ b/src/router.rs
@@ -10,6 +10,9 @@
+fn resolve_route(table: &RouteTable, path: &str) -> Option<HandlerId> {
+    table.lookup(path).or_else(|| table.wildcard(path))
+}
+// completely different content about routing internals and lookup tables";

    #[test]
    fn normalize_keeps_added_lines_only() {
        let lines = normalize_patch(PATCH_A);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("express is a fast"));
    }

    #[test]
    fn normalize_skips_file_markers_and_context() {
        let lines = normalize_patch(PATCH_A);
        assert!(!lines.iter().any(|l| l.contains("b/readme")));
        assert!(!lines.iter().any(|l| l.contains("body")));
    }

    #[test]
    fn whitespace_and_case_do_not_change_signature() {
        assert_eq!(simhash(PATCH_A).unwrap(), simhash(PATCH_A2).unwrap());
    }

    #[test]
    fn different_content_is_far() {
        let a = simhash(PATCH_A).unwrap();
        let b = simhash(PATCH_B).unwrap();
        assert!(hamming(a, b) > NEAR_HAMMING, "d={}", hamming(a, b));
    }

    // A long generated-looking patch: a one-word tweak is a small fraction
    // of the shingles, which is the regime simhash tolerance is for.
    // Short spam diffs cluster through exact matches and the text channel.
    fn patch_long() -> String {
        let mut p = String::from("--- a/README.md\n+++ b/README.md\n@@ -1,3 +1,33 @@\n");
        for i in 0..30 {
            p.push_str(&format!(
                "+Section {i} explains how the framework handles topic number \
                 {i} with examples covering setup, configuration and common \
                 pitfalls in real deployments.\n"
            ));
        }
        p
    }

    #[test]
    fn near_duplicate_is_near() {
        // One word changed out of thirty lines.
        let base = patch_long();
        let tweaked = base.replacen("common pitfalls", "typical pitfalls", 1);
        assert_ne!(base, tweaked);
        let a = simhash(&base).unwrap();
        let b = simhash(&tweaked).unwrap();
        assert!(hamming(a, b) <= NEAR_HAMMING, "d={}", hamming(a, b));
    }

    #[test]
    fn tiny_diffs_have_no_signature() {
        assert!(simhash("+hi\n").is_none());
        assert!(simhash("").is_none());
    }

    #[test]
    fn index_finds_exact_and_near() {
        let mut ix = SimhashIndex::default();
        let base = patch_long();
        let a = simhash(&base).unwrap();
        let b = simhash(PATCH_B).unwrap();
        let ia = ix.insert(a);
        ix.insert(b);
        let hits = ix.near(a);
        assert_eq!(hits, vec![ia]);
        let tweaked = simhash(&base.replacen("common pitfalls", "typical pitfalls", 1)).unwrap();
        assert_eq!(ix.near(tweaked), vec![ia]);
    }

    #[test]
    fn index_guarantees_d3_recall() {
        // Flip any 3 bits: at least one 16-bit band survives intact.
        let mut ix = SimhashIndex::default();
        let base = 0xDEAD_BEEF_CAFE_F00Du64;
        let id = ix.insert(base);
        for i in 0..62 {
            let flipped = base ^ (1 << i) ^ (1 << (i + 1)) ^ (1 << ((i + 31) % 64));
            assert!(ix.near(flipped).contains(&id), "missed at flip pattern {i}");
        }
    }

    #[test]
    fn whitespace_only_detection() {
        assert!(whitespace_only("+   \n+ ---\n+ !!!\n"));
        assert!(whitespace_only("+\n+  \n"));
        assert!(!whitespace_only("+   \n+real change\n"));
        assert!(!whitespace_only("")); // no added lines at all
        assert!(!whitespace_only("-removed only\n"));
    }

    #[test]
    fn pathset_hash_is_order_insensitive() {
        let a = pathset_hash(&["README.md".into(), "docs/x.md".into()]);
        let b = pathset_hash(&["docs/x.md".into(), "README.md".into()]);
        assert_eq!(a, b);
        let c = pathset_hash(&["README.md".into()]);
        assert_ne!(a, c);
    }
}
