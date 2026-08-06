//! Campaign clustering: union-find over signature matches inside a TTL
//! window, with burst scoring and a stylometry cohesion measure.
//!
//! A PR joining a hot cluster inherits the cluster's confidence through
//! engine rules (`CLUSTER_*`), not a separate code path.

use crate::diffsig::SimhashIndex;
use crate::engine::Fire;
use crate::stylometry::StyleVector;
use crate::textsig::MinHash;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Sliding window for burst measurement.
pub const BURST_WINDOW_SECS: i64 = 6 * 3600;
/// Signatures older than this fall out of the cluster store.
pub const TTL_SECS: i64 = 14 * 24 * 3600;

/// One PR's signature bundle as it enters the store.
#[derive(Debug, Clone)]
pub struct PrSignature {
    pub repo: String,
    pub pr_number: u64,
    /// Author login, for counting distinct accounts in a cluster. Local
    /// only: federation envelopes carry signatures without identity.
    pub author: String,
    pub arrived: DateTime<Utc>,
    pub diff_sim: Option<u64>,
    pub text_min: Option<MinHash>,
    pub pathset: u64,
    pub style: StyleVector,
}

/// What the store reports back for the PR just inserted.
#[derive(Debug, Clone)]
pub struct ClusterView {
    pub cluster_id: usize,
    pub size: usize,
    pub distinct_repos: usize,
    pub distinct_authors: usize,
    pub burst: f64,
    pub style_cohesion: f64,
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self { parent: Vec::new() }
    }
    fn add(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        id
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[rb] = ra;
        }
    }
}

/// In-memory cluster store. Persisted signatures are replayed into this on
/// startup; the store itself is the source of clustering truth at runtime.
pub struct ClusterStore {
    entries: Vec<PrSignature>,
    uf: UnionFind,
    diff_index: SimhashIndex,
    /// diff index id -> entry id (diffless entries never enter the index).
    diff_owner: Vec<usize>,
    text_buckets: HashMap<u64, Vec<usize>>,
    path_buckets: HashMap<u64, Vec<usize>>,
    /// Repo baseline: expected PR arrivals per burst window.
    baseline_per_window: f64,
}

impl ClusterStore {
    pub fn new(baseline_per_window: f64) -> Self {
        Self {
            entries: Vec::new(),
            uf: UnionFind::new(),
            diff_index: SimhashIndex::default(),
            diff_owner: Vec::new(),
            text_buckets: HashMap::new(),
            path_buckets: HashMap::new(),
            baseline_per_window: baseline_per_window.max(0.05),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a PR signature; link it to near-duplicates; return the view of
    /// its (possibly grown) cluster.
    pub fn insert(&mut self, sig: PrSignature) -> ClusterView {
        let id = self.uf.add();
        debug_assert_eq!(id, self.entries.len());

        // Diff channel.
        if let Some(sim) = sig.diff_sim {
            for hit in self.diff_index.near(sim) {
                self.uf.union(self.diff_owner[hit], id);
            }
            self.diff_index.insert(sim);
            self.diff_owner.push(id);
        }

        // Text channel: banded LSH candidates, verified by Jaccard.
        if let Some(min) = &sig.text_min {
            let mut seen: Vec<usize> = Vec::new();
            for key in min.band_keys() {
                if let Some(ids) = self.text_buckets.get(&key) {
                    for &other in ids {
                        if seen.contains(&other) {
                            continue;
                        }
                        if let Some(om) = &self.entries[other].text_min {
                            if min.jaccard(om) >= crate::textsig::NEAR_JACCARD {
                                self.uf.union(other, id);
                            }
                        }
                        seen.push(other);
                    }
                }
            }
            for key in min.band_keys() {
                self.text_buckets.entry(key).or_default().push(id);
            }
        }

        // Path-set channel records membership only; identical path-sets alone
        // do not merge clusters (every docs PR touches README.md), but they
        // sharpen burst measurement below.
        self.path_buckets.entry(sig.pathset).or_default().push(id);

        self.entries.push(sig);
        self.view_of(id)
    }

    fn view_of(&mut self, id: usize) -> ClusterView {
        let root = self.uf.find(id);
        let now = self.entries[id].arrived;
        let member_ids: Vec<usize> = (0..self.entries.len())
            .filter(|&e| self.uf.find(e) == root)
            .collect();
        let size = member_ids.len();
        let mut repos: Vec<&str> = member_ids
            .iter()
            .map(|&e| self.entries[e].repo.as_str())
            .collect();
        repos.sort_unstable();
        repos.dedup();
        let distinct_repos = repos.len();
        let mut authors: Vec<&str> = member_ids
            .iter()
            .map(|&e| self.entries[e].author.as_str())
            .collect();
        authors.sort_unstable();
        authors.dedup();
        let distinct_authors = authors.len();

        let in_window = member_ids
            .iter()
            .filter(|&&e| (now - self.entries[e].arrived).num_seconds() <= BURST_WINDOW_SECS)
            .count();
        let burst = burst_score(in_window as u64, self.baseline_per_window);

        let styles: Vec<StyleVector> = member_ids
            .iter()
            .map(|&e| self.entries[e].style.clone())
            .collect();
        let style_cohesion = cohesion(&styles);

        ClusterView {
            cluster_id: root,
            size,
            distinct_repos,
            distinct_authors,
            burst,
            style_cohesion,
        }
    }

    /// Drop entries older than the TTL. Rebuilds the store; clusters are
    /// recomputed from what remains.
    pub fn prune(&mut self, now: DateTime<Utc>) {
        let keep: Vec<PrSignature> = self
            .entries
            .drain(..)
            .filter(|e| (now - e.arrived).num_seconds() <= TTL_SECS)
            .collect();
        let baseline = self.baseline_per_window;
        *self = ClusterStore::new(baseline);
        for e in keep {
            self.insert(e);
        }
    }
}

/// Rules a cluster view contributes to its member PR's score.
///
/// Size, burst, and cohesion require at least two distinct authors: a
/// campaign is many accounts sending the same change, while one person's
/// own batch of similar PRs (a docs sweep, a maintenance series) is
/// normal work and belongs to the dossier lane if it is not. A single
/// account spraying the same change across repos still counts through
/// `CLUSTER_XREPO`.
pub fn cluster_rules(view: &ClusterView) -> Vec<Fire> {
    let mut out = Vec::new();
    if view.size >= 2 && view.distinct_authors >= 2 {
        let size_val = ((view.size as f64).ln() / (50.0f64).ln()).min(1.0);
        out.push(Fire::new("CLUSTER_SIZE_LOG", size_val));
        out.push(Fire::new("CLUSTER_BURST", view.burst));
        if view.style_cohesion > 0.9 {
            out.push(Fire::new("CLUSTER_STYLE_COHESION", view.style_cohesion));
        }
    }
    if view.size >= 2 && view.distinct_repos >= 2 {
        let x = ((view.distinct_repos - 1) as f64 / 9.0).min(1.0);
        out.push(Fire::new("CLUSTER_XREPO", x));
    }
    out
}

/// Poisson surprise of seeing `k` arrivals in a window with baseline
/// `lambda`, squashed to [0,1): `s / (s + 8)` where
/// `s = -ln P(X >= k)`.
pub fn burst_score(k: u64, lambda: f64) -> f64 {
    if k <= 1 {
        return 0.0;
    }
    let s = -ln_poisson_tail(k, lambda);
    s / (s + 5.0)
}

/// ln P(X >= k) for X ~ Poisson(lambda), via log-sum-exp over the tail.
fn ln_poisson_tail(k: u64, lambda: f64) -> f64 {
    if k == 0 {
        return 0.0;
    }
    // ln pmf(i) = -lambda + i ln lambda - ln i!
    let ln_lambda = lambda.ln();
    let mut ln_fact = 0.0;
    for i in 1..=k {
        ln_fact += (i as f64).ln();
    }
    let mut ln_term = -lambda + k as f64 * ln_lambda - ln_fact;
    let mut max_term = ln_term;
    let mut terms = vec![ln_term];
    let mut i = k + 1;
    loop {
        // pmf(i)/pmf(i-1) = lambda / i
        ln_term += ln_lambda - (i as f64).ln();
        if ln_term < max_term - 40.0 || i > k + 10_000 {
            break;
        }
        if ln_term > max_term {
            max_term = ln_term;
        }
        terms.push(ln_term);
        i += 1;
    }
    let sum: f64 = terms.iter().map(|t| (t - max_term).exp()).sum();
    (max_term + sum.ln()).min(0.0)
}

fn cohesion(styles: &[StyleVector]) -> f64 {
    if styles.len() < 2 {
        return 0.0;
    }
    match StyleVector::mean(styles) {
        Some(centroid) => {
            let sum: f64 = styles.iter().map(|s| s.cosine(&centroid)).sum();
            (sum / styles.len() as f64).clamp(0.0, 1.0)
        }
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffsig;
    use crate::stylometry::analyze;
    use crate::textsig::minhash;
    use chrono::TimeZone;

    fn t(minute: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1_700_000_000 + minute * 60, 0).unwrap()
    }

    fn readme_patch(variant: &str) -> String {
        format!(
            "+Express is a fast, unopinionated, minimalist web framework for node.\n\
             +It makes building web applications {variant} easy for everyone involved.\n"
        )
    }

    fn sig(repo: &str, n: u64, minute: i64, patch: &str, body: &str) -> PrSignature {
        // Distinct author per PR number: the wave shape, many accounts.
        sig_by(repo, &format!("acct{n}"), n, minute, patch, body)
    }

    fn sig_by(
        repo: &str,
        author: &str,
        n: u64,
        minute: i64,
        patch: &str,
        body: &str,
    ) -> PrSignature {
        PrSignature {
            repo: repo.into(),
            pr_number: n,
            author: author.into(),
            arrived: t(minute),
            diff_sim: diffsig::simhash(patch),
            text_min: minhash(body),
            pathset: diffsig::pathset_hash(&["README.md".into()]),
            style: analyze(body),
        }
    }

    const WAVE_BODY: &str = "This PR improves the README documentation and \
        fixes a typo in the installation instructions to help new users get started.";

    #[test]
    fn identical_diffs_form_one_cluster() {
        let mut store = ClusterStore::new(0.5);
        let mut last = None;
        for i in 0..5 {
            let v = store.insert(sig("o/r", i, i as i64, &readme_patch("really"), WAVE_BODY));
            last = Some(v);
        }
        let v = last.unwrap();
        assert_eq!(v.size, 5);
    }

    #[test]
    fn near_duplicate_diffs_still_cluster() {
        let mut store = ClusterStore::new(0.5);
        store.insert(sig("o/r", 1, 0, &readme_patch("really"), WAVE_BODY));
        let v = store.insert(sig("o/r", 2, 5, &readme_patch("very"), WAVE_BODY));
        assert_eq!(v.size, 2);
    }

    #[test]
    fn unrelated_prs_stay_apart() {
        let mut store = ClusterStore::new(0.5);
        store.insert(sig("o/r", 1, 0, &readme_patch("really"), WAVE_BODY));
        let other_patch = "+fn resolve(t: &Table, p: &str) -> Option<Id> {\n\
             +    t.lookup(p).or_else(|| t.wild(p))\n\
             +}\n+// routing internals lookup path resolution logic here\n";
        let other_body = "Refactor the route resolution to fall back to \
            wildcard lookup when the exact table match fails under load.";
        let v = store.insert(sig("o/r", 2, 5, other_patch, other_body));
        assert_eq!(v.size, 1);
    }

    #[test]
    fn forty_pr_wave_bursts() {
        let mut store = ClusterStore::new(0.5); // one PR every other window
        let mut last = None;
        for i in 0..40 {
            // Arrivals two minutes apart: 40 PRs where 0.5 was expected.
            let v = store.insert(sig(
                "o/r",
                i,
                (i as i64) * 2,
                &readme_patch("really"),
                WAVE_BODY,
            ));
            last = Some(v);
        }
        let v = last.unwrap();
        assert_eq!(v.size, 40);
        assert!(v.burst > 0.9, "burst {}", v.burst);
        assert!(v.style_cohesion > 0.9, "cohesion {}", v.style_cohesion);
        let rules = cluster_rules(&v);
        assert!(rules
            .iter()
            .any(|f| f.rule == "CLUSTER_BURST" && f.value > 0.9));
        assert!(rules.iter().any(|f| f.rule == "CLUSTER_SIZE_LOG"));
    }

    #[test]
    fn slow_trickle_does_not_burst() {
        let mut store = ClusterStore::new(2.0);
        // Two similar PRs, three days apart: same cluster, no burst.
        store.insert(sig("o/r", 1, 0, &readme_patch("really"), WAVE_BODY));
        let v = store.insert(sig("o/r", 2, 3 * 24 * 60, &readme_patch("very"), WAVE_BODY));
        assert_eq!(v.size, 2);
        assert!(v.burst < 0.3, "burst {}", v.burst);
    }

    #[test]
    fn cross_repo_membership_is_counted() {
        let mut store = ClusterStore::new(0.5);
        store.insert(sig("a/x", 1, 0, &readme_patch("really"), WAVE_BODY));
        let v = store.insert(sig("b/y", 1, 1, &readme_patch("really"), WAVE_BODY));
        assert_eq!(v.distinct_repos, 2);
        assert!(cluster_rules(&v).iter().any(|f| f.rule == "CLUSTER_XREPO"));
    }

    #[test]
    fn singleton_emits_no_cluster_rules() {
        let mut store = ClusterStore::new(0.5);
        let v = store.insert(sig("o/r", 1, 0, &readme_patch("really"), WAVE_BODY));
        assert!(cluster_rules(&v).is_empty());
    }

    #[test]
    fn single_author_batch_stays_quiet() {
        // One person's own batch of similar PRs clusters but convicts
        // nothing: size, burst, and cohesion need at least two accounts.
        let mut store = ClusterStore::new(0.5);
        let mut last = None;
        for i in 0..6 {
            let v = store.insert(sig_by(
                "o/r",
                "docs-team",
                i,
                i as i64,
                &readme_patch("really"),
                WAVE_BODY,
            ));
            last = Some(v);
        }
        let v = last.unwrap();
        assert_eq!(v.size, 6);
        assert_eq!(v.distinct_authors, 1);
        assert!(cluster_rules(&v).is_empty());
    }

    #[test]
    fn single_author_spray_across_repos_still_fires_xrepo() {
        let mut store = ClusterStore::new(0.5);
        store.insert(sig_by(
            "a/x",
            "sprayer",
            1,
            0,
            &readme_patch("really"),
            WAVE_BODY,
        ));
        let v = store.insert(sig_by(
            "b/y",
            "sprayer",
            1,
            1,
            &readme_patch("really"),
            WAVE_BODY,
        ));
        let rules = cluster_rules(&v);
        assert!(rules.iter().any(|f| f.rule == "CLUSTER_XREPO"));
        assert!(rules.iter().all(|f| f.rule == "CLUSTER_XREPO"));
    }

    #[test]
    fn prune_drops_expired_entries() {
        let mut store = ClusterStore::new(0.5);
        store.insert(sig("o/r", 1, 0, &readme_patch("really"), WAVE_BODY));
        store.insert(sig("o/r", 2, 1, &readme_patch("really"), WAVE_BODY));
        assert_eq!(store.len(), 2);
        store.prune(t(0) + chrono::Duration::seconds(TTL_SECS + 120));
        assert!(store.is_empty());
    }

    #[test]
    fn burst_score_monotonic_in_k() {
        let lambda = 1.0;
        let mut prev = 0.0;
        for k in 1..50 {
            let s = burst_score(k, lambda);
            assert!(s >= prev, "k={k}");
            prev = s;
        }
        assert!(burst_score(1, 1.0) == 0.0);
        assert!(burst_score(40, 0.5) > 0.95);
    }

    #[test]
    fn poisson_tail_sanity() {
        // P(X >= 1) for lambda=1 is 1 - e^-1 ~= 0.632.
        let p = ln_poisson_tail(1, 1.0).exp();
        assert!((p - 0.6321).abs() < 1e-3, "p={p}");
        // Large k, small lambda: vanishing tail, big surprise.
        assert!(ln_poisson_tail(40, 0.5) < -80.0);
    }
}
