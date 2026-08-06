//! Dependency-free hash primitives shared by the signature lanes.

/// FNV-1a 64-bit over bytes. Stable across runs and platforms.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// splitmix64 mixer; used to derive hash families from fixed seeds.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Alphanumeric token runs, lowercased. The shared tokenizer for shingling.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Hashes of `n`-token shingles. Empty when there are fewer than `n` tokens.
pub fn shingle_hashes(toks: &[String], n: usize) -> Vec<u64> {
    if toks.len() < n {
        return Vec::new();
    }
    toks.windows(n)
        .map(|w| {
            let joined = w.join("\u{1f}");
            fnv1a64(joined.as_bytes())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_known_vector() {
        // FNV-1a("a") per the reference implementation.
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
    }

    #[test]
    fn splitmix_is_deterministic_and_spreads() {
        assert_eq!(splitmix64(1), splitmix64(1));
        assert_ne!(splitmix64(1), splitmix64(2));
    }

    #[test]
    fn tokenizer_lowercases_and_splits() {
        assert_eq!(tokens("Hello, World-42!"), vec!["hello", "world", "42"]);
        assert!(tokens("...").is_empty());
    }

    #[test]
    fn shingles_need_enough_tokens() {
        let t = tokens("a b c");
        assert!(shingle_hashes(&t, 4).is_empty());
        assert_eq!(shingle_hashes(&t, 2).len(), 2);
    }

    #[test]
    fn shingles_are_order_sensitive() {
        let a = shingle_hashes(&tokens("x y z w"), 4);
        let b = shingle_hashes(&tokens("w z y x"), 4);
        assert_ne!(a, b);
    }
}
