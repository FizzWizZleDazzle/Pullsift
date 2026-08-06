//! Federation: what installations share, dormant in the MVP.
//!
//! Two record kinds with deliberately different physics:
//!
//! - Campaign signatures carry no usernames and no diff content, expire by
//!   TTL, and propagate the moment a cluster forms.
//! - Author verdicts name an account (as a salted hash), and only take
//!   effect after corroboration by at least `CORROBORATION_N` independent
//!   installations. Automatic, but braked; never a single repo's opinion.
//!
//! Envelopes are ed25519-signed per installation. The MVP exchanges records
//! through a local directory stub so the whole protocol is exercised by
//! tests; a public relay replaces the transport later.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PROTOCOL_VERSION: u32 = 1;
/// Independent installations that must agree before an author verdict binds.
pub const CORROBORATION_N: usize = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignSignatureRecord {
    pub v: u32,
    pub simhash: Option<u64>,
    /// Banded LSH keys of the text sketch, not the sketch itself.
    pub text_band_keys: Vec<u64>,
    pub pathset_hash: u64,
    pub style_centroid: Vec<f64>,
    pub cluster_size: usize,
    pub first_seen_unix: i64,
    pub ttl_secs: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorVerdictRecord {
    pub v: u32,
    /// sha256(lowercased login + salt); lets subscribers match without the
    /// relay publishing a name list.
    pub author_hash: String,
    /// sha256 of the evidence JSON that produced this verdict.
    pub evidence_hash: String,
    /// Verdict strength in [0,1].
    pub strength: f64,
    pub installation_id: u64,
    /// Owner account of the installation, for independence checks.
    pub installation_owner: String,
    pub issued_unix: i64,
    pub expires_unix: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Record {
    CampaignSignature(CampaignSignatureRecord),
    AuthorVerdict(AuthorVerdictRecord),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedEnvelope {
    pub payload: String,
    pub signature_hex: String,
    pub pubkey_hex: String,
}

pub fn author_hash(login: &str, salt: &str) -> String {
    let mut h = Sha256::new();
    h.update(login.to_lowercase().as_bytes());
    h.update(b"\x1f");
    h.update(salt.as_bytes());
    hex::encode(h.finalize())
}

pub fn evidence_hash(evidence_json: &str) -> String {
    let mut h = Sha256::new();
    h.update(evidence_json.as_bytes());
    hex::encode(h.finalize())
}

pub fn sign(record: &Record, key: &SigningKey) -> SignedEnvelope {
    let payload = serde_json::to_string(record).expect("record serializes");
    let sig = key.sign(payload.as_bytes());
    SignedEnvelope {
        payload,
        signature_hex: hex::encode(sig.to_bytes()),
        pubkey_hex: hex::encode(key.verifying_key().to_bytes()),
    }
}

pub fn verify(envelope: &SignedEnvelope) -> Result<Record, String> {
    let pk_bytes: [u8; 32] = hex::decode(&envelope.pubkey_hex)
        .map_err(|e| e.to_string())?
        .try_into()
        .map_err(|_| "bad pubkey length".to_string())?;
    let pk = VerifyingKey::from_bytes(&pk_bytes).map_err(|e| e.to_string())?;
    let sig_bytes: [u8; 64] = hex::decode(&envelope.signature_hex)
        .map_err(|e| e.to_string())?
        .try_into()
        .map_err(|_| "bad signature length".to_string())?;
    let sig = Signature::from_bytes(&sig_bytes);
    pk.verify(envelope.payload.as_bytes(), &sig)
        .map_err(|_| "signature verification failed".to_string())?;
    serde_json::from_str(&envelope.payload).map_err(|e| e.to_string())
}

/// Corroborated strength of an author across verdicts: counts one verdict
/// per installation owner, and binds only at `CORROBORATION_N` independent
/// owners. Returns the mean strength of the corroborating set, or 0.0.
pub fn corroborated_strength(
    verdicts: &[AuthorVerdictRecord],
    author_hash: &str,
    now_unix: i64,
) -> f64 {
    let mut by_owner: std::collections::BTreeMap<&str, f64> = std::collections::BTreeMap::new();
    for v in verdicts {
        if v.author_hash != author_hash || v.expires_unix <= now_unix {
            continue;
        }
        let entry = by_owner.entry(v.installation_owner.as_str()).or_insert(0.0);
        if v.strength > *entry {
            *entry = v.strength;
        }
    }
    if by_owner.len() < CORROBORATION_N {
        return 0.0;
    }
    let sum: f64 = by_owner.values().sum();
    (sum / by_owner.len() as f64).clamp(0.0, 1.0)
}

/// File-based exchange stub: append-only JSONL per installation, poll by
/// reading every file. The transport is the only thing a relay replaces.
pub mod exchange {
    use super::SignedEnvelope;
    use std::fs;
    use std::path::Path;

    pub fn publish(dir: &Path, installation_id: u64, env: &SignedEnvelope) -> std::io::Result<()> {
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("{installation_id}.jsonl"));
        let line = serde_json::to_string(env).expect("envelope serializes");
        let mut existing = fs::read_to_string(&path).unwrap_or_default();
        existing.push_str(&line);
        existing.push('\n');
        fs::write(path, existing)
    }

    pub fn poll(dir: &Path) -> std::io::Result<Vec<SignedEnvelope>> {
        let mut out = Vec::new();
        if !dir.exists() {
            return Ok(out);
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            for line in fs::read_to_string(entry.path())?.lines() {
                if let Ok(env) = serde_json::from_str::<SignedEnvelope>(line) {
                    out.push(env);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn verdict(owner: &str, author: &str, strength: f64) -> AuthorVerdictRecord {
        AuthorVerdictRecord {
            v: PROTOCOL_VERSION,
            author_hash: author_hash(author, "salt"),
            evidence_hash: evidence_hash("{}"),
            strength,
            installation_id: 1,
            installation_owner: owner.into(),
            issued_unix: 1_000,
            expires_unix: 2_000,
        }
    }

    #[test]
    fn sign_verify_roundtrip() {
        let rec = Record::AuthorVerdict(verdict("org-a", "ghost", 0.9));
        let env = sign(&rec, &key(1));
        assert_eq!(verify(&env).unwrap(), rec);
    }

    #[test]
    fn tampered_payload_fails() {
        let rec = Record::AuthorVerdict(verdict("org-a", "ghost", 0.9));
        let mut env = sign(&rec, &key(1));
        env.payload = env.payload.replace("0.9", "0.1");
        assert!(verify(&env).is_err());
    }

    #[test]
    fn wrong_key_fails() {
        let rec = Record::AuthorVerdict(verdict("org-a", "ghost", 0.9));
        let mut env = sign(&rec, &key(1));
        env.pubkey_hex = hex::encode(key(2).verifying_key().to_bytes());
        assert!(verify(&env).is_err());
    }

    #[test]
    fn corroboration_brake_needs_three_owners() {
        let a = author_hash("ghost", "salt");
        let two = vec![
            verdict("org-a", "ghost", 0.9),
            verdict("org-b", "ghost", 0.8),
        ];
        assert_eq!(corroborated_strength(&two, &a, 1_500), 0.0);
        let three = vec![
            verdict("org-a", "ghost", 0.9),
            verdict("org-b", "ghost", 0.8),
            verdict("org-c", "ghost", 0.7),
        ];
        let s = corroborated_strength(&three, &a, 1_500);
        assert!((s - 0.8).abs() < 1e-9);
    }

    #[test]
    fn same_owner_counts_once() {
        let a = author_hash("ghost", "salt");
        let stacked = vec![
            verdict("org-a", "ghost", 0.9),
            verdict("org-a", "ghost", 0.9),
            verdict("org-a", "ghost", 0.9),
            verdict("org-b", "ghost", 0.8),
        ];
        assert_eq!(corroborated_strength(&stacked, &a, 1_500), 0.0);
    }

    #[test]
    fn expired_verdicts_do_not_count() {
        let a = author_hash("ghost", "salt");
        let three = vec![
            verdict("org-a", "ghost", 0.9),
            verdict("org-b", "ghost", 0.8),
            verdict("org-c", "ghost", 0.7),
        ];
        assert_eq!(corroborated_strength(&three, &a, 3_000), 0.0);
    }

    #[test]
    fn different_author_does_not_match() {
        let other = author_hash("innocent", "salt");
        let three = vec![
            verdict("org-a", "ghost", 0.9),
            verdict("org-b", "ghost", 0.8),
            verdict("org-c", "ghost", 0.7),
        ];
        assert_eq!(corroborated_strength(&three, &other, 1_500), 0.0);
    }

    #[test]
    fn author_hash_is_salted_and_case_insensitive() {
        assert_eq!(author_hash("Ghost", "s"), author_hash("ghost", "s"));
        assert_ne!(author_hash("ghost", "s1"), author_hash("ghost", "s2"));
    }

    #[test]
    fn signature_record_carries_no_identity() {
        let rec = CampaignSignatureRecord {
            v: PROTOCOL_VERSION,
            simhash: Some(42),
            text_band_keys: vec![1, 2, 3],
            pathset_hash: 7,
            style_centroid: vec![0.1; 7],
            cluster_size: 12,
            first_seen_unix: 1_000,
            ttl_secs: 3600,
        };
        let json = serde_json::to_string(&Record::CampaignSignature(rec)).unwrap();
        assert!(!json.contains("login"));
        assert!(!json.contains("author"));
    }

    #[test]
    fn exchange_stub_roundtrip() {
        let dir = std::env::temp_dir().join(format!("slopfed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let rec = Record::AuthorVerdict(verdict("org-a", "ghost", 0.9));
        let env = sign(&rec, &key(1));
        exchange::publish(&dir, 11, &env).unwrap();
        exchange::publish(&dir, 12, &env).unwrap();
        let polled = exchange::poll(&dir).unwrap();
        assert_eq!(polled.len(), 2);
        assert_eq!(verify(&polled[0]).unwrap(), rec);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
