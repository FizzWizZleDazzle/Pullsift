//! Persistence. The `Store` trait is what the pipeline talks to; Postgres is
//! the system of record in production, and `MemStore` is the test double.

use crate::challenge::ChallengeState;
use crate::dossier::DossierFacts;
use crate::engine::{Verdict, Weights};
use crate::fit::Example;
use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use sqlx::postgres::PgPool;
use sqlx::Row;
use std::collections::HashMap;
use std::sync::Mutex;

pub const DOSSIER_TTL_DAYS: i64 = 7;

#[allow(async_fn_in_trait)]
pub trait Store {
    async fn record_event(
        &self,
        repo: &str,
        pr: u64,
        author: &str,
        action: &str,
        payload: &Value,
    ) -> Result<()>;

    async fn get_dossier(&self, login: &str, now: DateTime<Utc>) -> Result<Option<DossierFacts>>;
    async fn put_dossier(&self, facts: &DossierFacts, now: DateTime<Utc>) -> Result<()>;

    async fn save_verdict(&self, repo: &str, pr: u64, author: &str, v: &Verdict) -> Result<i64>;
    async fn log_action(
        &self,
        repo: &str,
        pr: u64,
        verdict_id: Option<i64>,
        action: &str,
        dry_run: bool,
        detail: &Value,
    ) -> Result<()>;

    async fn get_challenge(&self, repo: &str, pr: u64) -> Result<Option<ChallengeState>>;
    async fn put_challenge(&self, repo: &str, pr: u64, state: &ChallengeState) -> Result<()>;

    async fn add_feedback(&self, verdict_id: i64, is_slop: bool, correction: bool) -> Result<()>;
    /// Labeled examples for the learner, in stable id order.
    async fn load_examples(&self) -> Result<Vec<Example>>;

    async fn active_weights(&self) -> Result<Option<Weights>>;
    async fn promote_weights(&self, w: &Weights, reason: &str, auc: f64) -> Result<()>;
}

// ---------------------------------------------------------------------------
// In-memory store: the test double.
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct MemStore {
    inner: Mutex<MemInner>,
}

#[derive(Default)]
struct MemInner {
    events: Vec<(String, u64, String, String)>,
    dossiers: HashMap<String, (DossierFacts, DateTime<Utc>)>,
    verdicts: Vec<(String, u64, String, Verdict)>,
    actions: Vec<(String, u64, String, bool)>,
    challenges: HashMap<(String, u64), ChallengeState>,
    feedback: Vec<(i64, bool, bool)>,
    weights: Vec<(Weights, String, f64)>,
}

impl MemStore {
    pub fn actions(&self) -> Vec<(String, u64, String, bool)> {
        self.inner.lock().unwrap().actions.clone()
    }
    pub fn verdict_count(&self) -> usize {
        self.inner.lock().unwrap().verdicts.len()
    }
}

impl Store for MemStore {
    async fn record_event(
        &self,
        repo: &str,
        pr: u64,
        author: &str,
        action: &str,
        _payload: &Value,
    ) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .events
            .push((repo.into(), pr, author.into(), action.into()));
        Ok(())
    }

    async fn get_dossier(&self, login: &str, now: DateTime<Utc>) -> Result<Option<DossierFacts>> {
        Ok(self
            .inner
            .lock()
            .unwrap()
            .dossiers
            .get(login)
            .and_then(|(f, at)| {
                if now - *at <= Duration::days(DOSSIER_TTL_DAYS) {
                    Some(f.clone())
                } else {
                    None
                }
            }))
    }

    async fn put_dossier(&self, facts: &DossierFacts, now: DateTime<Utc>) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .dossiers
            .insert(facts.login.clone(), (facts.clone(), now));
        Ok(())
    }

    async fn save_verdict(&self, repo: &str, pr: u64, author: &str, v: &Verdict) -> Result<i64> {
        let mut inner = self.inner.lock().unwrap();
        inner
            .verdicts
            .push((repo.into(), pr, author.into(), v.clone()));
        Ok(inner.verdicts.len() as i64)
    }

    async fn log_action(
        &self,
        repo: &str,
        pr: u64,
        _verdict_id: Option<i64>,
        action: &str,
        dry_run: bool,
        _detail: &Value,
    ) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .actions
            .push((repo.into(), pr, action.into(), dry_run));
        Ok(())
    }

    async fn get_challenge(&self, repo: &str, pr: u64) -> Result<Option<ChallengeState>> {
        Ok(self
            .inner
            .lock()
            .unwrap()
            .challenges
            .get(&(repo.into(), pr))
            .cloned())
    }

    async fn put_challenge(&self, repo: &str, pr: u64, state: &ChallengeState) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .challenges
            .insert((repo.into(), pr), state.clone());
        Ok(())
    }

    async fn add_feedback(&self, verdict_id: i64, is_slop: bool, correction: bool) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .feedback
            .push((verdict_id, is_slop, correction));
        Ok(())
    }

    async fn load_examples(&self) -> Result<Vec<Example>> {
        let inner = self.inner.lock().unwrap();
        let mut out = Vec::new();
        for (id, is_slop, correction) in &inner.feedback {
            let ix = (*id - 1) as usize;
            if let Some((_, _, _, v)) = inner.verdicts.get(ix) {
                let fires = v
                    .evidence
                    .iter()
                    .map(|e| crate::engine::Fire::new(&e.rule, e.value))
                    .collect();
                out.push(if *correction {
                    Example::correction(fires, *is_slop)
                } else {
                    Example::new(fires, *is_slop)
                });
            }
        }
        Ok(out)
    }

    async fn active_weights(&self) -> Result<Option<Weights>> {
        Ok(self
            .inner
            .lock()
            .unwrap()
            .weights
            .last()
            .map(|(w, _, _)| w.clone()))
    }

    async fn promote_weights(&self, w: &Weights, reason: &str, auc: f64) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .weights
            .push((w.clone(), reason.into(), auc));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Postgres store: the system of record.
// ---------------------------------------------------------------------------

pub struct PgStore {
    pool: PgPool,
}

impl PgStore {
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = PgPool::connect(url).await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

impl Store for PgStore {
    async fn record_event(
        &self,
        repo: &str,
        pr: u64,
        author: &str,
        action: &str,
        payload: &Value,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO pr_events (repo, pr_number, author, action, payload) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(repo)
        .bind(pr as i64)
        .bind(author)
        .bind(action)
        .bind(payload)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_dossier(&self, login: &str, now: DateTime<Utc>) -> Result<Option<DossierFacts>> {
        let cutoff = now - Duration::days(DOSSIER_TTL_DAYS);
        let row = sqlx::query("SELECT facts FROM dossiers WHERE login = $1 AND fetched_at > $2")
            .bind(login)
            .bind(cutoff)
            .fetch_optional(&self.pool)
            .await?;
        Ok(match row {
            Some(r) => serde_json::from_value(r.get::<Value, _>("facts")).ok(),
            None => None,
        })
    }

    async fn put_dossier(&self, facts: &DossierFacts, now: DateTime<Utc>) -> Result<()> {
        sqlx::query(
            "INSERT INTO dossiers (login, facts, fetched_at) VALUES ($1, $2, $3) \
             ON CONFLICT (login) DO UPDATE SET facts = $2, fetched_at = $3",
        )
        .bind(&facts.login)
        .bind(serde_json::to_value(facts)?)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn save_verdict(&self, repo: &str, pr: u64, author: &str, v: &Verdict) -> Result<i64> {
        let row = sqlx::query(
            "INSERT INTO verdicts (repo, pr_number, author, probability, tier, evidence) \
             VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        )
        .bind(repo)
        .bind(pr as i64)
        .bind(author)
        .bind(v.probability)
        .bind(format!("{:?}", v.tier))
        .bind(serde_json::to_value(&v.evidence)?)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("id"))
    }

    async fn log_action(
        &self,
        repo: &str,
        pr: u64,
        verdict_id: Option<i64>,
        action: &str,
        dry_run: bool,
        detail: &Value,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO actions_log (repo, pr_number, verdict_id, action, dry_run, detail) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(repo)
        .bind(pr as i64)
        .bind(verdict_id)
        .bind(action)
        .bind(dry_run)
        .bind(detail)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_challenge(&self, repo: &str, pr: u64) -> Result<Option<ChallengeState>> {
        let row = sqlx::query("SELECT state FROM challenges WHERE repo = $1 AND pr_number = $2")
            .bind(repo)
            .bind(pr as i64)
            .fetch_optional(&self.pool)
            .await?;
        Ok(match row {
            Some(r) => serde_json::from_value(r.get::<Value, _>("state")).ok(),
            None => None,
        })
    }

    async fn put_challenge(&self, repo: &str, pr: u64, state: &ChallengeState) -> Result<()> {
        sqlx::query(
            "INSERT INTO challenges (repo, pr_number, state, updated_at) \
             VALUES ($1, $2, $3, now()) \
             ON CONFLICT (repo, pr_number) DO UPDATE SET state = $3, updated_at = now()",
        )
        .bind(repo)
        .bind(pr as i64)
        .bind(serde_json::to_value(state)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn add_feedback(&self, verdict_id: i64, is_slop: bool, correction: bool) -> Result<()> {
        sqlx::query("INSERT INTO feedback (verdict_id, is_slop, correction) VALUES ($1, $2, $3)")
            .bind(verdict_id)
            .bind(is_slop)
            .bind(correction)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn load_examples(&self) -> Result<Vec<Example>> {
        let rows = sqlx::query(
            "SELECT v.evidence, f.is_slop, f.correction \
             FROM feedback f JOIN verdicts v ON v.id = f.verdict_id \
             ORDER BY f.id",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out = Vec::new();
        for r in rows {
            let evidence: Vec<crate::engine::EvidenceItem> =
                serde_json::from_value(r.get::<Value, _>("evidence"))?;
            let fires = evidence
                .iter()
                .map(|e| crate::engine::Fire::new(&e.rule, e.value))
                .collect();
            let is_slop: bool = r.get("is_slop");
            out.push(if r.get::<bool, _>("correction") {
                Example::correction(fires, is_slop)
            } else {
                Example::new(fires, is_slop)
            });
        }
        Ok(out)
    }

    async fn active_weights(&self) -> Result<Option<Weights>> {
        let row = sqlx::query("SELECT weights FROM weights_versions WHERE active")
            .fetch_optional(&self.pool)
            .await?;
        Ok(match row {
            Some(r) => serde_json::from_value(r.get::<Value, _>("weights")).ok(),
            None => None,
        })
    }

    async fn promote_weights(&self, w: &Weights, reason: &str, auc: f64) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("UPDATE weights_versions SET active = FALSE WHERE active")
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "INSERT INTO weights_versions (weights, reason, auc, active) \
             VALUES ($1, $2, $3, TRUE)",
        )
        .bind(serde_json::to_value(w)?)
        .bind(reason)
        .bind(auc)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Fire, Weights};
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.timestamp_opt(1_750_000_000, 0).unwrap()
    }

    #[tokio::test]
    async fn dossier_cache_respects_ttl() {
        let s = MemStore::default();
        let facts = DossierFacts {
            login: "ghost".into(),
            ..Default::default()
        };
        s.put_dossier(&facts, now()).await.unwrap();
        assert!(s.get_dossier("ghost", now()).await.unwrap().is_some());
        let later = now() + Duration::days(DOSSIER_TTL_DAYS + 1);
        assert!(s.get_dossier("ghost", later).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn feedback_becomes_examples_with_correction_weight() {
        let s = MemStore::default();
        let w = Weights::default_table();
        let v = w.score(&[Fire::hit("AGENT_EMAIL")]);
        let id = s.save_verdict("o/r", 1, "ghost", &v).await.unwrap();
        s.add_feedback(id, true, false).await.unwrap();
        s.add_feedback(id, false, true).await.unwrap();
        let ex = s.load_examples().await.unwrap();
        assert_eq!(ex.len(), 2);
        assert_eq!(ex[0].sample_weight, 1.0);
        assert!(ex[0].is_slop);
        assert_eq!(ex[1].sample_weight, 5.0);
        assert!(!ex[1].is_slop);
        assert!(ex[1].fires.iter().any(|f| f.rule == "AGENT_EMAIL"));
    }

    #[tokio::test]
    async fn challenge_state_roundtrip() {
        let s = MemStore::default();
        assert!(s.get_challenge("o/r", 1).await.unwrap().is_none());
        s.put_challenge("o/r", 1, &ChallengeState::Passed)
            .await
            .unwrap();
        assert_eq!(
            s.get_challenge("o/r", 1).await.unwrap(),
            Some(ChallengeState::Passed)
        );
    }

    #[tokio::test]
    async fn weights_promotion_is_visible() {
        let s = MemStore::default();
        assert!(s.active_weights().await.unwrap().is_none());
        let w = Weights::default_table();
        s.promote_weights(&w, "bootstrap", 0.9).await.unwrap();
        assert!(s.active_weights().await.unwrap().is_some());
    }
}
