-- The system of record. Everything the service observes, decides, and learns
-- lives here: events, signatures, dossiers, verdicts, actions, challenges,
-- maintainer feedback, weight versions, and federation records.

CREATE TABLE installations (
    id          BIGINT PRIMARY KEY,
    owner       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE repos (
    full_name        TEXT PRIMARY KEY,
    installation_id  BIGINT NOT NULL REFERENCES installations (id),
    config_yaml      TEXT,
    -- Baseline PR arrivals per burst window, for Poisson surprise.
    baseline_per_window DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pr_events (
    id          BIGSERIAL PRIMARY KEY,
    repo        TEXT NOT NULL,
    pr_number   BIGINT NOT NULL,
    author      TEXT NOT NULL,
    action      TEXT NOT NULL,
    payload     JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX pr_events_repo_ix ON pr_events (repo, pr_number);

-- Campaign signatures; the in-memory cluster store is rebuilt from the TTL
-- window of these on startup.
CREATE TABLE signatures (
    id             BIGSERIAL PRIMARY KEY,
    repo           TEXT NOT NULL,
    pr_number      BIGINT NOT NULL,
    arrived        TIMESTAMPTZ NOT NULL,
    simhash        BIGINT,
    pathset_hash   BIGINT NOT NULL,
    text_band_keys BIGINT[] NOT NULL DEFAULT '{}',
    style          JSONB NOT NULL
);
CREATE INDEX signatures_arrived_ix ON signatures (arrived);

-- Author dossiers, cached with a TTL; the classification history of every
-- account the network has seen.
CREATE TABLE dossiers (
    login      TEXT PRIMARY KEY,
    facts      JSONB NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE verdicts (
    id          BIGSERIAL PRIMARY KEY,
    repo        TEXT NOT NULL,
    pr_number   BIGINT NOT NULL,
    author      TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    tier        TEXT NOT NULL,
    evidence    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX verdicts_repo_ix ON verdicts (repo, pr_number);
CREATE INDEX verdicts_author_ix ON verdicts (author);

CREATE TABLE actions_log (
    id         BIGSERIAL PRIMARY KEY,
    repo       TEXT NOT NULL,
    pr_number  BIGINT NOT NULL,
    verdict_id BIGINT REFERENCES verdicts (id),
    action     TEXT NOT NULL,
    dry_run    BOOLEAN NOT NULL,
    detail     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE challenges (
    repo       TEXT NOT NULL,
    pr_number  BIGINT NOT NULL,
    state      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, pr_number)
);

-- Maintainer feedback: the training labels for the learner. A correction is
-- a maintainer overriding one of our verdicts.
CREATE TABLE feedback (
    id         BIGSERIAL PRIMARY KEY,
    verdict_id BIGINT NOT NULL REFERENCES verdicts (id),
    is_slop    BOOLEAN NOT NULL,
    correction BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every weight table ever active; promotion inserts a row and flips active.
CREATE TABLE weights_versions (
    id         BIGSERIAL PRIMARY KEY,
    weights    JSONB NOT NULL,
    reason     TEXT NOT NULL,
    auc        DOUBLE PRECISION,
    active     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX weights_one_active ON weights_versions (active) WHERE active;

-- Federation intake (dormant in the MVP): signed records from peers.
CREATE TABLE federation_records (
    id          BIGSERIAL PRIMARY KEY,
    envelope    JSONB NOT NULL,
    record      JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
