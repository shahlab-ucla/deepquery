-- DuckDB/Parquet-facing canonical tables. JSONL remains the executable starter format.
-- Production loaders should write immutable Parquet partitions and expose these names as views.
-- Every load must first pass the runtime collection validator; SQL constraints are defense in depth.

CREATE TABLE IF NOT EXISTS source_snapshot (
    snapshot_id VARCHAR PRIMARY KEY,
    source_id VARCHAR NOT NULL,
    source_release VARCHAR NOT NULL,
    source_artifact VARCHAR NOT NULL,
    manifest_sha256 VARCHAR,
    artifact_sha256 VARCHAR,
    source_url VARCHAR,
    retrieved_at TIMESTAMP WITH TIME ZONE,
    license_uri VARCHAR,
    transformer_activity_id VARCHAR,
    transformer_version VARCHAR,
    UNIQUE (source_id, source_release, source_artifact),
    CHECK (manifest_sha256 IS NULL OR length(manifest_sha256) = 64),
    CHECK (artifact_sha256 IS NULL OR length(artifact_sha256) = 64)
);

CREATE TABLE IF NOT EXISTS contextual_observation (
    observation_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    predicate_id VARCHAR NOT NULL,
    object_id VARCHAR NOT NULL,
    interpretation VARCHAR NOT NULL,
    observation_status VARCHAR NOT NULL,
    proposition_negated BOOLEAN NOT NULL DEFAULT FALSE,
    record_origin VARCHAR NOT NULL,
    assertion_method_id VARCHAR,
    qualifiers_json JSON NOT NULL,
    biolink_knowledge_level VARCHAR,
    biolink_agent_type VARCHAR,
    snapshot_id VARCHAR NOT NULL,
    context_json JSON NOT NULL,
    canonical_observation_json JSON NOT NULL,
    canonical_sha256 VARCHAR NOT NULL CHECK (length(canonical_sha256) = 64),
    CHECK (interpretation IN ('observation', 'association', 'causal_claim')),
    CHECK (observation_status IN ('positive', 'explicit_negative', 'uncertain')),
    CHECK (record_origin IN ('curated', 'author_reported', 'extracted', 'computed')),
    CHECK (
        biolink_knowledge_level IS NULL
        OR biolink_knowledge_level IN (
            'knowledge_assertion', 'logical_entailment', 'prediction',
            'statistical_association', 'observation',
            'not_provided'
        )
    ),
    CHECK (
        biolink_agent_type IS NULL
        OR biolink_agent_type IN (
            'manual_agent', 'automated_agent', 'data_analysis_pipeline',
            'computational_model', 'text_mining_agent',
            'image_processing_agent', 'manual_validation_of_automated_agent',
            'not_provided'
        )
    ),
    CHECK (
        (observation_status = 'positive' AND proposition_negated = FALSE)
        OR (observation_status = 'explicit_negative' AND proposition_negated = TRUE)
        OR observation_status = 'uncertain'
    ),
    FOREIGN KEY (snapshot_id) REFERENCES source_snapshot(snapshot_id)
);

CREATE TABLE IF NOT EXISTS observation_context_value (
    context_value_id VARCHAR PRIMARY KEY,
    observation_id VARCHAR NOT NULL,
    dimension VARCHAR NOT NULL,
    value_id VARCHAR,
    source_value VARCHAR,
    mapping_record_id VARCHAR,
    mapping_status VARCHAR,
    reportedness VARCHAR NOT NULL,
    missing_reason VARCHAR,
    CHECK (
        reportedness IN (
            'reported', 'not_reported', 'not_applicable', 'ambiguous',
            'explicitly_absent'
        )
    ),
    FOREIGN KEY (observation_id) REFERENCES contextual_observation(observation_id)
);

CREATE TABLE IF NOT EXISTS observation_result (
    result_id VARCHAR PRIMARY KEY,
    observation_id VARCHAR NOT NULL UNIQUE,
    result_type VARCHAR NOT NULL,
    summary VARCHAR,
    CHECK (
        result_type IN (
            'qualitative', 'quantitative', 'categorical', 'trajectory',
            'image_derived', 'textual_claim', 'mixed'
        )
    ),
    FOREIGN KEY (observation_id) REFERENCES contextual_observation(observation_id)
);

CREATE TABLE IF NOT EXISTS result_measurement (
    measurement_id VARCHAR PRIMARY KEY,
    result_id VARCHAR NOT NULL,
    measured_feature_id VARCHAR NOT NULL,
    value_json JSON NOT NULL,
    statistic_id VARCHAR,
    uncertainty_json JSON,
    uncertainty_type_id VARCHAR,
    confidence_level DOUBLE CHECK (confidence_level BETWEEN 0 AND 1),
    p_value DOUBLE CHECK (p_value BETWEEN 0 AND 1),
    penetrance_json JSON,
    sample_size INTEGER CHECK (sample_size IS NULL OR sample_size > 0),
    biological_replicates INTEGER CHECK (
        biological_replicates IS NULL OR biological_replicates > 0
    ),
    technical_replicates INTEGER CHECK (
        technical_replicates IS NULL OR technical_replicates > 0
    ),
    measurement_time_json JSON,
    notes VARCHAR,
    FOREIGN KEY (result_id) REFERENCES observation_result(result_id)
);

CREATE TABLE IF NOT EXISTS result_comparison_group (
    comparison_group_id VARCHAR PRIMARY KEY,
    result_id VARCHAR NOT NULL,
    label VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    group_size INTEGER CHECK (group_size IS NULL OR group_size > 0),
    members_json JSON,
    genetic_background_json JSON,
    notes VARCHAR,
    FOREIGN KEY (result_id) REFERENCES observation_result(result_id)
);

CREATE TABLE IF NOT EXISTS measurement_comparison_group (
    measurement_id VARCHAR NOT NULL,
    comparison_group_id VARCHAR NOT NULL,
    PRIMARY KEY (measurement_id, comparison_group_id),
    FOREIGN KEY (measurement_id) REFERENCES result_measurement(measurement_id),
    FOREIGN KEY (comparison_group_id) REFERENCES result_comparison_group(comparison_group_id)
);

CREATE TABLE IF NOT EXISTS result_asset (
    asset_id VARCHAR PRIMARY KEY,
    result_id VARCHAR NOT NULL,
    uri VARCHAR NOT NULL,
    checksum_sha256 VARCHAR NOT NULL,
    media_type VARCHAR NOT NULL,
    byte_size BIGINT CHECK (byte_size IS NULL OR byte_size >= 0),
    role VARCHAR,
    CHECK (length(checksum_sha256) = 64),
    CHECK (
        lower(uri) NOT LIKE 'urn:sha256:%'
        OR lower(substr(uri, 12)) = lower(checksum_sha256)
    ),
    FOREIGN KEY (result_id) REFERENCES observation_result(result_id)
);

CREATE TABLE IF NOT EXISTS measurement_asset (
    measurement_id VARCHAR NOT NULL,
    asset_id VARCHAR NOT NULL,
    PRIMARY KEY (measurement_id, asset_id),
    FOREIGN KEY (measurement_id) REFERENCES result_measurement(measurement_id),
    FOREIGN KEY (asset_id) REFERENCES result_asset(asset_id)
);

CREATE TABLE IF NOT EXISTS evidence_line (
    evidence_id VARCHAR PRIMARY KEY,
    evidence_type_id VARCHAR NOT NULL,
    direction VARCHAR NOT NULL,
    source_record_id VARCHAR NOT NULL,
    figure_or_table VARCHAR,
    notes VARCHAR,
    CHECK (direction IN ('supports', 'disputes', 'neutral'))
);

CREATE TABLE IF NOT EXISTS observation_evidence (
    observation_id VARCHAR NOT NULL,
    evidence_id VARCHAR NOT NULL,
    PRIMARY KEY (observation_id, evidence_id),
    FOREIGN KEY (observation_id) REFERENCES contextual_observation(observation_id),
    FOREIGN KEY (evidence_id) REFERENCES evidence_line(evidence_id)
);

CREATE TABLE IF NOT EXISTS evidence_publication (
    evidence_id VARCHAR NOT NULL,
    publication_id VARCHAR NOT NULL,
    PRIMARY KEY (evidence_id, publication_id),
    FOREIGN KEY (evidence_id) REFERENCES evidence_line(evidence_id)
);

CREATE VIEW IF NOT EXISTS transport_coverage_long AS
SELECT
    o.observation_id,
    s.source_id,
    s.source_release,
    s.source_artifact,
    s.artifact_sha256,
    c.dimension,
    c.value_id,
    c.reportedness,
    c.missing_reason
FROM contextual_observation o
JOIN source_snapshot s USING (snapshot_id)
JOIN observation_context_value c USING (observation_id);
