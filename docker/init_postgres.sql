-- PostgreSQL initialization script for Expedia Travel Advertising Intelligence Platform
CREATE TABLE IF NOT EXISTS etl_audit_log (
    id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(128) NOT NULL,
    run_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(32) NOT NULL,
    details JSONB
);
