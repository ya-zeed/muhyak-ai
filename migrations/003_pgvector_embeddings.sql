-- Migration 003: introduce pgvector for indexed nearest-neighbor face search.
--
-- Adds a `vector_pg vector(512)` column alongside the existing `vector` ARRAY(Float),
-- backfills it from existing rows so search keeps working during/after deploy,
-- and creates an HNSW index for cosine-distance ordering.
--
-- The legacy `vector` ARRAY column is kept (nullable later) so we can roll back.

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'face_vectors'
        AND column_name = 'vector_pg'
    ) THEN
        ALTER TABLE face_vectors ADD COLUMN vector_pg vector(512);
    END IF;
END $$;

-- Backfill from the legacy float[] column.
-- The `real[]::vector(512)` cast works from pgvector 0.5.0 onward.
-- Rows whose array isn't exactly 512-d are skipped (they're noise from old experiments).
UPDATE face_vectors
SET vector_pg = vector::real[]::vector(512)
WHERE vector_pg IS NULL
  AND vector IS NOT NULL
  AND array_length(vector, 1) = 512;

-- HNSW (Hierarchical Navigable Small World) index for cosine-similarity ORDER BY.
-- Build params: m=16 (default; good for 512-d embeddings), ef_construction=64 (default).
-- pgvector picks this index automatically for `ORDER BY vector_pg <=> :q LIMIT k`.
CREATE INDEX IF NOT EXISTS idx_face_vectors_vector_pg_hnsw
    ON face_vectors USING hnsw (vector_pg vector_cosine_ops);

-- Useful filter index: most search queries are scoped by celebration_id.
CREATE INDEX IF NOT EXISTS idx_face_vectors_celebration_id ON face_vectors(celebration_id);
