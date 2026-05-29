-- Migration 002: Add embedding_model version tag to face_vectors.
-- Lets the system know which face model produced each vector
-- so we can safely upgrade models in the future.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'face_vectors'
        AND column_name = 'embedding_model'
    ) THEN
        ALTER TABLE face_vectors ADD COLUMN embedding_model VARCHAR(40);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_face_vectors_embedding_model ON face_vectors(embedding_model);

-- Existing rows pre-date the version tag (they were produced by buffalo_s @ 320 with no min-face-size filter).
-- Mark them as 'legacy_buffalo_s' so we can target a clean reprocess.
UPDATE face_vectors SET embedding_model = 'legacy_buffalo_s' WHERE embedding_model IS NULL;
