-- Migration: Add quality analysis features
-- This adds the quality_analyzed column to wedding_images table
-- and creates the quality_analysis_jobs and image_quality_flags tables

-- Add quality_analyzed column to wedding_images if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'wedding_images'
        AND column_name = 'quality_analyzed'
    ) THEN
        ALTER TABLE wedding_images ADD COLUMN quality_analyzed BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- Create quality_analysis_jobs table if it doesn't exist
CREATE TABLE IF NOT EXISTS quality_analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    celebration_id UUID NOT NULL REFERENCES celebrations(id),
    total_images INTEGER NOT NULL,
    processed_count INTEGER DEFAULT 0,
    flagged_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    threshold REAL DEFAULT 0.70,
    started_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'UTC'),
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Create image_quality_flags table if it doesn't exist
CREATE TABLE IF NOT EXISTS image_quality_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL REFERENCES wedding_images(id) ON DELETE CASCADE,
    issue_type VARCHAR(50) NOT NULL,
    confidence REAL NOT NULL,
    reviewed BOOLEAN DEFAULT FALSE,
    dismissed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_quality_analysis_jobs_celebration_id ON quality_analysis_jobs(celebration_id);
CREATE INDEX IF NOT EXISTS idx_quality_analysis_jobs_status ON quality_analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_image_quality_flags_image_id ON image_quality_flags(image_id);
CREATE INDEX IF NOT EXISTS idx_image_quality_flags_issue_type ON image_quality_flags(issue_type);
CREATE INDEX IF NOT EXISTS idx_image_quality_flags_reviewed ON image_quality_flags(reviewed);
