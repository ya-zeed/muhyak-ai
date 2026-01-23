#!/usr/bin/env python3
"""
Run database migration to add quality analysis features.
Usage: python run_migration.py
"""
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from config import settings

def run_migration():
    """Execute the migration SQL file."""
    migration_file = Path(__file__).parent / "migrations" / "001_add_quality_analysis.sql"

    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        sys.exit(1)

    print(f"📄 Reading migration from: {migration_file}")
    migration_sql = migration_file.read_text()

    print(f"🔗 Connecting to: {settings.DATABASE_URL}")
    engine = create_engine(settings.DATABASE_URL)

    try:
        with engine.connect() as conn:
            print("⚡ Executing migration...")
            conn.execute(text(migration_sql))
            conn.commit()
            print("✅ Migration completed successfully!")

            # Verify the changes
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'wedding_images'
                AND column_name = 'quality_analyzed'
            """))

            if result.fetchone():
                print("✅ Verified: quality_analyzed column exists")
            else:
                print("⚠️  Warning: quality_analyzed column not found after migration")

            # Check if quality tables exist
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('quality_analysis_jobs', 'image_quality_flags')
            """))

            tables = [row[0] for row in result.fetchall()]
            if 'quality_analysis_jobs' in tables:
                print("✅ Verified: quality_analysis_jobs table exists")
            if 'image_quality_flags' in tables:
                print("✅ Verified: image_quality_flags table exists")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        engine.dispose()

if __name__ == "__main__":
    run_migration()
