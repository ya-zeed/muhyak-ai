#!/usr/bin/env python3
"""
Apply pending SQL migrations from the ``migrations/`` directory in name order.

Tracks applied migrations in a ``schema_migrations`` table keyed by filename so
running this multiple times is safe.

Usage:
    python run_migration.py            # apply all pending
    python run_migration.py --list     # show applied vs pending
"""
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from config import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _ensure_tracking_table(conn):
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename VARCHAR(255) PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'UTC')
        )
    """))


def _applied_set(conn) -> set[str]:
    rows = conn.execute(text("SELECT filename FROM schema_migrations")).fetchall()
    return {r[0] for r in rows}


def _pending(conn) -> list[Path]:
    applied = _applied_set(conn)
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return [f for f in files if f.name not in applied]


def list_status():
    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as conn:
        _ensure_tracking_table(conn)
        conn.commit()
        applied = _applied_set(conn)
        files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        print(f"📂 Migrations dir: {MIGRATIONS_DIR}")
        for f in files:
            mark = "✅" if f.name in applied else "⏳"
            print(f"  {mark} {f.name}")
    engine.dispose()


def run_migrations():
    if not MIGRATIONS_DIR.exists():
        print(f"❌ Migrations dir not found: {MIGRATIONS_DIR}")
        sys.exit(1)

    print(f"🔗 Connecting to: {settings.DATABASE_URL}")
    engine = create_engine(settings.DATABASE_URL)

    try:
        with engine.connect() as conn:
            _ensure_tracking_table(conn)
            conn.commit()

            pending = _pending(conn)
            if not pending:
                print("✨ Database is up to date — no pending migrations.")
                return

            for f in pending:
                print(f"\n⚡ Applying {f.name}...")
                sql = f.read_text()
                conn.exec_driver_sql(sql)
                conn.execute(
                    text("INSERT INTO schema_migrations (filename) VALUES (:n)"),
                    {"n": f.name},
                )
                conn.commit()
                print(f"  ✅ {f.name} applied")

            print(f"\n✅ Applied {len(pending)} migration(s).")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_status()
    else:
        run_migrations()
