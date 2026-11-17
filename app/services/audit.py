import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import BASE_DIR, get_settings

s = get_settings()


def _connect() -> sqlite3.Connection:
    db_path = Path(s.audit_db_path)
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )"""
        )
        conn.commit()


def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
        )
        conn.commit()
