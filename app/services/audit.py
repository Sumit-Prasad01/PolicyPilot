import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

s = get_settings()

def init_db() -> None:
    conn = sqlite3.connect(s.audit_db_path)
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
    conn.close()


def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    conn = sqlite3.connect(s.audit_db_path)
    conn.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
    )

    conn.commit()
    conn.close()