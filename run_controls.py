"""Local persisted LLM-attempt accounting; not a restart-proof Cloud quota."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def consume_attempt(path, daily_limit):
    if isinstance(daily_limit, bool) or not isinstance(daily_limit, int) or daily_limit < 1:
        raise ValueError("LLM calls are disabled: the daily allowance must be a positive integer.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(path, timeout=5) as database:
        database.execute("CREATE TABLE IF NOT EXISTS attempts (day TEXT PRIMARY KEY, used INTEGER NOT NULL)")
        database.execute("BEGIN IMMEDIATE")
        row = database.execute("SELECT used FROM attempts WHERE day = ?", (today,)).fetchone()
        used = row[0] if row else 0
        if used >= daily_limit:
            raise ValueError("The shared daily AI-call allowance has been reached. Existing results are still available.")
        database.execute("INSERT INTO attempts(day, used) VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET used=excluded.used", (today, used + 1))
    return used + 1


def attempt_usage(path):
    path = Path(path)
    if not path.exists():
        return 0
    with sqlite3.connect(path, timeout=5) as database:
        row = database.execute("SELECT used FROM attempts WHERE day = ?", (datetime.now(timezone.utc).date().isoformat(),)).fetchone()
    return row[0] if row else 0