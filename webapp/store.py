"""
webapp/store.py

SQLite store for batch results. One row per judged batch (OK or NC), written
by webapp/runner.py the moment PadPullMonitor closes a batch. The Overview and
Analytics pages read from here.

Timestamps are local wall-clock time ("YYYY-MM-DD HH:MM:SS") at the moment the
batch was judged. That also applies when replaying a recorded video, so a demo
run lands on "now". Day/hour bucketing is plain string slicing.
"""
from __future__ import annotations

import sqlite3
import threading
from calendar import monthrange
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "analytics.db"
HOUR_LABELS = [f"{h:02d}:00" for h in range(24)]

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source TEXT,
    batch_no INTEGER NOT NULL,
    ts TEXT NOT NULL,
    pads_in INTEGER NOT NULL,
    pulled INTEGER NOT NULL,
    leftover INTEGER NOT NULL,
    result TEXT NOT NULL,          -- 'OK' or 'NC'
    reason TEXT,
    snapshot TEXT                  -- file name in webapp/snapshots/ (NC only)
);
CREATE INDEX IF NOT EXISTS idx_batches_ts ON batches(ts);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _lock, closing(_connect()) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def _iso(dt: datetime) -> str:
    return dt.isoformat(sep=" ", timespec="seconds")


def record_batch(run_id: str, source: str, batch_no: int, pads_in: int, pulled: int,
                 leftover: int, result: str, reason: str, snapshot: str | None,
                 ts: datetime | None = None) -> None:
    with _lock, closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO batches (run_id, source, batch_no, ts, pads_in, pulled, leftover, result, reason, snapshot)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, source, batch_no, _iso(ts or datetime.now()), pads_in, pulled, leftover,
             result, reason, snapshot),
        )
        conn.commit()


def batches_between(start: datetime, end: datetime) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM batches WHERE ts >= ? AND ts < ? ORDER BY ts ASC", (_iso(start), _iso(end))
        ).fetchall()
    return [dict(r) for r in rows]


def recent_nc(limit: int = 10) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM batches WHERE result = 'NC' ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ----------------------------------------------------------------- aggregation
def _empty() -> dict:
    return {"ok": 0, "nc": 0, "pulled": 0, "left": 0}


def _add(bucket: dict, b: dict) -> None:
    bucket["ok" if b["result"] == "OK" else "nc"] += 1
    bucket["pulled"] += b["pulled"]
    bucket["left"] += b["leftover"]


def _by_hour(batches: list[dict]) -> list[dict]:
    buckets = [_empty() for _ in range(24)]
    for b in batches:
        _add(buckets[int(b["ts"][11:13])], b)
    return buckets


def _by_day(batches: list[dict], days: list[date]) -> list[dict]:
    index = {d.isoformat(): i for i, d in enumerate(days)}
    buckets = [_empty() for _ in days]
    for b in batches:
        i = index.get(b["ts"][:10])
        if i is not None:
            _add(buckets[i], b)
    return buckets


def _average(buckets: list[dict], n: int) -> list[dict]:
    return [{k: round(v / n, 2) for k, v in b.items()} for b in buckets] if n > 0 else buckets


def totals(batches: list[dict]) -> dict:
    n = len(batches)
    nc = sum(1 for b in batches if b["result"] == "NC")
    return {
        "batches": n,
        "ok": n - nc,
        "nc": nc,
        "nc_rate": round(100 * nc / n, 1) if n else 0.0,
        "pads": sum(b["pads_in"] for b in batches),
        "pulled": sum(b["pulled"] for b in batches),
        "left": sum(b["leftover"] for b in batches),
    }


def _series(labels: list[str], buckets: list[dict]) -> dict:
    return {
        "labels": labels,
        "batches": {"ok": [b["ok"] for b in buckets], "nc": [b["nc"] for b in buckets]},
        "pads": {"pulled": [b["pulled"] for b in buckets], "left": [b["left"] for b in buckets]},
    }


def overview(today: date) -> dict:
    start = datetime.combine(today, datetime.min.time())
    batches = batches_between(start, start + timedelta(days=1))
    return {"date": today.isoformat(), "totals": totals(batches),
            **_series(HOUR_LABELS, _by_hour(batches)), "recent_nc": recent_nc(8)}


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def analytics(range_: str, ref_date: date | None = None, mode: str = "daily",
              from_date: date | None = None, to_date: date | None = None) -> dict:
    """range_: day | week | month | custom. mode (week/month only): daily | hourly_avg."""
    ref_date = ref_date or date.today()
    if range_ == "day":
        first = last = ref_date
    elif range_ == "week":
        first = ref_date - timedelta(days=ref_date.weekday())
        last = first + timedelta(days=6)
    elif range_ == "month":
        first = ref_date.replace(day=1)
        last = ref_date.replace(day=monthrange(ref_date.year, ref_date.month)[1])
    elif range_ == "custom":
        if from_date is None or to_date is None:
            raise ValueError("range='custom' requires from and to dates")
        first, last = sorted((from_date, to_date))
    else:
        raise ValueError(f"unknown range {range_!r}")

    start = datetime.combine(first, datetime.min.time())
    end = datetime.combine(last, datetime.min.time()) + timedelta(days=1)
    batches = batches_between(start, end)
    days = _days(first, last)

    if range_ == "day":
        series = _series(HOUR_LABELS, _by_hour(batches))
    elif range_ in ("week", "month") and mode == "hourly_avg":
        series = _series(HOUR_LABELS, _average(_by_hour(batches), len(days)))
    else:
        series = _series([d.strftime("%a %m/%d") for d in days], _by_day(batches, days))

    return {"from": first.isoformat(), "to": last.isoformat(), "totals": totals(batches),
            **series, "rows": batches[::-1]}
