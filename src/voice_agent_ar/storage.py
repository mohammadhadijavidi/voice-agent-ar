"""SQLite storage for calls, transcripts, events, bookings and the webhook outbox.

One file, no server. Operations are tiny and run on the event loop thread; for a
high-volume deployment swap this class for Postgres with the same methods.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  id TEXT PRIMARY KEY,
  call_sid TEXT, stream_sid TEXT,
  from_number TEXT, to_number TEXT,
  direction TEXT DEFAULT 'inbound',
  language TEXT, status TEXT DEFAULT 'in_progress',
  outcome TEXT, summary TEXT,
  started_at REAL, ended_at REAL,
  turns INTEGER DEFAULT 0, barge_ins INTEGER DEFAULT 0,
  llm_ms_avg REAL, first_audio_ms_avg REAL,
  providers TEXT, simulated INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  call_id TEXT, ts REAL, role TEXT, text TEXT, language TEXT,
  interrupted INTEGER DEFAULT 0, latency_ms REAL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  call_id TEXT, ts REAL, kind TEXT, data TEXT
);
CREATE TABLE IF NOT EXISTS bookings (
  id TEXT PRIMARY KEY,
  call_id TEXT, created_at REAL,
  code TEXT, kind TEXT,
  service_id TEXT, service_name TEXT, doctor_id TEXT,
  party_size INTEGER, area TEXT,
  date TEXT, time TEXT, duration_min INTEGER,
  customer_name TEXT, phone TEXT, notes TEXT, language TEXT,
  status TEXT DEFAULT 'confirmed'
);
CREATE TABLE IF NOT EXISTS notifications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  booking_id TEXT, call_id TEXT, created_at REAL,
  channel TEXT, to_number TEXT, body TEXT, request TEXT,
  status TEXT, provider_id TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at REAL, event TEXT, payload TEXT,
  attempts INTEGER DEFAULT 0, delivered_at REAL, last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_call ON turns(call_id, id);
CREATE INDEX IF NOT EXISTS idx_events_call ON events(call_id, id);
CREATE INDEX IF NOT EXISTS idx_bookings_slot ON bookings(date, time);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_code ON bookings(code);
CREATE INDEX IF NOT EXISTS idx_calls_started ON calls(started_at);
"""


class Store:
    def __init__(self, path: str = "voice_agent.db"):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> dict | None:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    # ---- calls ----------------------------------------------------------
    def create_call(self, *, call_sid: str = "", stream_sid: str = "", from_number: str = "",
                    to_number: str = "", direction: str = "inbound", language: str = "en",
                    providers: dict | None = None, simulated: bool = False,
                    started_at: float | None = None) -> str:
        cid = "call_" + uuid.uuid4().hex[:12]
        self._exec(
            "INSERT INTO calls (id, call_sid, stream_sid, from_number, to_number, direction, language,"
            " started_at, providers, simulated) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cid, call_sid, stream_sid, from_number, to_number, direction, language,
             started_at or time.time(), json.dumps(providers or {}), int(simulated)))
        return cid

    def update_call(self, call_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE calls SET {cols} WHERE id=?", (*fields.values(), call_id))

    def get_call(self, call_id: str) -> dict | None:
        c = self._one("SELECT * FROM calls WHERE id=?", (call_id,))
        if c:
            c["providers"] = json.loads(c["providers"] or "{}")
        return c

    def list_calls(self, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self._all(
            "SELECT c.*, (SELECT COUNT(*) FROM bookings b WHERE b.call_id=c.id) AS bookings "
            "FROM calls c ORDER BY started_at DESC LIMIT ? OFFSET ?", (limit, offset))
        for r in rows:
            r["providers"] = json.loads(r["providers"] or "{}")
        return rows

    # ---- transcript / events ------------------------------------------
    def add_turn(self, call_id: str, role: str, text: str, language: str = "", *,
                 interrupted: bool = False, latency_ms: float | None = None,
                 ts: float | None = None) -> int:
        cur = self._exec(
            "INSERT INTO turns (call_id, ts, role, text, language, interrupted, latency_ms)"
            " VALUES (?,?,?,?,?,?,?)",
            (call_id, ts or time.time(), role, text, language, int(interrupted), latency_ms))
        return cur.lastrowid

    def mark_interrupted(self, turn_id: int, spoken_text: str) -> None:
        self._exec("UPDATE turns SET interrupted=1, text=? WHERE id=?", (spoken_text, turn_id))

    def turns(self, call_id: str) -> list[dict]:
        return self._all("SELECT * FROM turns WHERE call_id=? ORDER BY id", (call_id,))

    def add_event(self, call_id: str, kind: str, data: dict | None = None, ts: float | None = None) -> None:
        self._exec("INSERT INTO events (call_id, ts, kind, data) VALUES (?,?,?,?)",
                   (call_id, ts or time.time(), kind, json.dumps(data or {}, ensure_ascii=False)))

    def events(self, call_id: str) -> list[dict]:
        rows = self._all("SELECT * FROM events WHERE call_id=? ORDER BY id", (call_id,))
        for r in rows:
            r["data"] = json.loads(r["data"] or "{}")
        return rows

    # ---- bookings ------------------------------------------------------
    def overlapping(self, date: str, start_min: int, end_min: int, **match) -> list[dict]:
        """Confirmed bookings on ``date`` overlapping [start_min, end_min); optional column filters."""
        sql = "SELECT * FROM bookings WHERE date=? AND status='confirmed'"
        args: list = [date]
        for k, v in match.items():
            if k not in ("area", "doctor_id", "service_id"):
                raise ValueError(k)
            sql += f" AND {k}=?"
            args.append(v)
        out = []
        for b in self._all(sql, tuple(args)):
            h, m = map(int, b["time"].split(":"))
            bs = h * 60 + m
            if bs < end_min and start_min < bs + b["duration_min"]:
                out.append(b)
        return out

    def get_booking(self, booking_id_or_code: str) -> dict | None:
        return self._one("SELECT * FROM bookings WHERE id=? OR code=?", (booking_id_or_code, booking_id_or_code))

    def update_booking(self, booking_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE bookings SET {cols} WHERE id=?", (*fields.values(), booking_id))

    # ---- notifications (WhatsApp / SMS confirmations) --------------------
    def add_notification(self, **n) -> int:
        n.setdefault("created_at", time.time())
        for k in ("request",):
            if isinstance(n.get(k), (dict, list)):
                n[k] = json.dumps(n[k], ensure_ascii=False)
        cols = ", ".join(n)
        cur = self._exec(f"INSERT INTO notifications ({cols}) VALUES ({','.join('?' * len(n))})",
                         tuple(n.values()))
        return cur.lastrowid

    def notifications(self, booking_id: str | None = None, call_id: str | None = None) -> list[dict]:
        if booking_id:
            rows = self._all("SELECT * FROM notifications WHERE booking_id=? ORDER BY id", (booking_id,))
        elif call_id:
            rows = self._all("SELECT * FROM notifications WHERE call_id=? ORDER BY id", (call_id,))
        else:
            rows = self._all("SELECT * FROM notifications ORDER BY id DESC LIMIT 200")
        for r in rows:
            try:
                r["request"] = json.loads(r["request"]) if r.get("request") else None
            except json.JSONDecodeError:
                pass
        return rows

    def bookings_on(self, date: str) -> list[dict]:
        return self._all("SELECT * FROM bookings WHERE date=? AND status='confirmed' ORDER BY time", (date,))

    def create_booking(self, **b) -> dict:
        b.setdefault("id", "bk_" + uuid.uuid4().hex[:10])
        b.setdefault("created_at", time.time())
        cols = ", ".join(b)
        self._exec(f"INSERT INTO bookings ({cols}) VALUES ({','.join('?' * len(b))})", tuple(b.values()))
        return self._one("SELECT * FROM bookings WHERE id=?", (b["id"],))

    def list_bookings(self, call_id: str | None = None, limit: int = 100) -> list[dict]:
        if call_id:
            return self._all("SELECT * FROM bookings WHERE call_id=? ORDER BY created_at", (call_id,))
        return self._all("SELECT * FROM bookings ORDER BY date DESC, time DESC LIMIT ?", (limit,))

    # ---- outbox --------------------------------------------------------
    def outbox_add(self, event: str, payload: dict) -> int:
        cur = self._exec("INSERT INTO outbox (created_at, event, payload) VALUES (?,?,?)",
                         (time.time(), event, json.dumps(payload, ensure_ascii=False)))
        return cur.lastrowid

    def outbox_pending(self, max_attempts: int = 8) -> list[dict]:
        rows = self._all("SELECT * FROM outbox WHERE delivered_at IS NULL AND attempts < ? ORDER BY id",
                         (max_attempts,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def outbox_result(self, oid: int, ok: bool, error: str = "") -> None:
        if ok:
            self._exec("UPDATE outbox SET attempts=attempts+1, delivered_at=?, last_error=NULL WHERE id=?",
                       (time.time(), oid))
        else:
            self._exec("UPDATE outbox SET attempts=attempts+1, last_error=? WHERE id=?", (error[:500], oid))

    def outbox_all(self) -> list[dict]:
        rows = self._all("SELECT * FROM outbox ORDER BY id")
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    # ---- stats ---------------------------------------------------------
    def stats(self) -> dict:
        row = self._one(
            "SELECT COUNT(*) AS calls, SUM(CASE WHEN status='in_progress' THEN 1 ELSE 0 END) AS live,"
            " AVG(CASE WHEN ended_at IS NOT NULL THEN ended_at-started_at END) AS avg_duration,"
            " SUM(barge_ins) AS barge_ins, AVG(first_audio_ms_avg) AS first_audio_ms FROM calls") or {}
        # bookings made on calls (excludes rows imported or seeded from elsewhere)
        row["bookings"] = (self._one("SELECT COUNT(*) AS n FROM bookings b JOIN calls c ON c.id=b.call_id "
                                     "WHERE b.status='confirmed'") or {}).get("n", 0)
        row["notifications_sent"] = (self._one(
            "SELECT COUNT(*) AS n FROM notifications WHERE status IN ('sent','dry_run')") or {}).get("n", 0)
        row["transfers"] = (self._one("SELECT COUNT(*) AS n FROM calls WHERE outcome='transferred'") or {}).get("n", 0)
        langs = self._all("SELECT language, COUNT(*) AS n FROM calls GROUP BY language")
        row["languages"] = {r["language"] or "?": r["n"] for r in langs}
        return row

    def close(self) -> None:
        with self._lock:
            self._db.close()
