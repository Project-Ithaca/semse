"""Parse Apple Calendar's Calendar.sqlitedb and yield event records for chunking."""
from __future__ import annotations

import datetime
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

from chunker import CalendarEvent

MAC_EPOCH_OFFSET = 978307200  # seconds between Unix epoch and Mac epoch (2001-01-01 UTC)

# Modern macOS keeps the store in the group container; older versions used
# ~/Library/Calendars directly. Probe in order.
CALENDAR_DB_CANDIDATES = (
    Path.home() / "Library" / "Group Containers" / "group.com.apple.calendar" / "Calendar.sqlitedb",
    Path.home() / "Library" / "Calendars" / "Calendar.sqlitedb",
)

# Calendars whose events are noise or duplicates of other indexed sources:
# birthday calendars mirror Contacts, and "Scheduled Reminders" mirrors the
# Reminders store (which we index separately via parse_reminders).
SKIP_CALENDAR_TITLES = {"birthdays", "facebook birthdays", "scheduled reminders"}

EVENT_ENTITY_TYPE = 2  # CalendarItem.entity_type for events (vs legacy todos)


def core_data_ts_to_iso(ts: float | int | None) -> str:
    """Convert Core Data seconds-since-2001 timestamp to naive-UTC ISO 8601."""
    if ts is None:
        return ""
    try:
        return (
            datetime.datetime.fromtimestamp(float(ts) + MAC_EPOCH_OFFSET, tz=datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
        )
    except (OverflowError, OSError, ValueError):
        return ""


def iso_to_core_data_ts(iso: str) -> float:
    """Inverse of core_data_ts_to_iso — naive-UTC ISO string to seconds-since-2001."""
    dt = datetime.datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp() - MAC_EPOCH_OFFSET


def _find_calendar_db() -> Path:
    for candidate in CALENDAR_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Calendar.sqlitedb not found at any of: {', '.join(str(p) for p in CALENDAR_DB_CANDIDATES)}"
    )


def copy_calendar_db_to_temp() -> tuple[str, str]:
    """Copy Calendar.sqlitedb (and -wal/-shm sidecars) to a fresh temp dir.

    Returns `(db_path, tmp_dir)`. The caller MUST clean up `tmp_dir`.
    """
    src_db = _find_calendar_db()
    tmp_dir = tempfile.mkdtemp(prefix="semse_calendar_")
    try:
        dest = Path(tmp_dir) / src_db.name
        shutil.copy2(src_db, dest)
        for suffix in ("-wal", "-shm"):
            sidecar = src_db.parent / (src_db.name + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, Path(tmp_dir) / sidecar.name)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return str(dest), tmp_dir


def _detect_item_columns(conn: sqlite3.Connection) -> dict[str, str | None]:
    """CalendarItem column names drift across macOS versions; auto-detect."""
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(CalendarItem)")
    cols = {row[1].lower() for row in cur.fetchall()}
    pick = lambda *cands: next((c for c in cands if c in cols), None)
    return {
        "summary": pick("summary", "title"),
        "start_date": pick("start_date"),
        "end_date": pick("end_date"),
        "all_day": pick("all_day"),
        "calendar_id": pick("calendar_id"),
        "description": pick("description", "notes"),
        "location_id": pick("location_id"),
        "entity_type": pick("entity_type"),
        "status": pick("status"),
    }


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def iter_events(db_path: str | None = None, since_iso: str | None = None) -> Iterator[CalendarEvent]:
    """Yield calendar events sorted by start date ASC.

    `since_iso` (when set) filters at SQL level — only events whose end date is
    strictly newer than the cutoff are returned, matching the additive
    `--update` watermark semantics of the other sources.
    """
    tmp_dir: str | None = None
    if db_path:
        path = db_path
    else:
        path, tmp_dir = copy_calendar_db_to_temp()
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = _detect_item_columns(conn)
        if not cols["summary"] or not cols["start_date"] or not cols["end_date"]:
            raise FileNotFoundError(f"CalendarItem schema unrecognized in {path}")

        cal_join = ""
        cal_select = "NULL AS calendar_name"
        if cols["calendar_id"] and _table_exists(conn, "Calendar"):
            cal_join = f"LEFT JOIN Calendar c ON ci.{cols['calendar_id']} = c.ROWID"
            cal_select = "c.title AS calendar_name"

        loc_join = ""
        loc_select = "NULL AS loc_title, NULL AS loc_address"
        if cols["location_id"] and _table_exists(conn, "Location"):
            loc_join = f"LEFT JOIN Location l ON ci.{cols['location_id']} = l.ROWID"
            loc_select = "l.title AS loc_title, l.address AS loc_address"

        desc_select = f"ci.{cols['description']}" if cols["description"] else "NULL"
        all_day_select = f"ci.{cols['all_day']}" if cols["all_day"] else "0"

        where = [f"ci.{cols['summary']} IS NOT NULL", f"trim(ci.{cols['summary']}) != ''"]
        if cols["entity_type"]:
            where.append(f"ci.{cols['entity_type']} = {EVENT_ENTITY_TYPE}")
        if since_iso:
            where.append(f"ci.{cols['end_date']} > {iso_to_core_data_ts(since_iso)}")

        query = f"""
            SELECT
              ci.ROWID,
              ci.{cols['summary']} AS summary,
              ci.{cols['start_date']} AS start_ts,
              ci.{cols['end_date']} AS end_ts,
              {all_day_select} AS all_day,
              {desc_select} AS notes,
              {cal_select},
              {loc_select}
            FROM CalendarItem ci
            {cal_join}
            {loc_join}
            WHERE {' AND '.join(where)}
            ORDER BY ci.{cols['start_date']} ASC
        """
        for row in conn.execute(query):
            (row_id, summary, start_ts, end_ts, all_day, notes,
             calendar_name, loc_title, loc_address) = row
            if calendar_name and calendar_name.strip().lower() in SKIP_CALENDAR_TITLES:
                continue
            start_iso = core_data_ts_to_iso(start_ts)
            if not start_iso:
                continue
            location = " — ".join(
                p.strip() for p in (loc_title, loc_address) if p and p.strip()
            )
            yield CalendarEvent(
                row_id=int(row_id),
                title=str(summary).strip(),
                start_iso=start_iso,
                end_iso=core_data_ts_to_iso(end_ts) or start_iso,
                all_day=bool(all_day),
                calendar_name=(calendar_name or "").strip() or None,
                location=location or None,
                notes=(notes or "").strip() or None,
            )
    finally:
        conn.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    events = list(iter_events())
    print(f"{len(events):,} events")
    for ev in events[:3]:
        print(f"  [{ev.start_iso}] {ev.title[:60]} ({ev.calendar_name}) @ {ev.location}")
