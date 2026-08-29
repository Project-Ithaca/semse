"""Parse the Apple Reminders Core Data stores and yield reminder records."""
from __future__ import annotations

import glob
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

from chunker import ReminderItem
from parse_calendar import core_data_ts_to_iso

# Modern macOS keeps the stores in the group container; older versions used
# ~/Library/Reminders directly. There may be several Data-*.sqlite store files
# (one per account) — iterate all of them.
REMINDERS_STORE_GLOBS = (
    str(
        Path.home()
        / "Library" / "Group Containers" / "group.com.apple.reminders"
        / "Container_v1" / "Stores" / "Data-*.sqlite"
    ),
    str(Path.home() / "Library" / "Reminders" / "Container_v1" / "Stores" / "Data-*.sqlite"),
)


def _find_store_paths() -> list[Path]:
    for pattern in REMINDERS_STORE_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return [Path(h) for h in hits]
    raise FileNotFoundError(
        f"No Reminders stores found at any of: {', '.join(REMINDERS_STORE_GLOBS)}"
    )


def copy_store_to_temp(src_db: Path) -> tuple[str, str]:
    """Copy one Data-*.sqlite store (and -wal/-shm sidecars) to a fresh temp dir.

    Returns `(db_path, tmp_dir)`. The caller MUST clean up `tmp_dir`.
    """
    tmp_dir = tempfile.mkdtemp(prefix="semse_reminders_")
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


def _detect_reminder_columns(conn: sqlite3.Connection) -> dict[str, str | None] | None:
    """ZREMCDREMINDER columns drift across macOS versions; auto-detect.

    Returns None when the store has no recognizable reminder table.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ZREMCDREMINDER'"
    )
    if not cur.fetchone():
        return None
    cur = conn.execute("PRAGMA table_info(ZREMCDREMINDER)")
    cols = {row[1].upper() for row in cur.fetchall()}
    pick = lambda *cands: next((c for c in cands if c in cols), None)
    title = pick("ZTITLE")
    if not title:
        return None
    return {
        "title": title,
        "notes": pick("ZNOTES"),
        "completed": pick("ZCOMPLETED"),
        "due_date": pick("ZDUEDATE", "ZDISPLAYDATEDATE"),
        "completion_date": pick("ZCOMPLETIONDATE"),
        "creation_date": pick("ZCREATIONDATE"),
        "list_fk": pick("ZLIST"),
        "deleted": pick("ZMARKEDFORDELETION"),
    }


def _iter_store(db_path: str, since_iso: str | None) -> Iterator[ReminderItem]:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = _detect_reminder_columns(conn)
        if cols is None:
            return

        list_join = ""
        list_select = "NULL AS list_name"
        if cols["list_fk"]:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ZREMCDBASELIST'"
            )
            if cur.fetchone():
                list_join = f"LEFT JOIN ZREMCDBASELIST l ON r.{cols['list_fk']} = l.Z_PK"
                list_select = "l.ZNAME AS list_name"

        sel = lambda key: f"r.{cols[key]}" if cols[key] else "NULL"
        where = [f"r.{cols['title']} IS NOT NULL", f"trim(r.{cols['title']}) != ''"]
        if cols["deleted"]:
            where.append(f"COALESCE(r.{cols['deleted']}, 0) = 0")

        query = f"""
            SELECT
              r.Z_PK,
              r.{cols['title']} AS title,
              {sel('notes')} AS notes,
              COALESCE({sel('completed')}, 0) AS completed,
              {sel('due_date')} AS due_ts,
              {sel('completion_date')} AS completion_ts,
              {sel('creation_date')} AS creation_ts,
              {list_select}
            FROM ZREMCDREMINDER r
            {list_join}
            WHERE {' AND '.join(where)}
        """
        for row in conn.execute(query):
            (row_id, title, notes, completed, due_ts, completion_ts,
             creation_ts, list_name) = row
            due_iso = core_data_ts_to_iso(due_ts)
            completion_iso = core_data_ts_to_iso(completion_ts)
            creation_iso = core_data_ts_to_iso(creation_ts)
            # Best-available date for chunk date ranges and the --update
            # watermark: completion beats due beats creation.
            date_iso = completion_iso or due_iso or creation_iso
            if not date_iso:
                continue
            if since_iso and date_iso <= since_iso:
                continue
            yield ReminderItem(
                row_id=int(row_id),
                title=str(title).strip(),
                notes=(notes or "").strip() or None,
                completed=bool(completed),
                due_iso=due_iso,
                completion_iso=completion_iso,
                date_iso=date_iso,
                list_name=(list_name or "").strip() or None,
            )
    finally:
        conn.close()


def iter_reminders(
    db_paths: list[str] | None = None, since_iso: str | None = None
) -> Iterator[ReminderItem]:
    """Yield reminders from every store file, sorted by date ASC.

    `since_iso` (when set) skips reminders whose best-available date
    (completion → due → creation) is `<= since_iso` — the `--update` path.
    """
    items: list[ReminderItem] = []
    if db_paths:
        for p in db_paths:
            items.extend(_iter_store(p, since_iso))
    else:
        for src in _find_store_paths():
            path, tmp_dir = copy_store_to_temp(src)
            try:
                items.extend(_iter_store(path, since_iso))
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    items.sort(key=lambda r: r.date_iso)
    yield from items


if __name__ == "__main__":
    reminders = list(iter_reminders())
    print(f"{len(reminders):,} reminders")
    for r in reminders[-3:]:
        status = f"done {r.completion_iso}" if r.completed else f"due {r.due_iso or '?'}"
        print(f"  [{status}] {r.title[:60]} ({r.list_name})")
