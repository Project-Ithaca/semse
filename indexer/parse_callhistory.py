"""Parse macOS call history (phone + FaceTime relayed to Mac) into monthly chunks."""
from __future__ import annotations

import datetime
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from chunker import Chunk, _new_id
from contacts import ContactResolver

# Core Data epoch (2001-01-01 UTC), same offset as iMessage/Safari.
CORE_DATA_EPOCH_OFFSET = 978307200

CALL_DB_PATH = (
    Path.home() / "Library" / "Application Support" / "CallHistoryDB" / "CallHistory.storedata"
)

CALLS_PER_CHUNK = 20

# ZCALLTYPE values observed in the wild: 1 = phone, 8 = FaceTime video, 16 = FaceTime audio.
_CALL_KINDS = {1: "call", 8: "FaceTime video call", 16: "FaceTime audio call"}


@dataclass
class CallRecord:
    row_id: int
    contact_name: str      # resolved name, or "Unknown (•1234)" style fallback
    is_known: bool         # True when the address resolved to an AddressBook contact
    is_outgoing: bool
    is_answered: bool
    duration_seconds: float
    date_iso: str          # naive-UTC ISO
    kind: str              # "call" / "FaceTime video call" / "FaceTime audio call"


def call_date_to_iso(seconds: float | None) -> str:
    """Core Data epoch seconds → naive-UTC ISO string."""
    if seconds is None:
        return ""
    try:
        return (
            datetime.datetime.fromtimestamp(seconds + CORE_DATA_EPOCH_OFFSET, tz=datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat(timespec="seconds")
        )
    except (OverflowError, OSError, ValueError):
        return ""


def _copy_sqlite_to_temp(src: Path) -> tuple[str, str]:
    """Copy the Core Data store (+ -wal/-shm) to a temp dir; never touch the original."""
    tmp_dir = tempfile.mkdtemp(prefix="semse_calls_")
    try:
        dest = Path(tmp_dir) / src.name
        shutil.copy2(src, dest)
        for suffix in ("-wal", "-shm"):
            sidecar = src.parent / (src.name + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, Path(tmp_dir) / (src.name + suffix))
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return str(dest), tmp_dir


def _detect_columns(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("PRAGMA table_info(ZCALLRECORD)")
    return {row[1].upper() for row in cur.fetchall()}


def iter_calls(
    resolver: ContactResolver | None = None,
    since_iso: str | None = None,
) -> Iterator[CallRecord]:
    """Yield every call record, oldest first, with addresses resolved to names.

    Missing DB (no Full Disk Access, or calls never synced) skips with a notice.
    """
    if not CALL_DB_PATH.exists():
        print(f"  calls: CallHistory.storedata not found at {CALL_DB_PATH}, skipping")
        return
    if resolver is None:
        resolver = ContactResolver()
    resolver.load()

    path, tmp_dir = _copy_sqlite_to_temp(CALL_DB_PATH)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = _detect_columns(conn)
        required = {"Z_PK", "ZADDRESS", "ZDATE"}
        if not required <= cols:
            print(f"  calls: ZCALLRECORD missing columns {sorted(required - cols)}, skipping")
            return
        pick = lambda name, default: f"ZCALLRECORD.{name}" if name in cols else default
        query = f"""
            SELECT
              Z_PK,
              ZADDRESS,
              ZDATE,
              {pick('ZDURATION', '0')}   AS duration,
              {pick('ZORIGINATED', '0')} AS originated,
              {pick('ZANSWERED', '0')}   AS answered,
              {pick('ZCALLTYPE', '1')}   AS call_type,
              {pick('ZNAME', 'NULL')}    AS raw_name
            FROM ZCALLRECORD
            ORDER BY ZDATE ASC
        """
        for row in conn.execute(query):
            pk, address, date_val, duration, originated, answered, call_type, raw_name = row
            date_iso = call_date_to_iso(date_val)
            if not date_iso:
                continue
            if since_iso and date_iso <= since_iso:
                continue
            address_str = str(address).strip() if address else ""
            contact = resolver.resolve(address_str) if address_str else None
            if contact:
                name, known = contact.name, True
            elif address_str:
                name, known = resolver.display_for(address_str), False
            elif raw_name and str(raw_name).strip():
                # Some FaceTime rows carry no address but a caller-ID name.
                name, known = str(raw_name).strip(), False
            else:
                name, known = "Unknown", False
            yield CallRecord(
                row_id=int(pk),
                contact_name=name,
                is_known=known,
                is_outgoing=bool(originated),
                is_answered=bool(answered),
                duration_seconds=float(duration or 0),
                date_iso=date_iso,
                kind=_CALL_KINDS.get(int(call_type or 1), "call"),
            )
    finally:
        conn.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _format_call_line(c: CallRecord) -> str:
    when = c.date_iso[:16].replace("T", " ")
    minutes = max(1, round(c.duration_seconds / 60)) if c.duration_seconds > 0 else 0
    duration = f", {minutes} min" if (c.is_answered and minutes) else ""
    if c.is_outgoing:
        verb = f"Outgoing {c.kind} to" if c.is_answered else f"Unanswered {c.kind} to"
    else:
        verb = f"Incoming {c.kind} from" if c.is_answered else f"Missed {c.kind} from"
    return f"{when} · {verb} {c.contact_name}{duration}"


def chunk_calls(
    calls: Iterable[CallRecord],
    per_chunk: int = CALLS_PER_CHUNK,
) -> Iterator[Chunk]:
    """Group calls by calendar month, then split each month into chunks.

    Snippet-style chunks: empty `messages`, `subject` carries the month label,
    `contact_names` holds the unique resolved (known) names in the group.
    """
    by_month: dict[str, list[CallRecord]] = {}
    for c in calls:
        by_month.setdefault(c.date_iso[:7], []).append(c)

    for month in sorted(by_month):
        group = sorted(by_month[month], key=lambda c: c.date_iso)
        for start in range(0, len(group), per_chunk):
            slab = group[start : start + per_chunk]
            names = sorted({c.contact_name for c in slab if c.is_known})
            yield Chunk(
                chunk_id=_new_id(),
                source="calls",
                contact_names=names,
                date_start=slab[0].date_iso,
                date_end=slab[-1].date_iso,
                text=f"Call history for {month}:\n"
                + "\n".join(_format_call_line(c) for c in slab),
                row_ids=[c.row_id for c in slab],
                messages=[],
                subject=f"Calls · {month}",
            )


if __name__ == "__main__":
    records = list(iter_calls())
    chunks = list(chunk_calls(records))
    known = sum(1 for r in records if r.is_known)
    print(f"\n{len(records)} calls ({known} resolved to contacts) → {len(chunks)} chunks")
    for c in chunks[-3:]:
        print(f"\n--- {c.subject} ({len(c.row_ids)} calls, contacts={c.contact_names[:4]}) ---")
        print("\n".join(c.text.splitlines()[:6]))
