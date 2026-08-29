"""Parse WhatsApp desktop's ChatStorage.sqlite and chunk conversation windows."""
from __future__ import annotations

import base64
import binascii
import datetime
import glob
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from chunker import Chunk, ChunkMessage, _new_id

APPLE_EPOCH_OFFSET = 978307200  # seconds between Unix epoch and Core Data epoch (2001-01-01 UTC)

CHAT_STORAGE_GLOBS = [
    str(Path.home() / "Library" / "Group Containers" / "*whatsapp*" / "ChatStorage.sqlite"),
    str(Path.home() / "Library" / "Group Containers" / "*WhatsApp*" / "ChatStorage.sqlite"),
    str(Path.home() / "Library" / "Containers" / "*WhatsApp*" / "Data" / "Library" / "**" / "ChatStorage.sqlite"),
]

MESSAGES_PER_CHUNK = 20

# ZMESSAGETYPE values that carry no conversational content:
# 6 = group event (member add/remove, subject change), 10 = membership/jid
# notice, 43 = WhatsApp system announcement. Plain text is 0; 7 (link) and
# 8 (document caption) keep their user-visible text.
NOISE_MESSAGE_TYPES = {6, 10, 43}

# ZSESSIONTYPE: 0 = individual, 1 = group, 2 = broadcast/status.
BROADCAST_SESSION_TYPE = 2


@dataclass
class WhatsAppRow:
    row_id: int
    chat_pk: int
    chat_name: str
    is_group: bool
    sender: str          # display name: "Me", partner name, group member name, or "Them"
    is_from_me: bool
    text: str
    date_iso: str


def core_data_to_iso(ts: float | None) -> str:
    """Core Data seconds-since-2001 -> naive-UTC ISO (no offset suffix)."""
    if ts is None:
        return ""
    try:
        return (
            datetime.datetime.fromtimestamp(float(ts) + APPLE_EPOCH_OFFSET, tz=datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def find_chat_storage() -> Path | None:
    """Newest ChatStorage.sqlite across known WhatsApp install locations, or None."""
    candidates: set[str] = set()
    for pattern in CHAT_STORAGE_GLOBS:
        candidates.update(glob.glob(pattern, recursive=True))
    if not candidates:
        return None
    return Path(max(candidates, key=lambda p: os.path.getmtime(p)))


def copy_chat_storage_to_temp(source: Path) -> tuple[str, str]:
    """Copy ChatStorage.sqlite (and -wal/-shm) to a fresh temp dir.

    Returns (db_path, tmp_dir); the caller MUST shutil.rmtree(tmp_dir) when done.
    Never touches the source DB.
    """
    tmp_dir = tempfile.mkdtemp(prefix="semse_whatsapp_")
    try:
        dest = Path(tmp_dir) / "ChatStorage.sqlite"
        shutil.copy2(source, dest)
        for suffix in ("-wal", "-shm"):
            src = source.parent / (source.name + suffix)
            if src.exists():
                shutil.copy2(src, Path(tmp_dir) / (dest.name + suffix))
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return str(dest), tmp_dir


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1].upper() for row in cur.fetchall()}


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def _load_sessions(conn: sqlite3.Connection) -> dict[int, tuple[str, bool]]:
    """Chat session Z_PK -> (display name, is_group), skipping status/broadcast."""
    cols = _table_columns(conn, "ZWACHATSESSION")
    name_col = "ZPARTNERNAME" if "ZPARTNERNAME" in cols else None
    jid_col = "ZCONTACTJID" if "ZCONTACTJID" in cols else None
    type_col = "ZSESSIONTYPE" if "ZSESSIONTYPE" in cols else None
    group_col = "ZGROUPINFO" if "ZGROUPINFO" in cols else None

    select = ["Z_PK"]
    select.append(name_col or "NULL")
    select.append(jid_col or "NULL")
    select.append(type_col or "NULL")
    select.append(group_col or "NULL")
    sessions: dict[int, tuple[str, bool]] = {}
    for pk, name, jid, stype, group_info in conn.execute(
        f"SELECT {', '.join(select)} FROM ZWACHATSESSION"
    ):
        if jid and str(jid).startswith("status@"):
            continue
        if stype is not None and int(stype) == BROADCAST_SESSION_TYPE:
            continue
        is_group = bool(group_info) or (jid is not None and str(jid).endswith("@g.us"))
        display = _decode_wrapped_name(str(name).strip()) if name else ""
        sessions[int(pk)] = (display or (str(jid) if jid else "Unknown"), is_group)
    return sessions


def _decode_wrapped_name(value: str) -> str:
    """Unwrap WhatsApp's "+<base64 protobuf>" name encoding.

    Newer WhatsApp versions store `+CgA...=` blobs in ZFIRSTNAME: base64 of a
    tiny proto whose field 1 (length-delimited) is the display name and field 2
    a timestamp varint. Returns the inner name, or "" when the blob has none.
    Plain human names pass through unchanged.
    """
    if not value.startswith("+Cg"):
        return value
    try:
        raw = base64.b64decode(value[1:], validate=True)
    except (binascii.Error, ValueError):
        return value
    # Expect: 0x0A (field 1, wire type 2), varint length, UTF-8 name bytes.
    if len(raw) < 2 or raw[0] != 0x0A:
        return ""
    length = 0
    shift = 0
    pos = 1
    while pos < len(raw):
        b = raw[pos]
        pos += 1
        length |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    if pos + length > len(raw):
        return ""
    try:
        return raw[pos : pos + length].decode("utf-8").strip()
    except UnicodeDecodeError:
        return ""


def _load_group_members(conn: sqlite3.Connection) -> dict[int, tuple[str, str | None]]:
    """ZWAGROUPMEMBER Z_PK -> (best display name, member JID).

    Names are sparse here; the member JID lets callers fall back to the
    push-name table (ZFROMJID on group messages is the group's JID, not the
    sender's, so this is the only route to the actual sender).
    """
    if not _has_table(conn, "ZWAGROUPMEMBER"):
        return {}
    cols = _table_columns(conn, "ZWAGROUPMEMBER")
    name_col = "ZCONTACTNAME" if "ZCONTACTNAME" in cols else "NULL"
    first_col = "ZFIRSTNAME" if "ZFIRSTNAME" in cols else "NULL"
    jid_col = "ZMEMBERJID" if "ZMEMBERJID" in cols else "NULL"
    members: dict[int, tuple[str, str | None]] = {}
    for pk, name, first, jid in conn.execute(
        f"SELECT Z_PK, {name_col}, {first_col}, {jid_col} FROM ZWAGROUPMEMBER"
    ):
        display = _decode_wrapped_name((name or first or "").strip())
        members[int(pk)] = (display, str(jid) if jid else None)
    return members


def _load_push_names(conn: sqlite3.Connection) -> dict[str, str]:
    """Sender JID -> self-reported push name (fallback for unnamed group members)."""
    if not _has_table(conn, "ZWAPROFILEPUSHNAME"):
        return {}
    cols = _table_columns(conn, "ZWAPROFILEPUSHNAME")
    if "ZJID" not in cols or "ZPUSHNAME" not in cols:
        return {}
    names: dict[str, str] = {}
    for jid, push in conn.execute("SELECT ZJID, ZPUSHNAME FROM ZWAPROFILEPUSHNAME"):
        display = _decode_wrapped_name(str(push).strip()) if push else ""
        if jid and display:
            names[str(jid)] = display
    return names


def iter_whatsapp_messages(db_path: str | None = None) -> Iterator[WhatsAppRow]:
    """Yield WhatsApp text messages ordered by chat then date ASC.

    Yields nothing (with a printed notice) when WhatsApp isn't installed.
    """
    tmp_dir: str | None = None
    if db_path:
        path = db_path
    else:
        source = find_chat_storage()
        if source is None:
            print("  WhatsApp ChatStorage.sqlite not found — skipping whatsapp source")
            return
        path, tmp_dir = copy_chat_storage_to_temp(source)
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        if not _has_table(conn, "ZWAMESSAGE"):
            print("  ChatStorage.sqlite has no ZWAMESSAGE table — skipping whatsapp source")
            return
        msg_cols = _table_columns(conn, "ZWAMESSAGE")
        required = {"ZTEXT", "ZMESSAGEDATE", "ZISFROMME", "ZCHATSESSION"}
        if not required.issubset(msg_cols):
            print("  ChatStorage.sqlite ZWAMESSAGE missing expected columns — skipping")
            return

        sessions = _load_sessions(conn)
        members = _load_group_members(conn)
        push_names = _load_push_names(conn)

        type_col = "ZMESSAGETYPE" if "ZMESSAGETYPE" in msg_cols else None
        member_col = "ZGROUPMEMBER" if "ZGROUPMEMBER" in msg_cols else None
        from_jid_col = "ZFROMJID" if "ZFROMJID" in msg_cols else None

        query = f"""
            SELECT
              Z_PK,
              ZCHATSESSION,
              ZTEXT,
              ZMESSAGEDATE,
              ZISFROMME,
              {type_col or 'NULL'} AS msg_type,
              {member_col or 'NULL'} AS member_pk,
              {from_jid_col or 'NULL'} AS from_jid
            FROM ZWAMESSAGE
            WHERE ZTEXT IS NOT NULL AND length(trim(ZTEXT)) > 0
            ORDER BY ZCHATSESSION, ZMESSAGEDATE ASC
        """
        for row in conn.execute(query):
            row_id, chat_pk, text, date_val, is_from_me, msg_type, member_pk, from_jid = row
            if chat_pk is None or int(chat_pk) not in sessions:
                continue
            if msg_type is not None and int(msg_type) in NOISE_MESSAGE_TYPES:
                continue
            chat_name, is_group = sessions[int(chat_pk)]
            if bool(is_from_me):
                sender = "Me"
            elif is_group:
                member_name, member_jid = (
                    members.get(int(member_pk), ("", None)) if member_pk is not None else ("", None)
                )
                sender = (
                    member_name
                    or (push_names.get(member_jid) if member_jid else None)
                    or "Them"
                )
            else:
                sender = chat_name
            yield WhatsAppRow(
                row_id=int(row_id),
                chat_pk=int(chat_pk),
                chat_name=chat_name,
                is_group=is_group,
                sender=sender,
                is_from_me=bool(is_from_me),
                text=str(text).strip(),
                date_iso=core_data_to_iso(date_val),
            )
    finally:
        conn.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Chunk builder ---

def _build_whatsapp_chunk(rows: list[WhatsAppRow]) -> Chunk:
    structured = [
        ChunkMessage(
            sender=r.sender,
            is_from_me=r.is_from_me,
            text=r.text,
            date_iso=r.date_iso,
            contact_key=None,
            known=True,
        )
        for r in rows
    ]
    first = rows[0]
    return Chunk(
        chunk_id=_new_id(),
        source="whatsapp",
        contact_names=[first.chat_name],
        date_start=rows[0].date_iso,
        date_end=rows[-1].date_iso,
        text="\n".join(f"{m.sender}: {m.text}" for m in structured),
        row_ids=[r.row_id for r in rows],
        messages=structured,
        chat_title=first.chat_name if first.is_group else None,
    )


def chunk_whatsapp_messages(
    rows,
    messages_per_chunk: int = MESSAGES_PER_CHUNK,
) -> Iterator[Chunk]:
    """Fixed-size conversation windows per chat, in date order.

    Rows must arrive sorted by chat then date (iter_whatsapp_messages does).
    """
    current_chat: int | None = None
    buffer: list[WhatsAppRow] = []

    def flush(buf: list[WhatsAppRow]) -> Iterator[Chunk]:
        for start in range(0, len(buf), messages_per_chunk):
            window = buf[start : start + messages_per_chunk]
            if window:
                yield _build_whatsapp_chunk(window)

    for row in rows:
        if current_chat is None:
            current_chat = row.chat_pk
        if row.chat_pk != current_chat:
            yield from flush(buffer)
            buffer = []
            current_chat = row.chat_pk
        buffer.append(row)
    yield from flush(buffer)


if __name__ == "__main__":
    rows = list(iter_whatsapp_messages())
    chunks = list(chunk_whatsapp_messages(rows))
    n_chats = len({r.chat_pk for r in rows})
    print(f"{len(rows)} messages across {n_chats} chats -> {len(chunks)} chunks")
    for chunk in chunks[:3]:
        preview = " ".join(chunk.text.split())[:100]
        label = chunk.chat_title or chunk.contact_names[0]
        print(f"[{chunk.date_start} .. {chunk.date_end}] {label!r} ({len(chunk.row_ids)} msgs): {preview}")
