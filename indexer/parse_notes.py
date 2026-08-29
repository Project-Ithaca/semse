"""Parse Apple Notes (NoteStore.sqlite) and chunk note bodies for indexing.

Note bodies are gzipped protobufs in ZICNOTEDATA.ZDATA. We extract plaintext
without a protobuf dependency by walking the wire format manually: the note
text lives at field path 2 (document) -> 3 (note) -> 2 (text string), with a
longest-valid-UTF-8-leaf fallback for blobs that don't match that shape.
"""
from __future__ import annotations

import datetime
import gzip
import shutil
import sqlite3
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from chunker import Chunk, _new_id

APPLE_EPOCH_OFFSET = 978307200  # seconds between Unix epoch and Core Data epoch (2001-01-01 UTC)

NOTE_STORE_PATH = (
    Path.home() / "Library" / "Group Containers" / "group.com.apple.notes" / "NoteStore.sqlite"
)

MAX_TOKENS_PER_CHUNK = 800
MAX_PROTO_DEPTH = 10


@dataclass
class NoteRecord:
    note_pk: int
    title: str
    folder: str | None
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


def decompress_note_data(blob: bytes) -> bytes:
    try:
        return gzip.decompress(blob)
    except (OSError, EOFError):
        return zlib.decompress(blob, 16 + zlib.MAX_WBITS)


# --- Minimal protobuf wire-format walker (no protobuf dependency) ---

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")
    raise ValueError("truncated varint")


def _proto_fields(buf: bytes) -> list[tuple[int, bytes]] | None:
    """Parse `buf` as one protobuf message; return its length-delimited fields.

    Returns [(field_number, payload), ...] covering only wire-type-2 fields
    (varint/fixed fields are skipped), or None if `buf` isn't a valid message.
    """
    pos = 0
    fields: list[tuple[int, bytes]] = []
    while pos < len(buf):
        try:
            key, pos = _read_varint(buf, pos)
        except ValueError:
            return None
        wire_type = key & 7
        field_num = key >> 3
        if field_num == 0 or field_num > 536870911:
            return None
        if wire_type == 0:  # varint
            try:
                _, pos = _read_varint(buf, pos)
            except ValueError:
                return None
        elif wire_type == 1:  # fixed64
            pos += 8
            if pos > len(buf):
                return None
        elif wire_type == 5:  # fixed32
            pos += 4
            if pos > len(buf):
                return None
        elif wire_type == 2:  # length-delimited
            try:
                length, pos = _read_varint(buf, pos)
            except ValueError:
                return None
            if pos + length > len(buf):
                return None
            fields.append((field_num, buf[pos : pos + length]))
            pos += length
        else:  # groups / invalid
            return None
    return fields


def _first_field(buf: bytes, field_num: int) -> bytes | None:
    fields = _proto_fields(buf)
    if fields is None:
        return None
    for num, payload in fields:
        if num == field_num:
            return payload
    return None


def extract_note_text(data: bytes) -> str | None:
    """Plaintext from a decompressed NoteStore proto blob.

    Exact known path first (2 -> 3 -> 2); falls back to the longest valid
    UTF-8 length-delimited leaf anywhere in the message tree.
    """
    document = _first_field(data, 2)
    if document is not None:
        note = _first_field(document, 3)
        if note is not None:
            text_bytes = _first_field(note, 2)
            if text_bytes is not None:
                try:
                    return text_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    pass

    best_leaf = ""
    best_any = ""

    def _recurse(buf: bytes, depth: int) -> None:
        nonlocal best_leaf, best_any
        if depth > MAX_PROTO_DEPTH:
            return
        fields = _proto_fields(buf)
        if fields is None:
            return
        for _, payload in fields:
            decoded: str | None
            try:
                decoded = payload.decode("utf-8")
            except UnicodeDecodeError:
                decoded = None
            if _proto_fields(payload):
                # Parses as a nested message — prefer its leaves, but keep the
                # raw bytes as a backup (real text can accidentally look like
                # a proto).
                if decoded and len(decoded) > len(best_any):
                    best_any = decoded
                _recurse(payload, depth + 1)
            elif decoded and len(decoded) > len(best_leaf):
                best_leaf = decoded

    _recurse(data, 0)
    return best_leaf or best_any or None


# --- DB access ---

def copy_note_store_to_temp() -> tuple[str, str]:
    """Copy NoteStore.sqlite (and -wal/-shm) to a fresh temp dir.

    Returns (db_path, tmp_dir); the caller MUST shutil.rmtree(tmp_dir) when done.
    Never touches the source DB.
    """
    if not NOTE_STORE_PATH.exists():
        raise FileNotFoundError(f"NoteStore.sqlite not found at {NOTE_STORE_PATH}")
    tmp_dir = tempfile.mkdtemp(prefix="semse_notes_")
    try:
        dest = Path(tmp_dir) / "NoteStore.sqlite"
        shutil.copy2(NOTE_STORE_PATH, dest)
        for suffix in ("-wal", "-shm"):
            src = NOTE_STORE_PATH.parent / (NOTE_STORE_PATH.name + suffix)
            if src.exists():
                shutil.copy2(src, Path(tmp_dir) / (dest.name + suffix))
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return str(dest), tmp_dir


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1].upper() for row in cur.fetchall()}


def _detect_columns(conn: sqlite3.Connection) -> dict[str, str | None]:
    """Apple's ZICCLOUDSYNCINGOBJECT columns drift across macOS versions; auto-detect.

    The table is a Core Data single-table inheritance dump, so note titles and
    folder titles live in differently-suffixed columns (ZTITLE1 vs ZTITLE2 on
    recent macOS; plain ZTITLE on old versions).
    """
    cols = _table_columns(conn, "ZICCLOUDSYNCINGOBJECT")
    pick = lambda *cands: next((c for c in cands if c in cols), None)
    return {
        "note_title": pick("ZTITLE1", "ZTITLE"),
        "folder_title": pick("ZTITLE2", "ZTITLE"),
        "mod_date": pick("ZMODIFICATIONDATE1", "ZMODIFICATIONDATE", "ZMODIFIEDDATE"),
        "password": pick("ZISPASSWORDPROTECTED"),
        "deleted": pick("ZMARKEDFORDELETION"),
        "folder_fk": pick("ZFOLDER"),
    }


def iter_notes(db_path: str | None = None) -> Iterator[NoteRecord]:
    """Yield every readable note, skipping password-protected and deleted ones."""
    tmp_dir: str | None = None
    if db_path:
        path = db_path
    else:
        path, tmp_dir = copy_note_store_to_temp()
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = _detect_columns(conn)
        title_col = f"o.{cols['note_title']}" if cols["note_title"] else "NULL"
        mod_col = f"o.{cols['mod_date']}" if cols["mod_date"] else "NULL"
        pw_col = f"o.{cols['password']}" if cols["password"] else "0"
        del_col = f"o.{cols['deleted']}" if cols["deleted"] else "0"
        folder_join = ""
        folder_col = "NULL"
        folder_del_col = "0"
        if cols["folder_fk"] and cols["folder_title"]:
            folder_join = f"LEFT JOIN ZICCLOUDSYNCINGOBJECT f ON o.{cols['folder_fk']} = f.Z_PK"
            folder_col = f"f.{cols['folder_title']}"
            if cols["deleted"]:
                folder_del_col = f"COALESCE(f.{cols['deleted']}, 0)"

        query = f"""
            SELECT
              o.Z_PK,
              {title_col}  AS title,
              {folder_col} AS folder,
              {mod_col}    AS mod_date,
              n.ZDATA      AS data
            FROM ZICNOTEDATA n
            JOIN ZICCLOUDSYNCINGOBJECT o ON n.ZNOTE = o.Z_PK
            {folder_join}
            WHERE n.ZDATA IS NOT NULL
              AND COALESCE({pw_col}, 0) = 0
              AND COALESCE({del_col}, 0) = 0
              AND {folder_del_col} = 0
            ORDER BY mod_date ASC
        """
        for note_pk, title, folder, mod_date, blob in conn.execute(query):
            if folder and folder.strip() == "Recently Deleted":
                continue
            try:
                data = decompress_note_data(blob)
            except (OSError, EOFError, zlib.error):
                continue
            text = extract_note_text(data)
            if not text or not text.strip():
                continue
            yield NoteRecord(
                note_pk=int(note_pk),
                title=str(title or "").strip(),
                folder=str(folder).strip() if folder else None,
                text=text.strip(),
                date_iso=core_data_to_iso(mod_date),
            )
    finally:
        conn.close()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Chunk builder ---

def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _split_text(text: str, max_tokens: int) -> list[str]:
    """Split on paragraph boundaries, accumulating up to ~max_tokens per part."""
    if _approx_token_count(text) <= max_tokens:
        return [text]
    max_chars = max_tokens * 4
    paragraphs = text.split("\n\n")
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        # A single paragraph beyond the budget gets hard-sliced.
        while len(para) > max_chars:
            if current:
                parts.append("\n\n".join(current))
                current = []
                current_len = 0
            parts.append(para[:max_chars])
            para = para[max_chars:]
        if current and current_len + len(para) > max_chars:
            parts.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2
    if current:
        tail = "\n\n".join(current)
        if tail.strip():
            parts.append(tail)
    return [p for p in parts if p.strip()]


def chunk_notes(
    records,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
) -> Iterator[Chunk]:
    """One chunk per note; oversized notes split into sequential parts.

    Snippet-style chunks: `messages` stays empty, `subject` carries the note
    title, `contact_names` stays empty (notes have no correspondents).
    """
    for rec in records:
        parts = _split_text(rec.text, max_tokens_per_chunk)
        total = len(parts)
        for i, part in enumerate(parts):
            subject = rec.title or None
            if subject and total > 1:
                subject = f"{subject} (part {i + 1}/{total})"
            body = part
            if rec.title and not part.startswith(rec.title):
                body = f"{rec.title}\n{part}"
            yield Chunk(
                chunk_id=_new_id(),
                source="notes",
                contact_names=[],
                date_start=rec.date_iso,
                date_end=rec.date_iso,
                text=body,
                row_ids=[rec.note_pk],
                messages=[],
                subject=subject,
            )


if __name__ == "__main__":
    records = list(iter_notes())
    chunks = list(chunk_notes(records))
    print(f"{len(records)} notes -> {len(chunks)} chunks")
    for rec in records[:3]:
        preview = " ".join(rec.text.split())[:100]
        print(f"[{rec.date_iso}] ({rec.folder or 'no folder'}) {rec.title!r}: {preview}")
