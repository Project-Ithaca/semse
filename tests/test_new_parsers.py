"""Unit tests for parse_notes and parse_whatsapp — synthetic data only."""
import gzip

from parse_notes import (
    NoteRecord,
    chunk_notes,
    core_data_to_iso,
    decompress_note_data,
    extract_note_text,
    _proto_fields,
)
from parse_whatsapp import (
    WhatsAppRow,
    chunk_whatsapp_messages,
    _decode_wrapped_name,
)
from parse_whatsapp import core_data_to_iso as wa_core_data_to_iso


# --- protobuf helpers for hand-building blobs ---

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld_field(num: int, payload: bytes) -> bytes:
    return _varint((num << 3) | 2) + _varint(len(payload)) + payload


def _varint_field(num: int, value: int) -> bytes:
    return _varint(num << 3) + _varint(value)


def _note_blob(text: str) -> bytes:
    """Outer message: field 2 (document) -> field 3 (note) -> field 2 (text)."""
    note = _varint_field(1, 42) + _ld_field(2, text.encode("utf-8"))
    document = _varint_field(1, 1) + _ld_field(3, note)
    return _ld_field(2, document)


# --- Notes: proto walker ---

def test_extract_note_text_exact_path():
    text = "Groceries\nmilk, eggs, bread"
    assert extract_note_text(_note_blob(text)) == text


def test_extract_note_text_fallback_longest_leaf():
    # No 2->3->2 path; the longest UTF-8 leaf should win.
    blob = _ld_field(5, _ld_field(1, b"short") + _ld_field(7, b"the much longer note body here"))
    assert extract_note_text(blob) == "the much longer note body here"


def test_extract_note_text_garbage_returns_none():
    assert extract_note_text(b"\xff\xff\xff\xff") is None


def test_proto_fields_rejects_truncated():
    valid = _ld_field(2, b"hello")
    assert _proto_fields(valid) == [(2, b"hello")]
    assert _proto_fields(valid[:-2]) is None


def test_decompress_note_data_gzip():
    blob = _note_blob("compressed body")
    assert decompress_note_data(gzip.compress(blob)) == blob


def test_core_data_to_iso_naive_utc():
    assert core_data_to_iso(0) == "2001-01-01T00:00:00"
    assert "+" not in core_data_to_iso(700000000)
    assert core_data_to_iso(None) == ""
    assert wa_core_data_to_iso(0) == "2001-01-01T00:00:00"


# --- Notes: chunk builder ---

def _note(pk=1, title="My Note", text="hello world", date="2025-06-01T12:00:00"):
    return NoteRecord(note_pk=pk, title=title, folder="Notes", text=text, date_iso=date)


def test_chunk_notes_single_chunk():
    chunks = list(chunk_notes([_note()]))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.source == "notes"
    assert c.subject == "My Note"
    assert c.contact_names == []
    assert c.messages == []
    assert c.row_ids == [1]
    assert c.date_start == c.date_end == "2025-06-01T12:00:00"
    assert "My Note" in c.text and "hello world" in c.text


def test_chunk_notes_splits_oversized():
    paragraphs = "\n\n".join(f"paragraph {i} " + "word " * 80 for i in range(20))
    chunks = list(chunk_notes([_note(text=paragraphs)], max_tokens_per_chunk=800))
    assert len(chunks) > 1
    assert all(c.source == "notes" for c in chunks)
    assert all(c.row_ids == [1] for c in chunks)
    assert chunks[0].subject.endswith(f"(part 1/{len(chunks)})")
    # Every part stays within a loose bound of the token budget.
    assert all(len(c.text) < 800 * 4 + 500 for c in chunks)
    # No content lost: every paragraph label appears somewhere.
    joined = "".join(c.text for c in chunks)
    assert all(f"paragraph {i} " in joined for i in range(20))


def test_chunk_notes_untitled():
    chunks = list(chunk_notes([_note(title="", text="body only")]))
    assert chunks[0].subject is None
    assert chunks[0].text == "body only"


# --- WhatsApp: name unwrapping ---

def test_decode_wrapped_name_extracts_inner():
    assert _decode_wrapped_name("+Cg9EciBTYW5qYXkgR3VwdGEQpNuIzQY=") == "Dr Sanjay Gupta"


def test_decode_wrapped_name_empty_inner():
    assert _decode_wrapped_name("+CgAQvP7Y0gY=") == ""


def test_decode_wrapped_name_passthrough():
    assert _decode_wrapped_name("Alice") == "Alice"
    assert _decode_wrapped_name("+1 (555) 123-4567") == "+1 (555) 123-4567"


# --- WhatsApp: chunk builder ---

def _wa_row(i, chat_pk=1, chat_name="Alice", is_group=False, sender=None):
    from_me = i % 2 == 0
    return WhatsAppRow(
        row_id=i,
        chat_pk=chat_pk,
        chat_name=chat_name,
        is_group=is_group,
        sender=sender or ("Me" if from_me else chat_name),
        is_from_me=from_me,
        text=f"message {i}",
        date_iso=f"2025-03-01T10:{i % 60:02d}:00",
    )


def test_chunk_whatsapp_windows_and_chat_boundaries():
    rows = [_wa_row(i, chat_pk=1) for i in range(25)] + [
        _wa_row(100 + i, chat_pk=2, chat_name="Bob") for i in range(3)
    ]
    chunks = list(chunk_whatsapp_messages(rows, messages_per_chunk=20))
    assert [len(c.row_ids) for c in chunks] == [20, 5, 3]
    assert chunks[0].contact_names == ["Alice"]
    assert chunks[2].contact_names == ["Bob"]
    # Chats never mix within a chunk.
    assert chunks[1].row_ids == list(range(20, 25))


def test_chunk_whatsapp_chunk_fields():
    rows = [_wa_row(i) for i in range(4)]
    c = list(chunk_whatsapp_messages(rows))[0]
    assert c.source == "whatsapp"
    assert c.chat_title is None  # 1:1 chat
    assert c.date_start == rows[0].date_iso
    assert c.date_end == rows[-1].date_iso
    assert c.text.splitlines()[0] == "Me: message 0"
    assert c.text.splitlines()[1] == "Alice: message 1"
    assert len(c.messages) == 4
    m = c.messages[1]
    assert (m.sender, m.is_from_me, m.contact_key, m.known) == ("Alice", False, None, True)
    assert m.text == "message 1"
    assert m.date_iso == rows[1].date_iso


def test_chunk_whatsapp_group_sets_chat_title():
    rows = [
        _wa_row(i, chat_name="Family Group", is_group=True, sender="Cousin")
        for i in range(1, 4, 2)
    ]
    c = list(chunk_whatsapp_messages(rows))[0]
    assert c.chat_title == "Family Group"
    assert c.contact_names == ["Family Group"]


def test_chunk_whatsapp_empty():
    assert list(chunk_whatsapp_messages([])) == []
