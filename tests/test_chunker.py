from chunker import chunk_imessages
from parse_imessage import IMessageRow


def _rows(n, chat_id=1, thread_guid=None):
    return [
        IMessageRow(
            row_id=i,
            text=f"msg {i}",
            date_iso=f"2025-01-01T00:00:{i:02d}",
            is_from_me=(i % 2 == 0),
            contact_id="+15551234567",
            chat_id=chat_id,
            display_name=None,
            chat_identifier="+15551234567",
            guid=f"g{i}",
            thread_originator_guid=thread_guid,
        )
        for i in range(n)
    ]


def test_rows_all_covered():
    chunks = list(chunk_imessages(_rows(35)))
    assert chunks
    covered = sorted(rid for c in chunks for rid in c.row_ids)
    assert covered == list(range(35))


def test_chunks_carry_contact_and_dates():
    chunks = list(chunk_imessages(_rows(10)))
    for c in chunks:
        assert c.source == "imessage"
        assert c.date_start <= c.date_end
        assert c.text


def test_separate_chats_never_mix():
    rows = _rows(10, chat_id=1) + [
        IMessageRow(
            row_id=100 + i,
            text=f"other {i}",
            date_iso=f"2025-01-02T00:00:{i:02d}",
            is_from_me=False,
            contact_id="+15559999999",
            chat_id=2,
            display_name=None,
            chat_identifier="+15559999999",
            guid=f"h{i}",
            thread_originator_guid=None,
        )
        for i in range(10)
    ]
    chunks = list(chunk_imessages(rows))
    for c in chunks:
        ids = set(c.row_ids)
        assert ids <= set(range(100)) or ids <= set(range(100, 200))
