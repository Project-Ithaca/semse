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


from chunker import (
    CalendarEvent,
    ReminderItem,
    chunk_calendar_events,
    chunk_reminders,
)


def _events(n, month="2026-08", calendar="Work"):
    return [
        CalendarEvent(
            row_id=i,
            title=f"event {i}",
            start_iso=f"{month}-{(i % 28) + 1:02d}T10:00:00",
            end_iso=f"{month}-{(i % 28) + 1:02d}T11:00:00",
            all_day=False,
            calendar_name=calendar,
            location="123 Main St" if i % 2 == 0 else None,
            notes=None,
        )
        for i in range(n)
    ]


def test_calendar_events_grouped_by_month():
    events = _events(3, month="2026-07") + _events(3, month="2026-08")
    chunks = list(chunk_calendar_events(events))
    assert len(chunks) == 2
    assert sorted(c.subject for c in chunks) == ["2026-07", "2026-08"]
    for c in chunks:
        assert c.source == "calendar"
        assert c.contact_names == []
        assert c.messages == []
        assert c.date_start <= c.date_end
        assert c.date_start[:7] == c.subject


def test_calendar_month_splits_at_cap():
    chunks = list(chunk_calendar_events(_events(35), per_chunk=15))
    assert len(chunks) == 3
    covered = sorted(rid for c in chunks for rid in c.row_ids)
    assert covered == list(range(35))


def test_calendar_event_rendering():
    ev = CalendarEvent(
        row_id=1,
        title="Dentist appointment",
        start_iso="2026-08-14T14:00:00",
        end_iso="2026-08-14T15:00:00",
        all_day=False,
        calendar_name="Health",
        location="123 Main St",
        notes="bring insurance card",
    )
    (chunk,) = chunk_calendar_events([ev])
    assert "2026-08-14 14:00 — Dentist appointment (Health) @ 123 Main St" in chunk.text
    assert "bring insurance card" in chunk.text


def test_calendar_all_day_renders_date_only():
    ev = CalendarEvent(
        row_id=1,
        title="Independence Day",
        start_iso="2026-07-04T00:00:00",
        end_iso="2026-07-04T23:59:59",
        all_day=True,
        calendar_name=None,
        location=None,
        notes=None,
    )
    (chunk,) = chunk_calendar_events([ev])
    assert "2026-07-04 — Independence Day" in chunk.text
    assert "00:00" not in chunk.text


def _reminders(n, list_name="Robotics", completed=False):
    return [
        ReminderItem(
            row_id=i,
            title=f"task {i}",
            notes=None,
            completed=completed,
            due_iso=f"2026-07-{(i % 28) + 1:02d}T09:00:00",
            completion_iso=f"2026-07-{(i % 28) + 1:02d}T18:00:00" if completed else "",
            date_iso=f"2026-07-{(i % 28) + 1:02d}T09:00:00",
            list_name=list_name,
        )
        for i in range(n)
    ]


def test_reminders_grouped_by_list_and_status():
    items = (
        _reminders(3, "Robotics", completed=False)
        + _reminders(3, "Robotics", completed=True)
        + _reminders(3, "Groceries", completed=False)
    )
    chunks = list(chunk_reminders(items))
    assert len(chunks) == 3
    assert sorted(c.subject for c in chunks) == ["Groceries", "Robotics", "Robotics"]
    for c in chunks:
        assert c.source == "reminders"
        assert c.contact_names == []
        assert c.messages == []
        assert c.date_start <= c.date_end


def test_reminders_split_at_cap():
    chunks = list(chunk_reminders(_reminders(35), per_chunk=15))
    assert len(chunks) == 3
    covered = sorted(rid for c in chunks for rid in c.row_ids)
    assert covered == list(range(35))


def test_reminder_rendering():
    done = ReminderItem(
        row_id=1,
        title="Buy filament",
        notes=None,
        completed=True,
        due_iso="2026-06-30T09:00:00",
        completion_iso="2026-07-01T10:00:00",
        date_iso="2026-07-01T10:00:00",
        list_name="Robotics",
    )
    pending = ReminderItem(
        row_id=2,
        title="Charge batteries",
        notes="both packs",
        completed=False,
        due_iso="2026-07-05T09:00:00",
        completion_iso="",
        date_iso="2026-07-05T09:00:00",
        list_name="Robotics",
    )
    chunks = list(chunk_reminders([done, pending]))
    text = "\n".join(c.text for c in chunks)
    assert "[done 2026-07-01] Buy filament (Robotics list)" in text
    assert "[due 2026-07-05] Charge batteries (Robotics list) — both packs" in text
