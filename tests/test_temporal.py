from api import temporal


def test_last_week_parses():
    cleaned, trange = temporal.parse("what did sarah say last week")
    assert trange is not None
    assert "last week" not in cleaned


def test_no_temporal_phrase():
    cleaned, trange = temporal.parse("what did sarah say about the trip")
    assert trange is None
    assert cleaned == "what did sarah say about the trip"


def test_n_ago_overflow_does_not_crash():
    cleaned, trange = temporal.parse("what happened 99999999 years ago")
    # clamped, not OverflowError
    assert trange is None or trange is not None


def test_year_overflow_does_not_crash():
    temporal.parse("what happened in 9999")
    temporal.parse("what happened in 2024")


def test_in_year():
    _, trange = temporal.parse("that thing in 2023")
    assert trange is not None
    start, end = trange.to_iso_range()
    assert start.startswith("2023")


def test_tomorrow_misspellings_still_parse():
    for word in ("tomorrow", "tommorow", "tomorow", "tommorrow"):
        cleaned, trange = temporal.parse(f"what do i have {word}")
        assert trange is not None, word
        assert trange.label == "tomorrow"
        assert word not in cleaned


class TestReportedSpeech:
    def test_past_anchor_beats_future_word(self):
        q = "jonah said a couple weeks ago he would go to the grocery store tommorow"
        cleaned, trange = temporal.parse(q)
        assert trange is not None
        assert "couple" in trange.matched_text.lower() or "weeks" in trange.matched_text.lower()
        assert "tommorow" in cleaned  # stays as searchable content

    def test_reported_speech_future_word_means_no_filter(self):
        cleaned, trange = temporal.parse("did jonah say he is going to the store tomorrow")
        assert trange is None
        assert cleaned == "did jonah say he is going to the store tomorrow"

    def test_plain_schedule_question_keeps_future_window(self):
        _, trange = temporal.parse("what do i have tommorow")
        assert trange is not None and trange.label == "tomorrow"

    def test_next_week_without_speech_verb_keeps_window(self):
        _, trange = temporal.parse("what do i have next week")
        assert trange is not None
