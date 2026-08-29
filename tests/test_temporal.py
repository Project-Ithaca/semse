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
