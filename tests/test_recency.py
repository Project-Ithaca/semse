import datetime as dt
import math
from types import SimpleNamespace

from api.search import (
    RECENCY_BOOST,
    RECENCY_HALFLIFE_DAYS,
    TEMPORAL_MIN_RECENT_CHUNKS,
    TEMPORAL_RECENT_DAYS,
    TEMPORAL_RECENT_FLOOR,
    _adaptive_recent_split,
    _parse_age_days,
    _recency_multiplier,
)

NOW = dt.datetime(2026, 8, 29, 12, 0, 0)


def _chunk(date_end: str) -> SimpleNamespace:
    return SimpleNamespace(date_end=date_end)


class TestRecencyMultiplier:
    def test_age_zero_gets_full_boost(self):
        assert _recency_multiplier(0.0) == 1.0 + RECENCY_BOOST

    def test_missing_date_no_boost(self):
        assert _recency_multiplier(None) == 1.0

    def test_future_date_clamped_to_full_boost(self):
        assert _recency_multiplier(-30.0) == 1.0 + RECENCY_BOOST

    def test_halflife_decay_value(self):
        expected = 1.0 + RECENCY_BOOST * math.exp(-1.0)
        assert _recency_multiplier(RECENCY_HALFLIFE_DAYS) == expected

    def test_monotonically_decreasing(self):
        ages = [0, 30, 90, 270, 1000, 3650]
        mults = [_recency_multiplier(a) for a in ages]
        assert mults == sorted(mults, reverse=True)

    def test_very_old_approaches_one(self):
        assert 1.0 < _recency_multiplier(3650) < 1.001

    def test_recent_wins_ties(self):
        old_score = 1.0 * _recency_multiplier(2000)
        new_score = 1.0 * _recency_multiplier(7)
        assert new_score > old_score

    def test_strong_old_match_still_beats_weak_recent(self):
        # Max uplift is 1.35x, so a 2x-stronger old match keeps winning.
        assert 2.0 * _recency_multiplier(2000) > 1.0 * _recency_multiplier(0)


class TestParseAgeDays:
    def test_parses_iso(self):
        age = _parse_age_days("2026-08-22T12:00:00", now=NOW)
        assert age == 7.0

    def test_missing_or_bad_dates(self):
        assert _parse_age_days(None, now=NOW) is None
        assert _parse_age_days("", now=NOW) is None
        assert _parse_age_days("not-a-date", now=NOW) is None

    def test_z_suffix_tolerated(self):
        assert _parse_age_days("2026-08-22T12:00:00Z", now=NOW) == 7.0


class TestAdaptiveRecentSplit:
    def test_empty(self):
        assert _adaptive_recent_split([], now=NOW) == ([], [])

    def test_plenty_in_90_days_uses_window(self):
        recent = [_chunk(f"2026-08-{d:02d}T10:00:00") for d in range(28, 8, -1)]
        older = [_chunk(f"2024-01-{d:02d}T10:00:00") for d in range(28, 8, -1)]
        r, o = _adaptive_recent_split(recent + older, now=NOW)
        assert len(r) == len(recent)
        assert len(o) == len(older)
        assert all(
            _parse_age_days(c.date_end, now=NOW) <= TEMPORAL_RECENT_DAYS for c in r
        )

    def test_thin_window_widens_to_fraction(self):
        # 4 chunks inside 90 days, 96 old ones → widen to 25% of 100 = 25.
        recent = [_chunk("2026-08-20T10:00:00")] * 4
        older = [_chunk("2023-05-01T10:00:00")] * 96
        r, o = _adaptive_recent_split(recent + older, now=NOW)
        assert len(r) == 25
        assert len(o) == 75

    def test_tiny_history_uses_floor(self):
        # 20 chunks, none recent → 25% = 5, floor lifts it to 10.
        chunks = [_chunk("2022-01-01T10:00:00")] * 20
        r, o = _adaptive_recent_split(chunks, now=NOW)
        assert len(r) == TEMPORAL_RECENT_FLOOR
        assert len(o) == 10

    def test_never_exceeds_total(self):
        chunks = [_chunk("2022-01-01T10:00:00")] * 5
        r, o = _adaptive_recent_split(chunks, now=NOW)
        assert len(r) == 5
        assert o == []

    def test_window_kept_when_already_wide_enough(self):
        n = TEMPORAL_MIN_RECENT_CHUNKS
        recent = [_chunk("2026-08-01T10:00:00")] * n
        older = [_chunk("2020-01-01T10:00:00")] * 200
        r, o = _adaptive_recent_split(recent + older, now=NOW)
        assert len(r) == n
        assert len(o) == 200
