"""Temporal query parsing.

Detect time expressions in a query ("yesterday", "a few years ago", "since
November") and return an ISO date range. The API uses this to filter retrieval
to chunks whose `date_start` falls in the range.

Two important details:
  - We use the SYSTEM LOCAL timezone for relative phrases like "yesterday".
    iMessage timestamps are stored in UTC; the user means local-time yesterday.
  - We accept VAGUE phrases ("a while ago", "a few years ago") with deliberately
    wide windows so they don't over-filter.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from datetime import timezone

import dateparser

MAX_AGO_DAYS = 365 * 200  # sanity cap for "N units ago"


def _local_now() -> dt.datetime:
    """Wall-clock time in the user's local timezone."""
    return dt.datetime.now().astimezone()


def _to_utc(d: dt.datetime) -> dt.datetime:
    if d.tzinfo is None:
        d = d.astimezone()
    return d.astimezone(timezone.utc)


# Order matters: longer/more-specific patterns first.
TEMPORAL_PATTERNS = [
    # tom{1,2}or{1,2}ow covers the common misspellings (tommorow, tomorow…) —
    # typos in the query must not silently drop the date filter.
    (re.compile(r"\b(yesterday|today|tom{1,2}or{1,2}ow)\b", re.I), "day"),
    (re.compile(r"\bthis (morning|afternoon|evening|night)\b", re.I), "partial_day"),
    (re.compile(r"\b(?:a\s+)?few\s+(days?|weeks?|months?|years?)\s+ago\b", re.I), "few_ago"),
    (re.compile(r"\b(?:a\s+)?couple\s+(?:of\s+)?(days?|weeks?|months?|years?)\s+ago\b", re.I), "few_ago"),
    (re.compile(r"\ba\s+while\s+(back|ago)\b", re.I), "a_while"),
    (re.compile(r"\b(?:long|forever)\s+ago\b", re.I), "long_ago"),
    (re.compile(r"\brecently\b", re.I), "recently"),
    (re.compile(r"\b(last|past|previous)\s+(week|month|year|quarter)\b", re.I), "rel_window"),
    (re.compile(r"\b(this|next)\s+(week|month|year|quarter)\b", re.I), "this_window"),
    (re.compile(r"\b(\d+)\s*(days?|weeks?|months?|years?)\s+ago\b", re.I), "n_ago"),
    (re.compile(r"\bsince\s+([A-Za-z]+(?:\s+\d{1,4})?)", re.I), "since"),
    (re.compile(r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?\b", re.I), "month"),
    (re.compile(r"\bin\s+(\d{4})\b"), "year"),
]


@dataclass
class TemporalRange:
    after: dt.datetime | None       # UTC, inclusive
    before: dt.datetime | None      # UTC, inclusive
    matched_text: str
    label: str
    is_vague: bool = False          # True for "a while ago" etc — wide window

    def to_iso_range(self) -> tuple[str | None, str | None]:
        a = self.after.replace(tzinfo=None).isoformat() if self.after else None
        b = self.before.replace(tzinfo=None).isoformat() if self.before else None
        return a, b


# "jonah SAID a couple weeks ago he'd go to the store TOMORROW" — the future
# word belongs to reported speech, anchored to the message date, not to today.
_REPORTED_SPEECH_RE = re.compile(
    r"\b(say|said|says|saying|mention|mentioned|mentions|tell|tells|told|"
    r"texted|promised|planned|planning|was going to|were going to|would|gonna)\b",
    re.I,
)

_TOMORROWISH_RE = re.compile(r"tom{1,2}or{1,2}ow", re.I)


def _is_future_ref(kind: str, m: re.Match) -> bool:
    if kind == "day":
        return bool(_TOMORROWISH_RE.fullmatch(m.group(1)))
    if kind == "this_window":
        return m.group(1).lower() == "next"
    return False


def parse(query: str, now: dt.datetime | None = None) -> tuple[str, TemporalRange | None]:
    """Returns (cleaned_query, range_or_None)."""
    base = now.astimezone() if (now and now.tzinfo) else (now or _local_now())
    if base.tzinfo is None:
        base = base.astimezone()
    matches: list[tuple[re.Match, str]] = []
    for pattern, kind in TEMPORAL_PATTERNS:
        m = pattern.search(query)
        if m:
            matches.append((m, kind))
    if not matches:
        return query, None
    # A past anchor beats a future word: "said a couple weeks ago ... tomorrow"
    # filters on the couple-weeks window, and "tomorrow" stays in the query as
    # searchable content (the message likely contains the word itself).
    past = [(m, k) for m, k in matches if not _is_future_ref(k, m)]
    if past:
        chosen = past[0]
    elif _REPORTED_SPEECH_RE.search(query):
        # Only future refs, inside reported speech: someone's tomorrow-at-the-
        # time, not the user's — no date filter at all.
        return query, None
    else:
        chosen = matches[0]
    for m, kind in [chosen]:
        rng = _interpret(m, kind, base)
        if rng is None:
            continue
        cleaned = (query[: m.start()] + query[m.end():]).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        # Strip leftover possessive 's, leading punctuation, or trailing fragments
        # like "'s plans" → "plans" after the temporal phrase is removed.
        cleaned = re.sub(r"^['s]+\s*", "", cleaned)
        cleaned = re.sub(r"\s['s]+\s", " ", cleaned)
        cleaned = cleaned.strip(".,!?'\"; ")
        return cleaned or query, rng
    return query, None


def _day_window(d: dt.date, tz: dt.tzinfo) -> tuple[dt.datetime, dt.datetime]:
    start_local = dt.datetime.combine(d, dt.time.min, tzinfo=tz)
    end_local = dt.datetime.combine(d, dt.time.max, tzinfo=tz)
    return _to_utc(start_local), _to_utc(end_local)


def _interpret(m: re.Match, kind: str, base: dt.datetime) -> TemporalRange | None:
    text = m.group(0)
    tz = base.tzinfo

    if kind == "day":
        word = m.group(1).lower()
        if word not in ("yesterday", "today"):
            word = "tomorrow"
        offset = {"yesterday": -1, "today": 0, "tomorrow": 1}[word]
        d = (base + dt.timedelta(days=offset)).date()
        after, before = _day_window(d, tz)
        return TemporalRange(after=after, before=before, matched_text=text, label=word)

    if kind == "partial_day":
        d = base.date()
        windows = {
            "morning":   (dt.time(0, 0), dt.time(12, 0)),
            "afternoon": (dt.time(12, 0), dt.time(17, 0)),
            "evening":   (dt.time(17, 0), dt.time(21, 0)),
            "night":     (dt.time(21, 0), dt.time(23, 59, 59)),
        }
        start, end = windows[m.group(1).lower()]
        after = _to_utc(dt.datetime.combine(d, start, tzinfo=tz))
        before = _to_utc(dt.datetime.combine(d, end, tzinfo=tz))
        return TemporalRange(after=after, before=before, matched_text=text, label=f"this {m.group(1).lower()}")

    if kind == "rel_window":
        unit = m.group(2).lower().rstrip("s")
        days = {"week": 7, "month": 30, "year": 365, "quarter": 92}[unit]
        return TemporalRange(
            after=_to_utc(base - dt.timedelta(days=days)),
            before=_to_utc(base),
            matched_text=text, label=f"last {unit}",
        )

    if kind == "this_window":
        unit = m.group(2).lower().rstrip("s")
        if unit == "week":
            after = base - dt.timedelta(days=base.weekday())
            after = after.replace(hour=0, minute=0, second=0, microsecond=0)
            before = after + dt.timedelta(days=7)
        elif unit == "month":
            after = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            next_month = after.replace(day=28) + dt.timedelta(days=4)
            before = next_month.replace(day=1)
        elif unit == "year":
            after = base.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            before = after.replace(year=after.year + 1)
        else:
            return None
        return TemporalRange(after=_to_utc(after), before=_to_utc(before),
                             matched_text=text, label=f"this {unit}")

    if kind == "n_ago":
        n = int(m.group(1))
        unit = m.group(2).lower().rstrip("s")
        unit_days = {"day": 1, "week": 7, "month": 30, "year": 365}.get(unit, 1)
        # Clamp: unbounded N ("99999999 years ago") overflows timedelta.
        n = min(n, MAX_AGO_DAYS // unit_days)
        d = base - dt.timedelta(days=unit_days * n)
        # Window of width = unit, centered on the target day.
        return TemporalRange(
            after=_to_utc(d - dt.timedelta(days=max(1, unit_days // 2))),
            before=_to_utc(d + dt.timedelta(days=max(1, unit_days // 2))),
            matched_text=text, label=f"{n} {unit}{'s' if n != 1 else ''} ago",
        )

    if kind == "few_ago":
        unit = m.group(1).lower().rstrip("s")
        unit_days = {"day": 1, "week": 7, "month": 30, "year": 365}.get(unit, 1)
        # "a few X" is fuzzy — accept 1-5 X back so ~1.5 years is captured by
        # "a few years ago".
        return TemporalRange(
            after=_to_utc(base - dt.timedelta(days=int(unit_days * 5))),
            before=_to_utc(base - dt.timedelta(days=int(unit_days * 1))),
            matched_text=text, label=f"a few {unit}s ago", is_vague=True,
        )

    if kind == "a_while":
        # Anywhere from ~3 months to ~3 years back.
        return TemporalRange(
            after=_to_utc(base - dt.timedelta(days=365 * 3)),
            before=_to_utc(base - dt.timedelta(days=90)),
            matched_text=text, label="a while ago", is_vague=True,
        )

    if kind == "long_ago":
        # >1 year ago, no upper bound.
        return TemporalRange(
            after=None,
            before=_to_utc(base - dt.timedelta(days=365)),
            matched_text=text, label="long ago", is_vague=True,
        )

    if kind == "recently":
        return TemporalRange(
            after=_to_utc(base - dt.timedelta(days=30)),
            before=_to_utc(base + dt.timedelta(days=1)),
            matched_text=text, label="recently", is_vague=True,
        )

    if kind == "since":
        parsed = dateparser.parse(
            m.group(1),
            settings={"RELATIVE_BASE": base.replace(tzinfo=None), "PREFER_DATES_FROM": "past"},
        )
        if not parsed or parsed > base.replace(tzinfo=None):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        return TemporalRange(after=_to_utc(parsed), before=_to_utc(base), matched_text=text,
                             label=f"since {m.group(1)}")

    if kind == "month":
        month_name = m.group(1)
        year_str = m.group(2)
        year = int(year_str) if year_str else _nearest_past_year_for_month(month_name, base)
        month_num = MONTHS[month_name.lower()]
        after = dt.datetime(year, month_num, 1, tzinfo=tz)
        if month_num == 12:
            before = dt.datetime(year + 1, 1, 1, tzinfo=tz)
        else:
            before = dt.datetime(year, month_num + 1, 1, tzinfo=tz)
        return TemporalRange(after=_to_utc(after), before=_to_utc(before),
                             matched_text=text, label=f"{month_name} {year}")

    if kind == "year":
        # Clamp so datetime(y + 1, ...) can't exceed datetime.MAXYEAR.
        y = min(max(int(m.group(1)), dt.MINYEAR), dt.MAXYEAR - 1)
        return TemporalRange(
            after=_to_utc(dt.datetime(y, 1, 1, tzinfo=tz)),
            before=_to_utc(dt.datetime(y + 1, 1, 1, tzinfo=tz)),
            matched_text=text, label=str(y),
        )
    return None


MONTHS = {n: i + 1 for i, n in enumerate([
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
])}


def _nearest_past_year_for_month(month_name: str, base: dt.datetime) -> int:
    month_num = MONTHS[month_name.lower()]
    if month_num <= base.month:
        return base.year
    return base.year - 1


if __name__ == "__main__":
    cases = [
        "who did I meet yesterday",
        "what happened last week",
        "messages about foothill in november",
        "this morning's plans",
        "any news since october",
        "in 2024 what was I working on",
        "3 days ago",
        "a few years ago",
        "a while ago",
        "long ago",
        "recently",
        "claude shirt",
    ]
    now = _local_now()
    print(f"local now: {now}")
    print(f"utc now:   {now.astimezone(timezone.utc)}")
    print()
    for q in cases:
        cleaned, rng = parse(q, now=now)
        if rng:
            after = rng.after.date() if rng.after else None
            before = rng.before.date() if rng.before else None
            tag = " (vague)" if rng.is_vague else ""
            print(f"  {q!r:42s} → cleaned={cleaned!r:32s}  range=[{after}..{before}]{tag}")
        else:
            print(f"  {q!r:42s} → no temporal")
