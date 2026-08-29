from parse_browser_history import (
    BrowserVisit,
    chunk_browser_visits,
    clean_url,
    core_data_time_to_iso,
    webkit_time_to_iso,
)
from parse_callhistory import CallRecord, call_date_to_iso, chunk_calls, _format_call_line


def _visit(i, day="2026-08-14", url=None, browser="Arc", title=None):
    return BrowserVisit(
        row_id=i,
        title=title or f"Page {i}",
        url=url or f"example.com/page/{i}",
        domain="example.com",
        visit_iso=f"{day}T10:{i % 60:02d}:00",
        browser=browser,
    )


def _call(i, month="2026-08", name="Jerry Yan", known=True, outgoing=True,
          answered=True, duration=720.0, kind="call"):
    return CallRecord(
        row_id=i,
        contact_name=name,
        is_known=known,
        is_outgoing=outgoing,
        is_answered=answered,
        duration_seconds=duration,
        date_iso=f"{month}-{(i % 27) + 1:02d}T18:32:00",
        kind=kind,
    )


class TestEpochConversions:
    def test_core_data_epoch_zero_is_2001(self):
        assert core_data_time_to_iso(0) == "2001-01-01T00:00:00"

    def test_core_data_known_value(self):
        # 2026-08-27T00:00:00 UTC = 809481600 seconds after 2001-01-01
        assert core_data_time_to_iso(809481600.5) == "2026-08-27T00:00:00"

    def test_core_data_none_is_empty(self):
        assert core_data_time_to_iso(None) == ""

    def test_webkit_epoch_at_unix_zero(self):
        assert webkit_time_to_iso(11644473600 * 1_000_000) == "1970-01-01T00:00:00"

    def test_webkit_known_value(self):
        # 2026-08-27T00:00:00 UTC = 1787788800 unix
        us = (1787788800 + 11644473600) * 1_000_000
        assert webkit_time_to_iso(us) == "2026-08-27T00:00:00"

    def test_webkit_zero_is_empty(self):
        assert webkit_time_to_iso(0) == ""
        assert webkit_time_to_iso(None) == ""

    def test_no_utc_offset_suffix(self):
        assert "+" not in core_data_time_to_iso(809568000)
        assert "+" not in webkit_time_to_iso(13432512213922916)

    def test_call_date_matches_core_data(self):
        assert call_date_to_iso(809568000) == core_data_time_to_iso(809568000)


class TestCleanUrl:
    def test_strips_query_and_fragment(self):
        assert clean_url("https://github.com/a/b?tab=readme#top") == ("github.com/a/b", "github.com")

    def test_strips_trailing_slash_and_lowercases_host(self):
        assert clean_url("https://GitHub.com/") == ("github.com", "github.com")

    def test_skips_non_http_schemes(self):
        for u in ("data:text/html,hi", "file:///etc/hosts", "chrome://settings",
                  "about:blank", "javascript:void(0)", "ftp://x.com/f"):
            assert clean_url(u) is None

    def test_skips_localhost(self):
        assert clean_url("http://localhost:8000/docs") is None
        assert clean_url("http://127.0.0.1:3000") is None

    def test_skips_auth_noise_hosts(self):
        assert clean_url("https://accounts.google.com/signin/oauth?x=1") is None

    def test_skips_overlong_urls(self):
        assert clean_url("https://example.com/" + "a" * 600) is None

    def test_skips_empty(self):
        assert clean_url(None) is None
        assert clean_url("") is None


class TestBrowserChunks:
    def test_groups_by_day(self):
        visits = [_visit(i, day="2026-08-14") for i in range(3)] + [
            _visit(i + 100, day="2026-08-15") for i in range(2)
        ]
        chunks = list(chunk_browser_visits(visits))
        assert len(chunks) == 2
        assert chunks[0].subject == "Browsing · 2026-08-14"
        assert chunks[1].subject == "Browsing · 2026-08-15"

    def test_splits_oversized_days(self):
        chunks = list(chunk_browser_visits([_visit(i) for i in range(60)], per_chunk=25))
        assert [len(c.row_ids) for c in chunks] == [25, 25, 10]

    def test_chunk_shape(self):
        [chunk] = chunk_browser_visits([_visit(1, title="FAISS docs", url="github.com/faiss")])
        assert chunk.source == "browsing"
        assert chunk.contact_names == []
        assert chunk.messages == []
        assert chunk.date_start == chunk.date_end == "2026-08-14T10:01:00"
        assert "2026-08-14 · FAISS docs — github.com/faiss (Arc)" in chunk.text

    def test_all_rows_covered(self):
        visits = [_visit(i, day=f"2026-08-{10 + i % 3:02d}") for i in range(40)]
        chunks = list(chunk_browser_visits(visits))
        assert sorted(rid for c in chunks for rid in c.row_ids) == list(range(40))


class TestCallChunks:
    def test_groups_by_month(self):
        calls = [_call(i, month="2026-07") for i in range(3)] + [
            _call(i + 100, month="2026-08") for i in range(2)
        ]
        chunks = list(chunk_calls(calls))
        assert len(chunks) == 2
        assert chunks[0].subject == "Calls · 2026-07"
        assert chunks[1].subject == "Calls · 2026-08"

    def test_splits_oversized_months(self):
        chunks = list(chunk_calls([_call(i) for i in range(45)], per_chunk=20))
        assert [len(c.row_ids) for c in chunks] == [20, 20, 5]

    def test_contact_names_unique_and_known_only(self):
        calls = [
            _call(1, name="Jerry Yan"),
            _call(2, name="Jerry Yan"),
            _call(3, name="Ana Wu"),
            _call(4, name="Unknown (•4821)", known=False),
        ]
        [chunk] = chunk_calls(calls)
        assert chunk.contact_names == ["Ana Wu", "Jerry Yan"]

    def test_chunk_shape(self):
        [chunk] = chunk_calls([_call(1)])
        assert chunk.source == "calls"
        assert chunk.messages == []
        assert chunk.date_start == chunk.date_end == "2026-08-02T18:32:00"

    def test_line_outgoing_answered(self):
        line = _format_call_line(_call(13, name="Jerry Yan", duration=720.0))
        assert line == "2026-08-14 18:32 · Outgoing call to Jerry Yan, 12 min"

    def test_line_missed_incoming(self):
        line = _format_call_line(
            _call(13, name="Unknown (•4821)", known=False, outgoing=False,
                  answered=False, duration=0.0)
        )
        assert line == "2026-08-14 18:32 · Missed call from Unknown (•4821)"

    def test_line_facetime_kind(self):
        line = _format_call_line(_call(13, kind="FaceTime video call", duration=65.0))
        assert "Outgoing FaceTime video call to Jerry Yan, 1 min" in line

    def test_unanswered_has_no_duration(self):
        line = _format_call_line(_call(13, answered=False, duration=30.0))
        assert "min" not in line
        assert line.startswith("2026-08-14 18:32 · Unanswered call to")
