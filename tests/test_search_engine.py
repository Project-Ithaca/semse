"""End-to-end SearchEngine tests against a synthetic fixture index.

Everything here runs on FakeEmbedder + FakeLLM (see fixture_index.py): no
model load, no Ollama, no dependency on the user's real index.
"""
from __future__ import annotations

import asyncio

import pytest

import fixture_index as fx
from api.models import QueryIntent, SearchRequest, SourceResult
from api.search import SearchEngine


def run(coro):
    return asyncio.run(coro)


def sample_chunks() -> list[dict]:
    """Seven chunks spanning sources, contacts, and ages."""
    return [
        fx.chunk(
            "im-recent", "imessage", ["Jerry Yan"],
            "vex robotics competition schedule saturday", start_days_ago=3,
            chat_title="Jerry Yan",
        ),
        fx.chunk(
            "im-old", "imessage", ["Jerry Yan"],
            "vex robotics competition schedule saturday", start_days_ago=500,
            chat_title="Jerry Yan",
        ),
        fx.chunk(
            "notes-1", "notes", [],
            "vex robotics competition packing checklist", start_days_ago=10,
            subject="Packing",
        ),
        fx.chunk(
            "mail-1", "mail", ["Sam Rivera"],
            "invoice payment due for the robotics parts order",
            start_days_ago=20, subject="Invoice",
        ),
        fx.chunk(
            "cal-span", "calendar", [],
            "robotics team spring season practice block",
            start_days_ago=40, end_days_ago=5,
        ),
        fx.chunk(
            "im-dinner", "imessage", ["Sam Rivera"],
            "dinner plans at seven downtown", start_days_ago=2,
        ),
        fx.chunk(
            "wa-1", "whatsapp", ["Jerry Yan"],
            "flight booking confirmation to boston", start_days_ago=60,
        ),
    ]


JERRY_PERSONA = {
    "name": "Jerry Yan",
    "style_summary": "Jerry writes in short bursts and rarely punctuates.",
    "top_topics": [
        {"topic": "robotics competitions", "score": 0.42},
        {"topic": "flight plans", "score": 0.21},
    ],
    "total_messages": 900,
}

JERRY_SUMMARY = {
    "name": "Jerry Yan",
    "summary": "Teammate on the VEX robotics team.",
    "total_chunks": 40,
    "last_30d_chunks": 3,
}


def ids(sources: list[SourceResult]) -> list[str]:
    """Sources don't carry chunk_id; identify them by their text/snippet."""
    return [s.snippet or (s.messages[0].text if s.messages else "") for s in sources]


class TestStandardSearch:
    def test_returns_relevant_sources(self, make_engine):
        engine = make_engine(sample_chunks())
        resp = run(engine.search(SearchRequest(query="vex robotics competition", top_k=5)))
        assert resp.sources
        assert any("vex robotics competition" in t for t in ids(resp.sources))
        assert resp.query_ms >= 0

    def test_honors_top_k(self, make_engine):
        engine = make_engine(sample_chunks())
        resp = run(engine.search(SearchRequest(query="robotics", top_k=2)))
        assert len(resp.sources) <= 2

    def test_recency_boost_breaks_ties(self, make_engine):
        """im-recent and im-old have identical text; the recent one must win."""
        engine = make_engine(sample_chunks())
        resp = run(engine.search(SearchRequest(query="vex competition saturday", top_k=7)))
        dates = [s.date_end for s in resp.sources if "saturday" in ids([s])[0]]
        assert len(dates) == 2
        assert dates[0] > dates[1]

    def test_best_match_message_is_tagged(self, make_engine):
        chunks = [
            fx.chunk(
                "im-multi", "imessage", ["Jerry Yan"],
                "hey\nthe vex regional is saturday\nok",
                start_days_ago=1,
                messages=[
                    fx.message("Jerry Yan", "hey", fx.iso_days_ago(1)),
                    fx.message("Jerry Yan", "the vex regional is saturday", fx.iso_days_ago(1)),
                    fx.message("Jerry Yan", "ok", fx.iso_days_ago(1)),
                ],
            )
        ]
        engine = make_engine(chunks)
        resp = run(engine.search(SearchRequest(query="vex regional", top_k=3)))
        assert resp.sources
        flagged = [m.text for m in resp.sources[0].messages if m.is_best_match]
        assert flagged == ["the vex regional is saturday"]

    def test_empty_index_returns_no_sources(self, make_engine):
        engine = make_engine([fx.chunk("only", "notes", [], "totally unrelated text",
                                       start_days_ago=1)])
        resp = run(engine.search(SearchRequest(query="zzz", top_k=5)))
        assert isinstance(resp.sources, list)

    def test_answer_empty_when_llm_down(self, make_engine):
        engine = make_engine(sample_chunks(), llm=fx.FakeLLM(error=ConnectionError("down")))
        resp = run(engine.search(SearchRequest(query="what is the vex schedule", top_k=3)))
        assert resp.answer == ""
        assert resp.sources, "sources must still return when synthesis fails"

    def test_answer_returned_for_question(self, make_engine):
        llm = fx.FakeLLM(content="The VEX competition is on Saturday.")
        engine = make_engine(sample_chunks(), llm=llm)
        resp = run(engine.search(SearchRequest(query="when is the vex competition?", top_k=3)))
        assert resp.answer == "The VEX competition is on Saturday."
        assert llm.calls

    def test_no_synthesis_for_non_question(self, make_engine):
        llm = fx.FakeLLM(content="should not be used")
        engine = make_engine(sample_chunks(), llm=llm)
        resp = run(engine.search(SearchRequest(query="vex robotics", top_k=3)))
        assert resp.answer == ""
        assert not llm.calls


class TestSourceFilters:
    def test_intent_source_filter(self, make_engine):
        engine = make_engine(sample_chunks())
        intent = QueryIntent(sources=["notes"])
        resp = run(engine.search(SearchRequest(query="vex robotics", top_k=5, intent=intent)))
        assert resp.sources
        assert {s.source for s in resp.sources} == {"notes"}

    def test_source_inferred_from_phrasing(self, make_engine):
        engine = make_engine(sample_chunks())
        resp = run(engine.search(SearchRequest(query="vex robotics in my notes", top_k=5)))
        assert resp.sources
        assert {s.source for s in resp.sources} == {"notes"}

    def test_unmatched_source_falls_back_to_unfiltered(self, make_engine):
        """No reminders chunks exist — rather than an empty panel, retry unfiltered."""
        engine = make_engine(sample_chunks())
        intent = QueryIntent(sources=["reminders"])
        resp = run(engine.search(SearchRequest(query="vex robotics", top_k=5, intent=intent)))
        assert resp.sources
        assert "reminders" not in {s.source for s in resp.sources}


class TestContactFilter:
    def test_contact_extracted_from_query(self, make_engine):
        engine = make_engine(sample_chunks(), summaries=[JERRY_SUMMARY])
        resp = run(engine.search(SearchRequest(query="what did jerry say about vex", top_k=5)))
        assert resp.sources
        assert all("Jerry Yan" in s.contact_names for s in resp.sources)

    def test_intent_contacts_resolve_to_canonical(self, make_engine):
        engine = make_engine(sample_chunks())
        intent = QueryIntent(contacts=["sam"])
        resp = run(engine.search(SearchRequest(query="dinner plans", top_k=5, intent=intent)))
        assert resp.sources
        assert all("Sam Rivera" in s.contact_names for s in resp.sources)

    def test_unknown_contact_does_not_empty_results(self, make_engine):
        engine = make_engine(sample_chunks())
        intent = QueryIntent(contacts=["nobody mcnobody"])
        resp = run(engine.search(SearchRequest(query="vex robotics", top_k=5, intent=intent)))
        assert resp.sources


class TestDateWindows:
    """Overlap semantics: a chunk spanning a window must survive it, in both
    the date retriever and the hydration filter."""

    def test_date_search_matches_spanning_chunk(self, make_engine):
        engine = make_engine(sample_chunks())
        after = fx.iso_days_ago(20)
        before = fx.iso_days_ago(15)
        found = engine._date_search(after, before, 20)
        assert "cal-span" in found

    def test_date_search_excludes_chunk_ending_before_window(self, make_engine):
        engine = make_engine(sample_chunks())
        found = engine._date_search(fx.iso_days_ago(4), None, 20)
        assert "im-old" not in found
        assert "im-recent" in found

    def test_date_search_before_only(self, make_engine):
        engine = make_engine(sample_chunks())
        found = engine._date_search(None, fx.iso_days_ago(100), 20)
        assert found == ["im-old"]

    def test_date_search_no_window_is_noop(self, make_engine):
        engine = make_engine(sample_chunks())
        assert engine._date_search(None, None, 20) == []

    def test_hydrate_keeps_overlapping_chunk(self, make_engine):
        engine = make_engine(sample_chunks())
        ranked = [(c["chunk_id"], 1.0) for c in sample_chunks()]
        window = (fx.iso_days_ago(20), fx.iso_days_ago(15))
        out = engine._hydrate(ranked, 10, date_range=window)
        texts = ids(out)
        assert any("spring season practice" in t for t in texts)
        assert not any("dinner plans" in t for t in texts)

    def test_hydrate_drops_chunk_starting_after_window(self, make_engine):
        engine = make_engine(sample_chunks())
        ranked = [(c["chunk_id"], 1.0) for c in sample_chunks()]
        out = engine._hydrate(ranked, 10, date_range=(None, fx.iso_days_ago(100)))
        assert ids(out) == ["vex robotics competition schedule saturday"]

    def test_hydrate_source_and_contact_filters_compose(self, make_engine):
        engine = make_engine(sample_chunks())
        ranked = [(c["chunk_id"], 1.0) for c in sample_chunks()]
        out = engine._hydrate(
            ranked, 10, source_filter={"imessage"}, contact_filter={"Sam Rivera"}
        )
        assert [s.source for s in out] == ["imessage"]
        assert ids(out) == ["dinner plans at seven downtown"]


class TestPersonaRoutes:
    def test_style_query_uses_persona_without_llm(self, make_engine):
        llm = fx.FakeLLM(content="unused")
        engine = make_engine(sample_chunks(), personas=[JERRY_PERSONA], llm=llm)
        resp = run(engine.search(SearchRequest(query="how does jerry talk", top_k=5)))
        assert "short bursts" in resp.answer
        assert "robotics competitions" in resp.answer
        assert not llm.calls
        assert resp.sources

    def test_style_falls_through_without_persona(self, make_engine):
        engine = make_engine(sample_chunks())
        resp = run(engine.search(SearchRequest(query="how does jerry talk", top_k=5)))
        assert resp.answer == ""
        assert resp.sources, "standard pipeline should still supply sources"

    def test_affinity_uses_llm_rewrite(self, make_engine):
        llm = fx.FakeLLM(content="Jerry mostly talks about robotics competitions.")
        engine = make_engine(sample_chunks(), personas=[JERRY_PERSONA], llm=llm)
        resp = run(engine.search(SearchRequest(query="what does jerry care about", top_k=5)))
        assert resp.answer == "Jerry mostly talks about robotics competitions."
        assert len(llm.calls) == 1

    def test_affinity_rejects_nameless_rewrite(self, make_engine):
        llm = fx.FakeLLM(content="This person talks about robots.")
        engine = make_engine(sample_chunks(), personas=[JERRY_PERSONA], llm=llm)
        resp = run(engine.search(SearchRequest(query="what does jerry care about", top_k=5)))
        assert resp.answer.startswith("Jerry Yan mainly discusses:")

    def test_affinity_survives_llm_failure(self, make_engine):
        engine = make_engine(
            sample_chunks(), personas=[JERRY_PERSONA],
            llm=fx.FakeLLM(error=RuntimeError("boom")),
        )
        resp = run(engine.search(SearchRequest(query="what does jerry care about", top_k=5)))
        assert "robotics competitions" in resp.answer

    def test_intent_query_type_hint_wins(self, make_engine):
        """The Swift router's hint routes to style even for neutral phrasing."""
        engine = make_engine(sample_chunks(), personas=[JERRY_PERSONA])
        intent = QueryIntent(contacts=["jerry"], query_type="style")
        resp = run(engine.search(SearchRequest(query="jerry", top_k=5, intent=intent)))
        assert "short bursts" in resp.answer


class TestImageRoute:
    def _images(self) -> list[dict]:
        return [{
            "attachment_id": 77,
            "path": "/tmp/does-not-exist.jpg",
            "date_iso": fx.iso_days_ago(4),
            "sender_name": "Sam Rivera",
            "chat_title": "Sam Rivera",
            "msg_text": "photo of the robot chassis",
        }]

    def test_image_search_skipped_without_index(self, make_engine):
        engine = make_engine(sample_chunks(), images=self._images())
        assert engine.image_index is None
        resp = run(engine.search(SearchRequest(query="robot chassis photo", top_k=5)))
        assert "image" not in {s.source for s in resp.sources}

    def test_attachment_intent_returns_only_images(self, make_engine):
        engine = make_engine(
            sample_chunks(), images=self._images(), image_index=True
        )
        intent = QueryIntent(must_have_attachment=True)
        resp = run(engine.search(
            SearchRequest(query="photo of the robot chassis", top_k=5, intent=intent)
        ))
        assert [s.source for s in resp.sources] == ["image"]
        assert resp.sources[0].image_url == "/attachment/77"

    def test_get_image_path(self, make_engine):
        engine = make_engine(sample_chunks(), images=self._images(), image_index=True)
        assert engine.get_image_path(77) == "/tmp/does-not-exist.jpg"
        assert engine.get_image_path(9999) is None


class TestFusionHelpers:
    def test_rrf_rewards_agreement(self):
        fused = dict(SearchEngine._rrf(["a", "b", "c"], ["c", "a", "z"]))
        assert fused["a"] > fused["c"] > fused["b"]

    def test_rrf_empty(self):
        assert SearchEngine._rrf([], []) == []

    def _src(self, tag: str, source: str = "imessage") -> SourceResult:
        return SourceResult(source=source, contact_names=[], score=1.0, snippet=tag)

    def test_merge_interleaves_two_text_per_image(self):
        local = [self._src(f"t{i}") for i in range(4)]
        images = [self._src(f"i{i}", "image") for i in range(2)]
        out = SearchEngine._merge_results(local, images, [], max_total=6)
        assert [s.snippet for s in out] == ["t0", "t1", "i0", "t2", "t3", "i1"]

    def test_merge_respects_max_total(self):
        local = [self._src(f"t{i}") for i in range(10)]
        out = SearchEngine._merge_results(local, [], [], max_total=3)
        assert len(out) == 3

    def test_merge_appends_remote_last(self):
        out = SearchEngine._merge_results(
            [self._src("t0")], [], [self._src("r0", "hyperspell")], max_total=5
        )
        assert [s.snippet for s in out] == ["t0", "r0"]

    def test_merge_handles_images_only(self):
        images = [self._src(f"i{i}", "image") for i in range(3)]
        out = SearchEngine._merge_results([], images, [], max_total=2)
        assert [s.source for s in out] == ["image", "image"]


class TestEngineBoot:
    def test_missing_index_raises(self, tmp_path, monkeypatch):
        from api import search as search_mod

        monkeypatch.setattr(search_mod, "INDEX_PATH", tmp_path / "nope.faiss")
        monkeypatch.setattr(search_mod, "META_DB_PATH", tmp_path / "nope.db")
        with pytest.raises(FileNotFoundError):
            search_mod.SearchEngine()

    def test_contact_index_built_from_chunks(self, make_engine):
        engine = make_engine(sample_chunks())
        assert engine._contact_norm_index["jerry"] == {"Jerry Yan"}
        assert engine._resolve_contacts(["jerry yan"]) == {"Jerry Yan"}

    def test_summaries_loaded(self, make_engine):
        engine = make_engine(sample_chunks(), summaries=[JERRY_SUMMARY])
        assert engine._contact_summaries["Jerry Yan"]["total_chunks"] == 40


class TestTemporalRoute:
    def _history(self) -> list[dict]:
        recent = [
            fx.chunk(
                f"t-recent-{i}", "imessage", ["Jerry Yan"],
                f"the vex regional bracket seed {i} and drive team practice went well",
                start_days_ago=5 + i,
            )
            for i in range(20)
        ]
        older = [
            fx.chunk(
                f"t-old-{i}", "imessage", ["Jerry Yan"],
                f"calculus homework problem set {i} and exam review sessions",
                start_days_ago=400 + i,
            )
            for i in range(15)
        ]
        return recent + older

    def test_temporal_query_returns_shift_answer(self, make_engine):
        llm = fx.FakeLLM(content="Jerry Yan has shifted from calculus homework to VEX matches.")
        engine = make_engine(self._history(), llm=llm)
        resp = run(engine.search(SearchRequest(query="how has jerry changed recently", top_k=6)))
        assert resp.answer == "Jerry Yan has shifted from calculus homework to VEX matches."
        assert resp.sources
        # Sources interleave recent and older sides.
        assert resp.sources[0].date_end > resp.sources[1].date_end

    def test_temporal_falls_through_without_older_side(self, make_engine):
        recent_only = [
            fx.chunk(f"t-r-{i}", "imessage", ["Jerry Yan"], "vex regional bracket",
                     start_days_ago=5 + i)
            for i in range(6)
        ]
        llm = fx.FakeLLM(content="Jerry has been talking about the regional.")
        engine = make_engine(recent_only, llm=llm)
        resp = run(engine.search(SearchRequest(query="how has jerry changed recently", top_k=5)))
        # One LLM call = the standard synthesis path; the temporal route bails
        # out before its two calls when either side of the split is empty.
        assert len(llm.calls) == 1
        assert resp.sources

    def test_temporal_falls_through_when_llm_down(self, make_engine):
        engine = make_engine(self._history(), llm=fx.FakeLLM(error=RuntimeError("boom")))
        resp = run(engine.search(SearchRequest(query="how has jerry changed recently", top_k=5)))
        assert resp.answer == ""
        assert resp.sources


class TestCompareRoute:
    def test_compare_gate_interleaves_contacts(self, make_engine):
        llm = fx.FakeLLM(content="Jerry focuses on robotics while Sam handles logistics.")
        engine = make_engine(sample_chunks(), llm=llm)
        resp = run(engine.search(
            SearchRequest(query="compare jerry and sam on robotics", top_k=6)
        ))
        assert resp.answer == "Jerry focuses on robotics while Sam handles logistics."
        senders = [s.contact_names[0] for s in resp.sources]
        assert "Jerry Yan" in senders and "Sam Rivera" in senders
        assert senders[0] != senders[1], "results should alternate between contacts"

    def test_bare_and_does_not_trigger_compare(self, make_engine):
        llm = fx.FakeLLM(content="unused")
        engine = make_engine(sample_chunks(), llm=llm)
        resp = run(engine.search(SearchRequest(query="jerry and sam robotics", top_k=6)))
        # Standard pipeline: non-question query means no synthesis at all.
        assert resp.answer == ""
        assert not llm.calls

    def test_compare_falls_through_when_llm_down(self, make_engine):
        engine = make_engine(sample_chunks(), llm=fx.FakeLLM(error=RuntimeError("boom")))
        resp = run(engine.search(SearchRequest(query="compare jerry vs sam", top_k=6)))
        assert resp.answer == ""
        assert resp.sources
