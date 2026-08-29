from api.search import (
    _build_fts_match,
    _classify_query_type,
    _normalize_name,
    _persona_topics,
    _strip_invented_quote_marks,
    _strip_query_tokens,
    _validate_no_invented_quotes,
)


class TestClassifyQueryType:
    def test_style_queries(self):
        assert _classify_query_type("how does jerry talk", True) == "style"
        assert _classify_query_type("How does Jerry write?", True) == "style"
        assert _classify_query_type("what is jerry like", True) == "style"
        assert _classify_query_type("what's jerry like", True) == "style"

    def test_affinity_queries(self):
        assert _classify_query_type("what does sarah care about", True) == "affinity"
        assert _classify_query_type("what topics does sarah like", True) == "affinity"
        assert _classify_query_type("what does sarah talk about", True) == "affinity"

    def test_temporal_queries(self):
        assert _classify_query_type("how has alex changed recently", True) == "temporal"
        assert (
            _classify_query_type("what has alex been thinking about recently", True)
            == "temporal"
        )

    def test_standard_queries(self):
        assert _classify_query_type("what did sarah say about the trip", True) == "standard"
        assert _classify_query_type("dinner plans", True) == "standard"

    def test_requires_contact(self):
        assert _classify_query_type("how does jerry talk", False) == "standard"
        assert _classify_query_type("what does sarah care about", False) == "standard"


class TestNormalizeName:
    def test_lowercase_and_strip(self):
        assert _normalize_name("Jerry Yan") == _normalize_name("jerry yan")
        assert _normalize_name("O'Brien") == _normalize_name("obrien")


class TestBuildFtsMatch:
    def test_stopwords_and_punctuation_stripped(self):
        match = _build_fts_match("when is the vex regional competition?")
        assert match == "vex* OR regional* OR competition*"

    def test_all_stopwords_yields_empty(self):
        assert _build_fts_match("what is the") == ""

    def test_punctuation_only_yields_empty(self):
        assert _build_fts_match("?!...") == ""


class TestStripQueryTokens:
    def test_strips_contact_tokens(self):
        out = _strip_query_tokens("what did ruthvik say about cad", {"ruthvik"})
        assert out == "what did say about cad"

    def test_case_and_punctuation_insensitive(self):
        out = _strip_query_tokens("What did Ruthvik, say", {"ruthvik"})
        assert out == "What did say"

    def test_name_only_query_kept_intact(self):
        assert _strip_query_tokens("ruthvik", {"ruthvik"}) == "ruthvik"

    def test_no_tokens_no_change(self):
        q = "what did ruthvik say"
        assert _strip_query_tokens(q, set()) == q


class TestPersonaTopics:
    def test_junk_labels_filtered_at_read_time(self):
        import json

        persona = {
            "top_topics": json.dumps(
                [
                    {"topic": "scored them", "score": 0.3},
                    {"topic": "who", "score": 0.25},
                    {"topic": "request", "score": 0.2},
                    {"topic": "uncertainty", "score": 0.15},
                ]
            )
        }
        topics = [t["topic"] for t in _persona_topics(persona)]
        assert topics == ["request", "uncertainty"]

    def test_malformed_json_returns_empty(self):
        assert _persona_topics({"top_topics": "not json"}) == []
        assert _persona_topics({"top_topics": None}) == []


class TestQuoteValidation:
    def test_valid_quote_passes(self):
        ok, _ = _validate_no_invented_quotes(
            'Sarah said "see you at noon" yesterday.',
            "Sarah: see you at noon\nMe: ok",
        )
        assert ok

    def test_invented_quote_fails(self):
        ok, bad = _validate_no_invented_quotes(
            'Sarah said "I never want to go" apparently.',
            "Sarah: see you at noon\nMe: ok",
        )
        assert not ok
        assert bad

    def test_strip_invented_quotes_keeps_text(self):
        out = _strip_invented_quote_marks(
            'Sarah said "I never want to go" apparently.',
            "Sarah: see you at noon",
        )
        assert '"I never want to go"' not in out
        assert "I never want to go" in out


class TestInferSources:
    def test_notes_phrase(self):
        from api.search import _infer_sources_from_query
        assert _infer_sources_from_query("what did I write in my notes about robotics") == {"notes"}

    def test_whatsapp(self):
        from api.search import _infer_sources_from_query
        assert _infer_sources_from_query("what did we talk about on whatsapp") == {"whatsapp"}

    def test_calls(self):
        from api.search import _infer_sources_from_query
        assert _infer_sources_from_query("who did I call last month") == {"calls"}

    def test_no_source_mentioned(self):
        from api.search import _infer_sources_from_query
        assert _infer_sources_from_query("what did jerry say about the trip") is None

    def test_bare_notes_word_not_hijacked(self):
        from api.search import _infer_sources_from_query
        assert _infer_sources_from_query("did sarah take notes during class") is None
