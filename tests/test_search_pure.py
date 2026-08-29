from api.search import (
    _classify_query_type,
    _normalize_name,
    _strip_invented_quote_marks,
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
