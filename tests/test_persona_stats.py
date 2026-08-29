from persona_builder import (
    _compute_style_stats,
    _emoji_count,
    escape_like,
    filter_topic_labels,
    is_junk_topic_label,
)


def _msgs(*texts):
    return [{"text": t} for t in texts]


class TestEmojiCount:
    def test_plain_latin_is_zero(self):
        assert _emoji_count("hey are we still on for tonight?") == 0

    def test_cjk_is_not_emoji(self):
        assert _emoji_count("今天晚上一起吃饭吗？我们七点见") == 0

    def test_emoji_counted(self):
        assert _emoji_count("sounds good 😂🎉") == 2


class TestStyleStats:
    def test_brief_bucket(self):
        avg, freq, style = _compute_style_stats(_msgs("ok", "yes", "lol sure"))
        assert style == "brief"
        assert freq == 0

    def test_verbose_bucket(self):
        long = "x" * 250
        _, _, style = _compute_style_stats(_msgs(long, long))
        assert style == "verbose"

    def test_cjk_contact_low_emoji_freq(self):
        _, freq, _ = _compute_style_stats(_msgs("今天晚上一起吃饭吗", "好的没问题"))
        assert freq == 0


class TestJunkTopicLabels:
    def test_question_words_are_junk(self):
        assert is_junk_topic_label("who")
        assert is_junk_topic_label("what")

    def test_verb_phrases_are_junk(self):
        assert is_junk_topic_label("scored them")
        assert is_junk_topic_label("did say")

    def test_short_or_letterless_is_junk(self):
        assert is_junk_topic_label("ok")
        assert is_junk_topic_label("")
        assert is_junk_topic_label("123!")

    def test_real_topics_kept(self):
        assert not is_junk_topic_label("robotics competition")
        assert not is_junk_topic_label("uncertainty")
        assert not is_junk_topic_label("request")
        assert not is_junk_topic_label("weekend plans")

    def test_filter_drops_junk_and_dupes(self):
        topics = [
            {"topic": "robotics competition", "score": 0.4},
            {"topic": "who", "score": 0.3},
            {"topic": "Robotics Competition", "score": 0.2},
            {"topic": "scored them", "score": 0.1},
            {"topic": "college apps", "score": 0.1},
        ]
        out = filter_topic_labels(topics)
        assert [t["topic"] for t in out] == ["robotics competition", "college apps"]


class TestEscapeLike:
    def test_wildcards_escaped(self):
        assert escape_like("50%_off") == "50\\%\\_off"

    def test_plain_name_unchanged(self):
        assert escape_like("Jerry Yan") == "Jerry Yan"
