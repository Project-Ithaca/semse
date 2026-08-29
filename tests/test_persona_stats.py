from persona_builder import _compute_style_stats, _emoji_count, escape_like


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


class TestEscapeLike:
    def test_wildcards_escaped(self):
        assert escape_like("50%_off") == "50\\%\\_off"

    def test_plain_name_unchanged(self):
        assert escape_like("Jerry Yan") == "Jerry Yan"
