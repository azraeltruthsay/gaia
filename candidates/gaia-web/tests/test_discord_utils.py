"""Tests for Discord utility functions (no bot connection required)."""

from gaia_web.discord_interface import DiscordInterface


class TestSplitMessage:
    """Test the _split_message helper used for Discord's 2000 char limit."""

    def _splitter(self, content, max_length=2000):
        # DiscordInterface.__init__ just stores bot_token and core_endpoint
        iface = DiscordInterface(bot_token=None, core_endpoint=None)
        return iface._split_message(content, max_length=max_length)

    def test_short_message_unchanged(self):
        result = self._splitter("Hello world")
        assert result == ["Hello world"]

    def test_exact_limit(self):
        msg = "A" * 2000
        result = self._splitter(msg)
        assert len(result) == 1
        assert result[0] == msg

    def test_long_message_splits(self):
        msg = "A" * 3000
        result = self._splitter(msg, max_length=2000)
        assert len(result) >= 2
        # Reassembled content should cover the original
        total_len = sum(len(chunk) for chunk in result)
        assert total_len >= 3000

    def test_splits_on_newline(self):
        # Build a message with a newline near the split point
        first_half = "A" * 1500 + "\n"
        second_half = "B" * 600
        msg = first_half + second_half
        result = self._splitter(msg, max_length=2000)
        # Should prefer splitting at the newline
        assert result[0].endswith("A")

    def test_splits_on_space(self):
        # No newlines, but has spaces
        msg = ("word " * 500).strip()  # ~2499 chars
        result = self._splitter(msg, max_length=2000)
        assert len(result) >= 2
        # First chunk should end on a word boundary
        assert not result[0].endswith(" ")

    def test_empty_string(self):
        result = self._splitter("")
        assert result == [""]
