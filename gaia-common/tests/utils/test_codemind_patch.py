"""Tests: CodeMind SEARCH/REPLACE patch format (s4r2) — exact and unique
matches only; ambiguity is an error, never a guess."""

from gaia_common.utils.codemind_patch import (
    PatchBlock,
    apply_patch,
    parse_patch_blocks,
)

SOURCE = """def greet(name):
    msg = "hello"
    return msg + name


def farewell(name):
    msg = "bye"
    return msg + name
"""


def _mk(search, replace):
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


class TestParse:
    def test_single_block(self):
        blocks, err = parse_patch_blocks(_mk('    msg = "hello"', '    msg = "hello, "'))
        assert err is None and len(blocks) == 1
        assert blocks[0].search == '    msg = "hello"'

    def test_multiple_blocks(self):
        text = _mk("a = 1", "a = 2") + "\n\n" + _mk("b = 3", "b = 4")
        blocks, err = parse_patch_blocks(text)
        assert err is None and len(blocks) == 2

    def test_empty_response(self):
        blocks, err = parse_patch_blocks("   ")
        assert err and not blocks

    def test_no_blocks(self):
        blocks, err = parse_patch_blocks("here is my fix: change x to y")
        assert err and "no SEARCH/REPLACE blocks" in err

    def test_stray_marker_rejected(self):
        text = _mk("a = 1", "a = 2") + "\n=======\nleftover"
        blocks, err = parse_patch_blocks(text)
        assert err and "stray" in err

    def test_identical_search_replace_rejected(self):
        blocks, err = parse_patch_blocks(_mk("a = 1", "a = 1"))
        assert err and "identical" in err


class TestApply:
    def test_simple_replace(self):
        blocks, _ = parse_patch_blocks(_mk('    msg = "hello"', '    msg = "hello, "'))
        out, err = apply_patch(SOURCE, blocks)
        assert err is None
        assert '"hello, "' in out
        assert out.count("def ") == 2  # untouched code survives

    def test_ambiguous_search_rejected(self):
        # "    return msg + name" appears in both functions
        blocks = [PatchBlock(search="    return msg + name", replace="    return name")]
        out, err = apply_patch(SOURCE, blocks)
        assert out is None and "2 locations" in err

    def test_no_match_rejected(self):
        blocks = [PatchBlock(search="not in the file", replace="x")]
        out, err = apply_patch(SOURCE, blocks)
        assert out is None and "not found" in err

    def test_multi_block_sequential(self):
        text = _mk('    msg = "hello"', '    msg = "hi"') + "\n" + _mk('    msg = "bye"', '    msg = "later"')
        blocks, perr = parse_patch_blocks(text)
        assert perr is None
        out, err = apply_patch(SOURCE, blocks)
        assert err is None and '"hi"' in out and '"later"' in out

    def test_uniqueness_by_context(self):
        # The ambiguous line becomes unique with a disambiguating neighbor
        blocks, _ = parse_patch_blocks(
            _mk('    msg = "bye"\n    return msg + name',
                '    msg = "bye"\n    return msg + ", " + name'))
        out, err = apply_patch(SOURCE, blocks)
        assert err is None
        assert 'msg + ", " + name' in out
        assert out.index("hello") < out.index('", " + name')  # greet untouched

    def test_no_change_rejected(self):
        out, err = apply_patch(SOURCE, [])
        assert out is None and err
