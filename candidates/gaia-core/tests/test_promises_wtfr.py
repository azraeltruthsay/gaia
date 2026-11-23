"""wtfr: what counts as a promise to look something up."""
import pytest

from gaia_core.cognition.promises import made_promise


@pytest.mark.parametrize("reply,asked,want", [
    ("I'll look it up. Give me a sec.", "", True),
    ("Good question. I'll check the SBCC site and get back to you.", "", True),
    ("Sure, I can check.", "Can you check and get back to me?", True),
    ("I can search for it. Want me to?", "", False),
    ("The 2021 IBC is in effect in Washington.", "", False),
    # the tool call is the lookup; the answer follows in the same reply
    ("Let me check.\n<|tool>issue(action=list, status=open)<tool|>", "", False),
    ('Let me check. <tool_call>{"tool": "issue", "action": "list"}</tool_call>', "", False),
])
def test_made_promise(reply, asked, want):
    assert made_promise(reply, asked) is want
