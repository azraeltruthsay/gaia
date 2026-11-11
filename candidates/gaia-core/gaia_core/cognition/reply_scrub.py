"""Scrub what Core writes around a reply, after generation (8ail, vjrv).

The same few things turn up at the head of a reply, mostly when she answers
from a tool result: an alpaca-corpus tag or JSON wrapper from her training
data, the engine's clock line, or the continuation prompt's own cue. None of
it is hers to say. A process, not a prompt instruction: telling a small model
what not to write backfires.

scrub_reply() cleans a finished reply. HeadScrubber does the same to a token
stream, holding back only as much as it needs to decide.
"""
import json
import re
from typing import AsyncIterator

_TAG_RE = re.compile(r"</?(?:nooutput|blockquote)>", re.IGNORECASE)
_CLOCK_RE = re.compile(r"\[Clock:[^\]\n]*\]\s*")
_CUE_RE = re.compile(
    r"(?:They asked|The user asked|You already did that).*?Your reply to them[^:\n]*:\s*",
    re.DOTALL | re.IGNORECASE,
)
_WRAPPER_RE = re.compile(r'\{\s*"(?:content|response|output|text)"\s*:\s*"')
_WRAPPER_END = '"}'
# Her own turn marker, written back at the start of a reply.
_TURN_RE = re.compile(r"(?:<\|im_start\|>\s*)?assistant\s*(?:\n|:)\s*", re.IGNORECASE)
# A note to herself, which the streaming path never took out (c0rv).
_SEED_RE = re.compile(r"THOUGHT_SEED:[^\n]*\n+\s*")
_JUNK = (_TAG_RE, _CLOCK_RE, _CUE_RE, _WRAPPER_RE, _TURN_RE, _SEED_RE)
# What may open a reply without being the reply: the tier label, a status line.
_LABEL_RE = re.compile(r"(?:\[(?!Clock:)[^\]\n]{1,40}\]|\*\[[^\n]*\]\*)\s+")

_OPENINGS = ("thought_seed:", "<nooutput>", "<blockquote>", "<|im_start|>", "assistant", "[clock:", "they asked", "the user asked",
             "you already did that", '{"content"', '{"response"',
             '{"output"', '{"text"')
_TAGS = ("<nooutput>", "</nooutput>", "<blockquote>", "</blockquote>")
# A cue echo quotes the request, so it can run long; past this it isn't one.
_MAX_HOLD = 1500


def _may_grow_into_something(body: str) -> bool:
    """Whether more of the stream could make `body` junk or a label."""
    if len(body) >= _MAX_HOLD:
        return False
    low = body.lower()
    if low.startswith("assistant") and len(low) > len("assistant") and low[len("assistant")] not in " \n:":
        return False
    if low.startswith(("assistant ",)) and low[len("assistant"):].strip(" ")[:1] not in ("", "\n", ":"):
        return False
    if any(o.startswith(low) or low.startswith(o) for o in _OPENINGS):
        return True
    if body.startswith("["):
        return "]" not in body and len(body) <= 41 or body.endswith("]")
    return body.startswith("*") and "\n" not in body and (len(body) == 1 or body.startswith("*["))


def _read_head(body: str):
    """The next thing at the front of `body` (no leading space): ('junk', its
    match), ('label', its match), or ('body', None)."""
    for pattern in _JUNK:
        m = pattern.match(body)
        if m:
            return "junk", m
    m = _LABEL_RE.match(body)
    return ("label", m) if m else ("body", None)


def scrub_reply(text: str) -> str:
    """A finished reply without the junk. A reply that is nothing but junk is
    left alone: she may really have been asked for the clock."""
    if not text:
        return text
    s = HeadScrubber()
    cleaned = s.feed(text) + s.flush()
    return cleaned if cleaned.strip() else text


class HeadScrubber:
    """Scrubs a token stream, holding back only what it can't yet decide on.
    Each flush starts a fresh head, since what follows a tool call is a new
    generation."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self._head = ""       # text not yet decided on
        self._space = ""      # space after what was let through, kept for the reply
        self._after_junk = False
        self._in_head = True
        self._wrapped = False
        self._tail = ""       # body held back: maybe a tag or wrapper end

    def feed(self, text: str) -> str:
        if not self._in_head:
            return self._body(text)
        self._head += text
        out = ""
        while self._in_head:
            body = self._head.lstrip()
            if not body:
                break
            # The space junk leaves behind goes with it.
            lead = "" if self._after_junk else self._head[:len(self._head) - len(body)]
            kind, m = _read_head(body)
            if kind == "junk":
                self._wrapped = self._wrapped or m.re is _WRAPPER_RE
                self._space += lead
                self._after_junk = True
                self._head = body[m.end():]
            elif kind == "label":
                label = m.group(0).rstrip()
                out += self._space + lead + label
                self._space = m.group(0)[len(label):]
                self._after_junk = False
                self._head = body[m.end():]
            elif _may_grow_into_something(body):
                break
            else:
                self._in_head = False
                self._head = ""
                out += self._body(self._space + lead + body)
                self._space = ""
        return out

    def _body(self, text: str) -> str:
        text = _TAG_RE.sub("", self._tail + text)
        low = text.lower()
        hold = max((n for tag in _TAGS for n in range(1, len(tag)) if low.endswith(tag[:n])), default=0)
        if self._wrapped:
            hold = max(hold, min(len(text), len(_WRAPPER_END) + 2))
        self._tail = text[len(text) - hold:] if hold else ""
        return text[:len(text) - hold] if hold else text

    def flush(self) -> str:
        """Everything still held, at a flush or the end of the stream."""
        head = self._head.lstrip() if self._after_junk else self._head
        out = _TAG_RE.sub("", self._space + head + self._tail)
        wrapped = self._wrapped
        self._reset()
        if wrapped and out.rstrip().endswith(_WRAPPER_END):
            out = out.rstrip()[:-len(_WRAPPER_END)]
        return out


# How a sentence sounds when more is coming: an announcement or a pleasantry.
_NOT_AN_ANSWER_RE = re.compile(
    r"(?:i'?ll|i will|i'?m|i am|i can|i need|i should|i'?d|let me|let'?s|one moment|just a|"
    r"hold on|give me|checking|looking|searching|fetching|reading|sure|okay|ok|of course|"
    r"certainly|absolutely|alright|right|hi|hello|hey|good|great|thanks|thank you|to answer|"
    r"first|here)\b",
    re.IGNORECASE,
)


# A reply that ends by handing the turn back: "What can I help with?". Told
# someone's pronouns, she said "Nice to meet you, Tamsin. What's on your mind?",
# then called a memory tool, and the continuation said it all again (z838).
_HANDS_BACK_RE = re.compile(
    r"^(?:so,?\s+|and\s+)?(?:what\s+(?:can|shall|should)\s+(?:I|we)\s+(?:help|do|get|start)|how\s+can\s+I\s+help"
    r"|what'?s\s+on\s+your\s+mind|what\s+brings\s+you|is\s+there\s+anything|anything\s+(?:else\s+)?I\s+can"
    r"|what\s+(?:are\s+you|would\s+you\s+like)\s+(?:up\s+to|working\s+on|to\s+(?:do|talk|work))"
    r"|what\s+can\s+I\s+get\s+(?:started\s+on|into))", re.IGNORECASE)


def hands_back(prose: str) -> bool:
    """Whether a reply ends by giving the turn back to them. A tool call after
    that is never wanted, whatever the intent: "Hi GAIA, I'm Tamsin. My
    pronouns are xe/xem." read as identity, not chat, so 8ail's check didn't run."""
    text = (prose or "").strip()
    return bool(text) and text[-1] == "?" and bool(_HANDS_BACK_RE.search(re.split(r"(?<=[.!?])\s+", text)[-1]))


def answered_already(prose: str) -> bool:
    """Whether what she wrote before a tool call already answers: 'Paris.' does,
    'I'll fetch the page.' and 'Hi there!' don't. The call after an answer
    brought back a second one and a line of nonsense (8ail, 2026-11-03)."""
    text = (prose or "").strip()
    while True:
        m = _LABEL_RE.match(text + " ")
        if not m:
            break
        text = text[m.end():].strip()
    if hands_back(text):
        return True
    if not text or text[-1] not in ".!":
        return False
    last = re.split(r"(?<=[.!?])\s+", text)[-1]
    return not _NOT_AN_ANSWER_RE.match(last)


_DECLINES_RE = re.compile(
    r"(?:i'?m sorry|sorry|i'?m not sure|i don'?t|i do not|i can'?t|i cannot|i didn'?t|could you|what )",
    re.IGNORECASE,
)
_QUOTED_WORDS = 6


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'"))


def own_words(response: str, source: str) -> str:
    """The first sentence of what she said after reciting `source`, if it can
    stand beside the words: said in her own words, adding no name or number
    the text doesn't hold. Otherwise "", and the words go out alone. She read
    them either way. What she makes of them can still be wrong; this only
    catches what can be checked (uyor)."""
    first = re.split(r"(?<=[.!?])\s|\n", (response or "").strip(), maxsplit=1)[0].strip()
    if not first or len(first) > 300 or first[0] in '<{["*' or not first.endswith((".", "!")):
        return ""
    if _DECLINES_RE.match(first):
        return ""
    known = set(_words(source))
    said = _words(first)
    if any(any(c.isdigit() for c in w) and w not in known for w in said):
        return ""
    # A capital mid-sentence is a name: "Allen Poe", "November".
    for word in re.findall(r"(?<=[a-z,;:] )[A-Z][\w'\u2019-]*", first):
        if not set(_words(word)) <= known:
            return ""
    source_words = " " + " ".join(_words(source)) + " "
    for i in range(len(said) - _QUOTED_WORDS + 1):
        if " " + " ".join(said[i:i + _QUOTED_WORDS]) + " " in source_words:
            return ""
    return first


_CLAIMS_SUCCESS_RE = re.compile(
    r"\b(?:success\w*|succeeded|i(?:'ve| have)? (?:\w+ )?(?:updated|saved|added|stored|recorded|noted|"
    r"remembered|written|wrote|learned|fetched|found|read)|"
    r"(?:has|have|had) been \w+|(?:is|are|was|were) (?:now )?(?:updated|saved|added|stored|recorded|noted)|all set|done)\b",
    re.IGNORECASE,
)
_OWNS_FAILURE_RE = re.compile(
    r"\b(?:fail\w*|couldn'?t|can'?t|cannot|didn'?t|did not|unable|error|wasn'?t|isn'?t|not)\b", re.IGNORECASE,
)


def claims_success(reply: str) -> bool:
    """Whether a reply says a thing was done without owning any failure."""
    return bool(_CLAIMS_SUCCESS_RE.search(reply)) and not _OWNS_FAILURE_RE.search(reply)


# "I do recall that conversation now", "it comes back to me"; not "I don't recall".
_CLAIMS_MEMORY_RE = re.compile(
    r"\bI(?:'m| am)?\s+(?:(?:do|now|still|clearly|definitely|vaguely|certainly)\s+)*(?:recall|remember)\b"
    r"(?!\s+(?:if|whether)\b)"
    r"|\b(?:it|that|this)\s+(?:all\s+)?comes?\s+back\s+to\s+me\b",
    re.IGNORECASE,
)
_PALACE_RECALL_ALIASES = {"recalling", "remember", "remembering", "search", "find", "lookup",
                          "look_up", "retrieve", "query", "get", "read"}


def claims_memory(reply: str) -> bool:
    """Whether a reply says she remembers something."""
    return bool(_CLAIMS_MEMORY_RE.search(reply or ""))


def palace_recall_params(params: dict, user_input: str) -> dict:
    """A palace call as the tool takes it: near-miss actions are a recall, and a
    recall with nothing to look for looks for what she was just told."""
    params = dict(params or {})
    action = str(params.get("action") or "").strip().lower()
    if action in _PALACE_RECALL_ALIASES:
        params["action"] = action = "recall"
    if action == "recall" and not str(params.get("query") or "").strip():
        params["query"] = (user_input or "")[:500]
    return params


def honest_about_failure(events, tool_label: str, memory: bool = False):
    """A continuation after a failed tool, held until it can be checked: told
    the tool failed, she still said it worked 4/6 (knowledge add, 2026-11-03).
    Status markers were the only sign of the truth until 8ail removed them.
    `memory`: the tool was a memory lookup, and saying she remembers is the
    claim to check: "I do recall that conversation now" after palace refused
    the call (2026-11-11)."""
    words, others = [], []
    for event in events:
        if isinstance(event, dict) and event.get("type") == "token":
            words.append(event.get("value", ""))
        else:
            others.append(event)
    reply = "".join(words)
    if memory and claims_memory(reply):
        reply = "I tried to look that up in my memory just now and couldn't, so I won't claim to remember it."
    elif claims_success(reply):
        reply = f"That didn't work: {tool_label} failed, so nothing changed."
    if reply:
        yield {"type": "token", "value": reply}
    yield from others


# n828: corrected about something her last reply never said, she opened "My
# mistake" 4/4 anyway. Words that only open a sentence aren't names.
_NOT_NAMES = {"oh", "no", "yes", "well", "so", "actually", "but", "and", "i", "gaia", "hey", "ok", "okay",
              "the", "he", "she", "they", "it", "that", "this", "you", "we", "my", "your", "his", "her"}
_NAMEWORD_RE = re.compile(r"\b[A-Z][a-z]{1,}\b")
_APOLOGY_RE = re.compile(
    r"^(?P<lead>\s*(?:\[[^\]\n]{1,40}\]\s*)?)"
    r"(?:my\s+(?:mistake|apologies|bad)|(?:i'?m\s+)?sorry|apologies|i\s+apologi[sz]e)\b[^\u2014\u2013.!\n]{0,40}"
    r"(?:[\u2014\u2013.!]|\s-\s)\s*"
    r"(?:(?:you'?re|you\s+are)\s+right[,.!\u2014\u2013]?\s*)?"
    r"(?:(?:let\s+me|i'?ll)\s+(?:correct|fix|clarify)\s+(?:it|that|myself)[.!]\s*)?",
    re.IGNORECASE)
_APOLOGY_HOLD = 160


def unfounded_apology(asked: str, last_reply: str) -> bool:
    """Whether an apology would be for something her last reply never said: a
    correction naming someone her last reply doesn't mention. With no name to
    go by it can't tell, and says no."""
    from gaia_common.utils.conversation_memory import is_correction
    if not is_correction(asked) or not (last_reply or "").strip():
        return False
    names = {w for w in _NAMEWORD_RE.findall(asked or "") if w.lower() not in _NOT_NAMES}
    said = (last_reply or "").lower()
    return bool(names) and not any(n.lower() in said for n in names)


def trim_apology(text: str) -> str:
    """`text` without an apology opening its reply."""
    m = _APOLOGY_RE.match(text or "")
    if not m:
        return text
    rest = text[m.end():]
    return m.group("lead") + rest[:1].upper() + rest[1:]


async def scrub_stream(lines: AsyncIterator[str], drop_apology: bool = False) -> AsyncIterator[str]:
    """The /process_packet NDJSON stream with its token events scrubbed; every
    other event passes through, after anything held back. With `drop_apology`,
    the opening of her reply is held until an apology there can be cut."""
    scrubber = HeadScrubber()
    hold = [""] if drop_apology else None   # reply text held while it may open with one

    def _release(final: bool = False) -> str:
        text = hold[0]
        body = text.lstrip()
        if body.startswith("*[") and "\n" not in body and not final:
            return ""          # a header line still arriving
        if body.startswith("*["):
            hold[0] = ""
            return text        # a header line, not her reply: keep holding after it
        if not final and len(text) < _APOLOGY_HOLD:
            return ""
        hold.clear()           # decided, once
        return trim_apology(text)

    def _out(text: str) -> str:
        if hold is None or not hold:
            return text
        hold[0] += text
        return _release()

    async for line in lines:
        try:
            event = json.loads(line)
            is_token = isinstance(event, dict) and event.get("type") == "token"
        except (TypeError, ValueError):
            event, is_token = None, False
        if is_token:
            text = _out(scrubber.feed(event.get("value") or ""))
            if text:
                yield json.dumps({"type": "token", "value": text}) + "\n"
            continue
        held = _out(scrubber.flush())
        if hold and hold[0]:
            held += _release(final=True)
        if held:
            yield json.dumps({"type": "token", "value": held}) + "\n"
        yield line
    held = _out(scrubber.flush())
    if hold and hold[0]:
        held += _release(final=True)
    if held:
        yield json.dumps({"type": "token", "value": held}) + "\n"
