"""Letter-counting questions: she works them out herself, in a scratch step.

Her tokenizer reads "strawberry" in chunks, so asked cold she guesses. It
used to be code that counted and handed her the answer as "pre-computed
data", and on 2026-11-26 it counted the word "word" for 'how many l's are in
the word "hillbilly"' and told her 0, so she answered "None".

Measured on Core (temperature 0, 12 words): plain question 8/12; spell it
out then count 9/12 (she spelled every word right, then jumped to a wrong
total); one letter per line with a running count of matches 12/12, as good
as code. So code only recognises the question; she walks the word in a
hidden step and answers from her own working.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

_QUOTES = "\"'“”‘’`*"
# Words that sit between "in" and the word being asked about.
_FILLER = frozenset("the word a an this that name term spelling of".split())

_Q = "[" + re.escape(_QUOTES) + "]"
# The letter must be marked as one -- "the letter r", "'r'", "r's", "rs" --
# or "how many I can eat in a day" would count I's in "day".
_COUNT_RE = re.compile(
    r"how\s+many\s+(?:times\s+(?:does|do|is|will)\s+)?(?:the\s+)?"
    r"(?:(?:letter|character)\s+" + _Q + r"?(?P<a>[a-z])" + _Q + r"?"
    r"|" + _Q + r"(?P<b>[a-z])" + _Q + r"(?:['’]?s)?"
    r"|(?P<c>[a-z])['’]?s)\b(?P<rest>.*)",
    re.IGNORECASE | re.DOTALL)
_SPELL_RE = re.compile(
    r"(?:\bspell(?:ing)?|\b(?:letters?|characters?)\s+(?:are\s+|is\s+)?(?:there\s+)?(?:in|does|of)"
    r"|\bbreak\s*down)\b(?P<rest>.*)",
    re.IGNORECASE | re.DOTALL)
_QUOTED_RE = re.compile(r"[\"“”‘’'`*]+([A-Za-z][A-Za-z'-]*[A-Za-z]|[A-Za-z])[\"“”‘’'`*]+")


def _target_word(rest: str, after_in: bool) -> Optional[str]:
    """The word a counting/spelling question is about, from the text after
    the letter (or after 'spell')."""
    quoted = _QUOTED_RE.search(rest)
    if quoted:
        return quoted.group(1).lower()
    if after_in:
        m = re.search(r"\b(?:in|of|inside)\b(.*)", rest, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        rest = m.group(1)
    for w in re.findall(r"[A-Za-z][A-Za-z'-]*", rest):
        w = w.strip("'-").lower()
        if w and w not in _FILLER:
            return w if len(w) >= 2 else None
    return None


def letter_question(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """(word, letter) for 'how many r's in strawberry', (word, None) for
    'spell necessary', or None."""
    m = _COUNT_RE.search(text or "")
    if m:
        word = _target_word(m.group("rest"), after_in=True)
        if word:
            return word, (m.group("a") or m.group("b") or m.group("c")).lower()
    m = _SPELL_RE.search(text or "")
    if m:
        word = _target_word(m.group("rest"), after_in=False)
        if word and len(word) >= 3:
            return word, None
    return None


WORKING_MAX_TOKENS = 400


def working_prompt(word: str, letter: str) -> str:
    return (f'How many {letter}\'s are in the word "{word}"? Go through the word one letter per line. '
            f'After each "{letter}", write how many you have counted so far. Then give the total on the last line.')


def work_it_out(text: str, generate: Callable[[List[dict]], str]) -> str:
    """Her own letter-by-letter working for a letter-counting question, as a
    note for her prompt, or "" (not such a question, or the step failed).
    `generate(messages) -> str` is the mind that will answer."""
    found = letter_question(text)
    if not found or not found[1]:
        return ""       # spelling alone she does right without help
    word, letter = found
    try:
        working = (generate([{"role": "user", "content": working_prompt(word, letter)}]) or "").strip()
    except Exception:
        return ""
    if not working:
        return ""
    return (f'[Your own working, done a moment ago: you went through "{word}" letter by letter, '
            f'counting each "{letter}".]\n{working}\n[Answer the question from your working.]')
