"""Letter-counting questions: she works them out herself, in a scratch step.

Her tokenizer reads "strawberry" in chunks, so asked cold she guesses. It
used to be code that counted and handed her the answer as "pre-computed
data", and on 2026-11-26 it counted the word "word" for 'how many l's are in
the word "hillbilly"' and told her 0, so she answered "None".

Measured 2026-11-26 at temperature 0, 18 words (strawberry ... possession):
- Prime can't do it: she spells 2 of 12 words right, collapsing double
  letters ("strawbery", "misipi", "comite"). So Core walks the word in every
  gear; in gear 2 Prime answers from Core's working.
- Core asked cold: 8/12. Spell then count: 9/12 (spelled right, then jumped
  to a wrong total). Walking the word letter by letter:
                        running  hyphen-first  numbered
      Core on GPU (Q6)   18/18      17/18        17/18
      Core on CPU (Q4)   15/18      16/18        17/18
- Two different walks that agree were right every time (95 of 95 across
  both copies and every pair); every wrong total showed up as a
  disagreement. So she walks it twice, with the best pair for where Core
  runs; if the totals differ, the third walk breaks the tie (majority of
  three: GPU 18/18, CPU 17/18 with one no-majority); if no two agree, she is
  told her counts differ, not handed one.
So code only recognises the question and compares her two totals; the
counting is hers.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

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

WALKS: Dict[str, Callable[[str, str], str]] = {
    "running": lambda w, c: (f'How many {c}\'s are in the word "{w}"? Go through the word one letter per line. '
                             f'After each "{c}", write how many you have counted so far. Then give the total on the last line.'),
    "hyphen": lambda w, c: (f'How many {c}\'s are in the word "{w}"? First write the word with a hyphen between every '
                            f'letter. Then go through that spelling one letter per line, and after each "{c}" write how '
                            f'many you have counted so far. Give the total on the last line.'),
    "numbered": lambda w, c: (f'How many {c}\'s are in the word "{w}"? Write every letter of the word on its own line, '
                              f'numbered from 1. After each "{c}", also write how many you have counted so far. '
                              f'Give the total on the last line.'),
}
# The pair that measured best where Core runs (see the table above).
PAIR_ON_GPU = ("running", "numbered")
PAIR_ON_CPU = ("numbered", "hyphen")

_NUMBER_WORDS = {"zero": 0, "none": 0, "once": 1, "one": 1, "twice": 2, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def working_prompt(word: str, letter: str, walk: str = "running") -> str:
    return WALKS[walk](word, letter)


def total_of(working: str) -> Optional[int]:
    """The total her working ends with (its last number)."""
    found = re.findall(r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\b", (working or "").lower())
    if not found:
        return None
    last = found[-1]
    return int(last) if last.isdigit() else _NUMBER_WORDS[last]


def work_it_out(text: str, generate: Callable[[List[dict]], str], on_gpu: bool = True) -> str:
    """Her own letter-by-letter working for a letter-counting question, as a
    note for her prompt, or "" (not such a question, or the walks failed).
    `generate(messages) -> str` is Core, wherever it runs."""
    found = letter_question(text)
    if not found or not found[1]:
        return ""       # spelling alone she does right without help
    word, letter = found
    pair = PAIR_ON_GPU if on_gpu else PAIR_ON_CPU
    third = next(w for w in WALKS if w not in pair)
    workings: List[str] = []
    for walk in (*pair, third):
        if len(workings) == 2 and total_of(workings[0]) == total_of(workings[1]):
            break               # two walks agree: no tiebreak needed
        try:
            w = (generate([{"role": "user", "content": working_prompt(word, letter, walk)}]) or "").strip()
        except Exception:
            w = ""
        if w and total_of(w) is not None:
            workings.append(w)
    totals = [total_of(w) for w in workings]
    agreed = next((t for t in totals if totals.count(t) >= 2), None)
    if agreed is not None:
        working = next(w for w in workings if total_of(w) == agreed)
        return (f'[Your own working, done a moment ago: you went through "{word}" letter by letter, '
                f'counting each "{letter}".]\n{working}\n[Answer the question from your working.]')
    if len(totals) >= 2:
        # No two walks agree: one or more misspelled the word. Only the fact
        # reaches her: shown the workings, she picked the wrong one and talked
        # about "the second one", which the person never saw (mississippi).
        got = " and ".join(str(t) for t in dict.fromkeys(totals))
        return (f'[You counted the "{letter}"s in "{word}" letter by letter {len(totals)} times and got different '
                f'totals ({got}), so you are not sure of the count.]')
    return (f'[Your own working, done a moment ago: you went through "{word}" letter by letter, '
            f'counting each "{letter}".]\n{workings[0]}\n[Answer the question from your working.]') if workings else ""
