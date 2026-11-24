"""What kind of site a web source is, in a few plain words (w8c9).

2026-11-23 DM, "civilizations earlier than the Sumerians?": a search returned
ithy.com, a site whose pages an AI writes, and she said "Research published
recently on sites like Ithy confirms...". She was shown a title and an
address, nothing about what kind of place it was, so she couldn't weigh it
against the encyclopedia next to it.

The words go next to the source in what she reads ("AI-written answer page",
"encyclopedia"). A site not listed gets nothing: its address is still there,
and a guessed kind would be worse than none.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

AI_ANSWER = "AI-written answer page, not research"

# Lowercase prose, short: all-caps labels get quoted back by Core.
_SITES = {
    # Pages written by a language model in answer to a query.
    AI_ANSWER: ("ithy.com", "perplexity.ai", "you.com", "phind.com", "iask.ai",
                "komo.ai", "andisearch.com", "aitopics.org", "aiforanswers.com",
                "chatgpt.com", "gemini.google.com", "copilot.microsoft.com"),
    "encyclopedia": ("wikipedia.org", "britannica.com", "worldhistory.org",
                     "newworldencyclopedia.org", "encyclopedia.com",
                     "plato.stanford.edu", "iep.utm.edu", "scholarpedia.org"),
    "library of texts": ("wikisource.org", "gutenberg.org", "archive.org",
                         "poetryfoundation.org", "poets.org"),
    "journal or paper": ("arxiv.org", "nature.com", "science.org", "ncbi.nlm.nih.gov",
                         "pubmed.ncbi.nlm.nih.gov", "jstor.org", "doi.org", "sciencedirect.com",
                         "springer.com", "link.springer.com", "wiley.com", "plos.org",
                         "biorxiv.org", "medrxiv.org", "pnas.org", "cell.com",
                         "tandfonline.com", "academic.oup.com",
                         "semanticscholar.org", "researchgate.net"),
    "dictionary": ("merriam-webster.com", "dictionary.com", "thefreedictionary.com",
                   "dictionary.cambridge.org", "collinsdictionary.com", "oed.com",
                   "wiktionary.org", "vocabulary.com", "etymonline.com"),
    "news site": ("bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "nytimes.com",
                  "theguardian.com", "washingtonpost.com", "npr.org", "cnn.com",
                  "aljazeera.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com",
                  "theatlantic.com", "economist.com", "ft.com", "wsj.com", "bloomberg.com",
                  "politico.com", "axios.com", "usatoday.com", "latimes.com"),
    "forum or Q&A site": ("reddit.com", "quora.com", "stackoverflow.com",
                          "stackexchange.com", "answers.com", "brainly.com",
                          "answers.yahoo.com", "news.ycombinator.com"),
    "blog": ("medium.com", "substack.com", "blogspot.com", "wordpress.com",
             "tumblr.com", "livejournal.com"),
    "code documentation": ("docs.python.org", "developer.mozilla.org", "cppreference.com",
                           "rust-lang.org", "readthedocs.io"),
    "code repository": ("github.com", "gitlab.com"),
    "video": ("youtube.com", "youtu.be", "vimeo.com"),
}
_BY_SITE = {site: kind for kind, sites in _SITES.items() for site in sites}

_BLOG_PATH_RE = re.compile(r"^/(?:blog|blogs|posts?)(?:/|$)", re.I)


def _host_and_path(url_or_domain: str):
    s = (url_or_domain or "").strip()
    if not s:
        return "", ""
    parts = urlparse(s if "://" in s else f"https://{s}")
    return parts.netloc.lower().split(":")[0].removeprefix("www."), parts.path or ""


def site_kind(url_or_domain: str) -> Optional[str]:
    """'https://ithy.com/article/x' -> 'AI-written answer page, not research';
    a site not known -> None."""
    host, path = _host_and_path(url_or_domain)
    if not host:
        return None
    # Most specific first: "plato.stanford.edu" before "stanford.edu".
    labels = host.split(".")
    for i in range(len(labels) - 1):
        kind = _BY_SITE.get(".".join(labels[i:]))
        if kind:
            return kind
    if labels[-1] == "gov" or ".gov." in f".{host}.":
        return "government site"
    if labels[-1] == "edu" or re.search(r"\.ac\.[a-z]{2}$", host):
        return "university site"
    if _BLOG_PATH_RE.match(path) or labels[0] == "blog":
        return "blog post"
    return None
