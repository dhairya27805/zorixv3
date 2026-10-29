"""
zorix/data/news_data.py

News sentiment (Section 17), refactored out of app.py.

IMPORTANT CHANGE from the original app.py: the old build_sentiment_features()
fabricated a full history of "sentiment" as 0.45 * return*10 + noise — i.e.
it derived a historical feature FROM the return it was later used to predict.
That is leakage dressed up as a feature, and it is removed here.

What this module does instead:
  - fetch_headlines(): live headlines via Google News RSS (no API key)
  - score_headlines(): VADER sentiment on those headlines
  - FinBERT hook: use_finbert() if transformers+FinBERT are installed, else
    falls back to VADER and says so — never silently swaps quality without
    telling the caller (Section 43).

There is no synthetic historical sentiment. If you want a real historical
sentiment feature for model TRAINING (not just today's snapshot), you need
a real historical news archive/API — plug it in as a new function here
following the same "raise / return unavailable, never fabricate" pattern
as data/options_data.py and data/futures_data.py.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("zorix.data.news_data")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from transformers import pipeline as _hf_pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False


@dataclass
class SentimentResult:
    compound: float
    pos: float
    neg: float
    neutral: float
    n_headlines: int
    engine: str  # "finbert" | "vader" | "unavailable"


def fetch_headlines(query: str, max_headlines: int = 15, timeout: int = 8) -> List[str]:
    """Live headlines from Google News RSS. Returns [] on any network failure
    — callers must show 'no headlines available', never invent any.
    """
    try:
        q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", raw)
        if not titles:
            titles = re.findall(r"<title>(.*?)</title>", raw)
        return [t for t in titles if len(t) > 10][1:max_headlines + 1]
    except Exception as exc:
        logger.warning("Headline fetch failed for query=%r: %s", query, exc)
        return []


_finbert_pipe = None


def _get_finbert():
    global _finbert_pipe
    if _finbert_pipe is None and FINBERT_AVAILABLE:
        _finbert_pipe = _hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _finbert_pipe


def score_headlines(headlines: List[str], prefer_finbert: bool = True) -> SentimentResult:
    """Score headlines; prefers FinBERT if installed & requested, else VADER,
    else reports engine='unavailable' with neutral zeros (not fabricated
    non-zero values).
    """
    if not headlines:
        return SentimentResult(0.0, 0.0, 0.0, 1.0, 0, "unavailable")

    if prefer_finbert and FINBERT_AVAILABLE:
        try:
            pipe = _get_finbert()
            results = pipe(headlines, truncation=True)
            pos = sum(1 for r in results if r["label"].lower() == "positive") / len(results)
            neg = sum(1 for r in results if r["label"].lower() == "negative") / len(results)
            neu = 1 - pos - neg
            compound = pos - neg
            return SentimentResult(compound, pos, neg, neu, len(headlines), "finbert")
        except Exception as exc:
            logger.warning("FinBERT scoring failed, falling back to VADER: %s", exc)

    if VADER_AVAILABLE:
        analyser = SentimentIntensityAnalyzer()
        scores = [analyser.polarity_scores(h) for h in headlines]
        return SentimentResult(
            compound=float(sum(s["compound"] for s in scores) / len(scores)),
            pos=float(sum(s["pos"] for s in scores) / len(scores)),
            neg=float(sum(s["neg"] for s in scores) / len(scores)),
            neutral=float(sum(s["neu"] for s in scores) / len(scores)),
            n_headlines=len(headlines),
            engine="vader",
        )

    return SentimentResult(0.0, 0.0, 0.0, 1.0, len(headlines), "unavailable")


def score_individual_headlines(headlines: List[str]) -> List[dict]:
    """Per-headline scores for display (e.g. colored dots next to each headline)."""
    if VADER_AVAILABLE:
        analyser = SentimentIntensityAnalyzer()
        return [
            {"headline": h, "compound": analyser.polarity_scores(h)["compound"]}
            for h in headlines
        ]
    return [{"headline": h, "compound": None} for h in headlines]
