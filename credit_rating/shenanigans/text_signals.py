"""Text-based shenanigan signal detectors.

Covers Gunning Fog readability index, FinBERT sentiment tone shift
between consecutive MD&A sections, and TF-IDF cosine similarity
for boilerplate detection across years.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.domain.shenanigans import TextSignals, ToneShiftResult

logger = logging.getLogger(__name__)


class TextSignalDetector:
    """Detect manipulation signals from textual disclosures.

    Args:
        settings: Configuration for all detection thresholds.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
    ) -> None:
        self._s = settings or CreditRatingSettings()

    def detect(
        self,
        risk_factors_text: str,
        current_mda: str = "",
        prior_mda: str = "",
        prior_risk_text: str = "",
    ) -> TextSignals:
        """Run all text-based signal detectors.

        Args:
            risk_factors_text: Current-period risk factors section.
            current_mda: Current-period MD&A text.
            prior_mda: Prior-period MD&A text (empty if unavailable).
            prior_risk_text: Prior-period risk factors (empty if
                unavailable).

        Returns:
            A :class:`TextSignals` with all computed signals.
        """
        fog = compute_fog_index(risk_factors_text)
        fog_flagged = fog > self._s.fog_index_cutoff

        tone_shift = None
        if current_mda and prior_mda:
            tone_shift = self.compute_tone_shift(current_mda, prior_mda)

        boilerplate_sim = None
        boilerplate_flagged = False
        if risk_factors_text and prior_risk_text:
            boilerplate_sim = compute_boilerplate_score(
                risk_factors_text, prior_risk_text,
            )
            boilerplate_flagged = boilerplate_sim > self._s.cosine_similarity_cutoff

        return TextSignals(
            fog_index=fog,
            fog_is_flagged=fog_flagged,
            tone_shift=tone_shift,
            boilerplate_similarity=boilerplate_sim,
            boilerplate_is_flagged=boilerplate_flagged,
        )

    def compute_tone_shift(
        self,
        current_mda: str,
        prior_mda: str,
    ) -> ToneShiftResult:
        """Measure sentiment shift between two MD&A sections.

        Uses FinBERT (or a simple heuristic fallback) to score the
        positive sentiment of each text and compute the delta.

        Args:
            current_mda: Current-period MD&A.
            prior_mda: Prior-period MD&A.

        Returns:
            A :class:`ToneShiftResult` with scores and flag.
        """
        current_score = _sentiment_score(current_mda)
        prior_score = _sentiment_score(prior_mda)
        delta = abs(current_score - prior_score)
        return ToneShiftResult(
            current_positive_score=current_score,
            prior_positive_score=prior_score,
            delta=delta,
            is_flagged=delta > self._s.tone_shift_delta_cutoff,
        )


# ------------------------------------------------------------------
# Gunning Fog Index
# ------------------------------------------------------------------


def compute_fog_index(text: str) -> float:
    """Compute the Gunning Fog readability index.

    ``FOG = 0.4 * (ASL + PHW)``

    where ASL = average sentence length, PHW = percentage of
    hard words (3+ syllables).

    Args:
        text: The document text to analyse.

    Returns:
        The Fog index as a float. Higher values indicate more
        complex, harder-to-read text.
    """
    sentences = _split_sentences(text)
    words = _split_words(text)
    if not sentences or not words:
        return 0.0

    average_sentence_length = len(words) / len(sentences)
    hard_word_count = sum(
        1 for w in words if _count_syllables(w) >= 3
    )
    percent_hard_words = (hard_word_count / len(words)) * 100

    return 0.4 * (average_sentence_length + percent_hard_words)


# ------------------------------------------------------------------
# Boilerplate detection
# ------------------------------------------------------------------


def compute_boilerplate_score(
    current_text: str,
    prior_text: str,
) -> float:
    """Compute TF-IDF cosine similarity between two texts.

    High similarity indicates boilerplate copying without meaningful
    disclosure updates.

    Args:
        current_text: Current-period disclosure.
        prior_text: Prior-period disclosure.

    Returns:
        Cosine similarity in ``[0, 1]``.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform([current_text, prior_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except ImportError:
        logger.warning("sklearn not available; returning 0.0 similarity")
        return 0.0


# ------------------------------------------------------------------
# NLP helpers
# ------------------------------------------------------------------


def _sentiment_score(text: str) -> float:
    """Compute a positive-sentiment score for *text*.

    Tries FinBERT first; falls back to a simple positive-word ratio.
    """
    try:
        return _finbert_positive_score(text[:5000])
    except (ImportError, RuntimeError):
        return _simple_positive_ratio(text)


def _finbert_positive_score(text: str) -> float:
    """Score positive sentiment using the FinBERT pipeline."""
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
    )
    result = classifier(text[:512])
    for item in result:
        if item["label"].lower() == "positive":
            return float(item["score"])
    return 0.0


_POSITIVE_WORDS = frozenset(
    "growth increase improve strong positive gain profit".split()
)
_NEGATIVE_WORDS = frozenset(
    "decline decrease loss risk negative weak impair".split()
)


def _simple_positive_ratio(text: str) -> float:
    """Fallback: ratio of positive to total sentiment words."""
    words = text.lower().split()
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.5
    return pos / total


def _split_sentences(text: str) -> list:
    """Split text into sentences using punctuation."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _split_words(text: str) -> list:
    """Split text into words."""
    return re.findall(r"[a-zA-Z]+", text)


def _count_syllables(word: str) -> int:
    """Estimate syllable count for an English word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)
