"""Language-specific text normalization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List


TOKEN_SPLIT_RE = re.compile(r"[\s,;:.!?()\[\]{}<>/\\\"'«»]+")


class NormalizationDependencyError(ImportError):
    """Raised when required normalization dependencies are missing."""


def _load_morph_analyzer():
    try:
        import pymorphy2  # type: ignore
    except ImportError as exc:  # pragma: no cover - informative error
        raise NormalizationDependencyError(
            "pymorphy2 is required for Russian lemmatization. "
            "Install it via `pip install pymorphy2`."
        ) from exc
    return pymorphy2.MorphAnalyzer()


@dataclass
class RussianTextNormalizer:
    """Normalizes Russian text snippets for downstream matching."""

    keep_digits: bool = False
    _lemma_cache: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._morph = _load_morph_analyzer()

    def normalize_tokens(self, text: str) -> List[str]:
        """Return lemmatized tokens from text."""
        text_lower = self._preprocess(text)
        tokens: List[str] = []
        for raw_token in TOKEN_SPLIT_RE.split(text_lower):
            token = raw_token.strip()
            if not token:
                continue
            if not self.keep_digits and token.isdigit():
                continue
            tokens.append(self._lemmatize(token))
        return tokens

    def normalize_text(self, text: str, separator: str = " ") -> str:
        """Return normalized text joined by separator."""
        return separator.join(self.normalize_tokens(text))

    def _preprocess(self, text: str) -> str:
        normalized = text.lower()
        return re.sub(r"\s+", " ", normalized)

    def _lemmatize(self, token: str) -> str:
        if token in self._lemma_cache:
            return self._lemma_cache[token]
        parsed = self._morph.parse(token)
        lemma = parsed[0].normal_form if parsed else token
        self._lemma_cache[token] = lemma
        return lemma


def normalize_lines(lines: Iterable[str], normalizer: RussianTextNormalizer) -> List[str]:
    """Normalize each line with the provided normalizer."""
    return [normalizer.normalize_text(line) for line in lines if line.strip()]
