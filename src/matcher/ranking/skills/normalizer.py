"""Skill normalization utilities for Russian/English technical vocab."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

from matcher.normalization.text import RussianTextNormalizer

_SKILL_SYNONYMS = {
    "js": "javascript",
    "javascript": "javascript",
    "nodejs": "node.js",
    "node": "node.js",
    "ts": "typescript",
    "cs": "c#",
    "csharp": "c#",
    "golang": "go",
    "reactjs": "react",
    "react.js": "react",
    "html5": "html",
    "css3": "css",
    "postgresql": "postgres",
    "pgsql": "postgres",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "аналитик данных": "data analyst",
    "devops": "devops",
    "docker-compose": "docker",
    "ms sql": "mssql",
    "sql server": "mssql",
    "jira": "jira",
    "git": "git",
}

_STOPWORDS = {
    "и",
    "та",
    "такая",
    "такой",
    "на",
    "с",
    "опыт",
    "опытом",
    "работа",
    "работой",
    "уровень",
}

_TOKEN_SPLIT = re.compile(r"[\s/,;]+")


class SkillNormalizer:
    """Normalize skill phrases, handling Russian/English variants."""

    def __init__(self, normalizer: RussianTextNormalizer | None = None) -> None:
        self.normalizer = normalizer or RussianTextNormalizer(keep_digits=True)

    @lru_cache(maxsize=2048)
    def normalize(self, skill: str) -> str:
        lowered = skill.strip().lower()
        lowered = lowered.replace("+", " ").replace("#", " #")
        tokens: List[str] = []
        for raw_token in _TOKEN_SPLIT.split(lowered):
            if not raw_token:
                continue
            lemma = self.normalizer.normalize_text(raw_token)
            lemma = lemma.replace("  ", " ")
            if lemma in _STOPWORDS:
                continue
            lemma = _SKILL_SYNONYMS.get(lemma, lemma)
            tokens.append(lemma)
        return " ".join(tokens).strip()

    def normalize_many(self, skills: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        for skill in skills:
            norm = self.normalize(skill)
            if norm:
                normalized.append(norm)
        return normalized

