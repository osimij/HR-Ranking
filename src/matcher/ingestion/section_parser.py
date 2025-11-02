"""Heuristic section extractors for resumes and job descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from matcher.ingestion.pdf_reader import PdfDocument
from matcher.normalization.text import RussianTextNormalizer


RESUME_SECTION_KEYWORDS = {
    "header": {"контакты", "информация", "summary", "профиль"},
    "experience": {
        "опыт работы",
        "опыт",
        "experience",
        "профессиональный опыт",
        "projects",
    },
    "education": {"образование", "education"},
    "skills": {
        "навыки",
        "skills",
        "профессиональные навыки",
        "hard skills",
        "soft skills",
        "компетенции",
    },
    "certifications": {"сертификаты", "сертификации"},
}

JD_SECTION_KEYWORDS = {
    "responsibilities": {
        "обязанности",
        "responsibilities",
        "чем ты будешь заниматься",
        "вы будете",
        "вы быть",
        "основные задачи",
        "чем предстоит заниматься",
    },
    "requirements": {
        "требования",
        "requirements",
        "что мы ожидаем",
        "ожидаем",
        "мы ожидаем",
        "мы ожидать",
        "что мы ожидать",
    },
    "conditions": {
        "условия",
        "conditions",
        "что мы предлагаем",
        "мы предлагаем",
        "предлагаем",
        "что мы предлагать",
        "мы предлагать",
    },
    "company": {"компания", "company", "о компании"},
}


@dataclass
class ParsedSections:
    """Container for extracted section text."""

    sections: Dict[str, str]

    def get(self, key: str, default: str = "") -> str:
        return self.sections.get(key, default)


class SectionExtractor:
    """Extracts text blocks following heuristic section headers."""

    def __init__(self, keywords: Dict[str, Iterable[str]], *, normalizer: Optional[RussianTextNormalizer] = None) -> None:
        self.normalizer = normalizer or RussianTextNormalizer()
        self.keywords = {
            name: {self._normalize_keyword(keyword) for keyword in values}
            for name, values in keywords.items()
        }

    def from_document(self, document: PdfDocument) -> ParsedSections:
        sections = {name: [] for name in self.keywords}
        current_section: Optional[str] = None
        for raw_line in document.lines:
            normalized_line = raw_line.strip()
            if not normalized_line:
                continue
            matched_section = self._match_section(normalized_line)
            if matched_section:
                current_section = matched_section
                continue
            if current_section:
                sections[current_section].append(normalized_line)
        result = {section: "\n".join(lines).strip() for section, lines in sections.items() if lines}
        return ParsedSections(result)

    def _match_section(self, line: str) -> Optional[str]:
        normalized_line = self.normalizer.normalize_text(line)
        for section, keywords in self.keywords.items():
            for keyword in keywords:
                if normalized_line.startswith(keyword) or keyword in normalized_line:
                    return section
        return None

    def _normalize_keyword(self, keyword: str) -> str:
        return self.normalizer.normalize_text(keyword)


def extract_resume_sections(document: PdfDocument) -> ParsedSections:
    extractor = SectionExtractor(RESUME_SECTION_KEYWORDS)
    return extractor.from_document(document)


def extract_job_sections(document: PdfDocument) -> ParsedSections:
    extractor = SectionExtractor(JD_SECTION_KEYWORDS)
    return extractor.from_document(document)
