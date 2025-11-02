"""Structured representations for parsed resumes and job descriptions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from matcher.ingestion.pdf_reader import PdfDocument
from matcher.ingestion.section_parser import (
    ParsedSections,
    extract_job_sections,
    extract_resume_sections,
)
from matcher.normalization.text import RussianTextNormalizer


ITEM_SPLIT_RE = re.compile(r"[•\-–—·•]+|\n+")


def _split_items(text: str) -> List[str]:
    items = []
    for raw in ITEM_SPLIT_RE.split(text):
        candidate = raw.strip(" •-\t\r\n")
        if candidate:
            items.append(candidate)
    return items


@dataclass
class ResumeContent:
    path: Path
    sections: ParsedSections
    tokens: List[str]
    skills: List[str]
    experiences: List[str]
    educations: List[str]
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    location_code: Optional[str] = None
    locality: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    schedule_types: List[str] = field(default_factory=list)
    relocation_status: Optional[str] = None
    travel_status: Optional[str] = None
    education_level: Optional[int] = None
    experience_years: Optional[float] = None
    last_experience_years: Optional[float] = None
    skill_text: str = ""
    experience_text: str = ""


@dataclass
class JobDescriptionContent:
    path: Path
    sections: ParsedSections
    tokens: List[str]
    requirements: List[str]
    responsibilities: List[str]
    conditions: List[str]
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    location_code: Optional[str] = None
    languages_required: List[str] = field(default_factory=list)
    schedule_types: List[str] = field(default_factory=list)
    relocation_required: Optional[str] = None
    travel_required: Optional[str] = None
    education_level_required: Optional[int] = None
    experience_required_years: Optional[float] = None
    requirement_text: str = ""
    responsibility_text: str = ""


def parse_resume(document: PdfDocument, normalizer: RussianTextNormalizer | None = None) -> ResumeContent:
    normalizer = normalizer or RussianTextNormalizer()
    sections = extract_resume_sections(document)
    tokens = normalizer.normalize_tokens(document.text)
    skills = _split_items(sections.get("skills"))
    experiences = _split_items(sections.get("experience"))
    educations = _split_items(sections.get("education"))
    return ResumeContent(
        path=document.path,
        sections=sections,
        tokens=tokens,
        skills=skills,
        experiences=experiences,
        educations=educations,
        salary_min=None,
        salary_max=None,
        location_code=None,
        locality=None,
        languages=[],
        schedule_types=[],
        relocation_status=None,
        travel_status=None,
        education_level=None,
        experience_years=None,
        last_experience_years=None,
        skill_text="\n".join(skills),
        experience_text="\n".join(experiences),
    )


def parse_job_description(document: PdfDocument, normalizer: RussianTextNormalizer | None = None) -> JobDescriptionContent:
    normalizer = normalizer or RussianTextNormalizer()
    sections = extract_job_sections(document)
    tokens = normalizer.normalize_tokens(document.text)
    requirements = _split_items(sections.get("requirements"))
    responsibilities = _split_items(sections.get("responsibilities"))
    conditions = _split_items(sections.get("conditions"))
    return JobDescriptionContent(
        path=document.path,
        sections=sections,
        tokens=tokens,
        requirements=requirements,
        responsibilities=responsibilities,
        conditions=conditions,
        salary_min=None,
        salary_max=None,
        location_code=None,
        languages_required=[],
        schedule_types=[],
        relocation_required=None,
        travel_required=None,
        education_level_required=None,
        experience_required_years=None,
        requirement_text="\n".join(requirements),
        responsibility_text="\n".join(responsibilities),
    )
