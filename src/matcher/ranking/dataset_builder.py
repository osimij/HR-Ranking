"""Build feature tables from Kaggle dataset rows."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from matcher.ingestion.dataset import (
    STATUS_RELEVANCE_MAP,
    group_by_vacancy,
    sample_balanced_groups,
)
from matcher.ingestion.models import JobDescriptionContent, ParsedSections, ResumeContent
from matcher.normalization.text import RussianTextNormalizer
from matcher.ranking.features import FeatureExtractor, FeatureVector, features_from_bm25
from matcher.ranking.skills.normalizer import SkillNormalizer
from matcher.retrieval.bm25 import BM25ResumeIndex
from matcher.embeddings.semantic import SemanticEmbedder


SKILL_NORMALIZER = SkillNormalizer()

_TOKEN_SPLIT_RE = re.compile(r"[\n•;\-–—]+")
_EDUCATION_LEVELS = {
    "дошкольное": 0,
    "начальное": 1,
    "среднее": 2,
    "среднее профессиональное": 3,
    "неполное высшее": 3,
    "бакалавр": 4,
    "специалист": 5,
    "магистр": 6,
    "магистратура": 6,
    "аспирантура": 7,
    "phd": 7,
    "кандидат наук": 7,
}


def _safe_json_load(value: str) -> Iterable:
    value = (value or "").strip()
    if not value or value in {"[]", "null", "None", "nan"}:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Attempt to unescape double quotes
        try:
            return json.loads(value.replace('""', '"'))
        except json.JSONDecodeError:
            return []


def _flatten_json_items(value: str) -> List[str]:
    data = _safe_json_load(value)
    items: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    items.append(candidate)
            elif isinstance(item, dict):
                for payload in item.values():
                    if isinstance(payload, str):
                        candidate = payload.strip()
                        if candidate:
                            items.append(candidate)
    return items


def _field_to_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    value_str = str(value).strip()
    if not value_str or value_str in {"[]", "nan"}:
        return []
    json_items = _flatten_json_items(value_str)
    if json_items:
        return json_items
    pieces = re.split(r"[\\n•;\\-–—]+", value_str)
    return [piece.strip() for piece in pieces if piece.strip()]


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    value_str = str(value).strip().replace(" ", " ")
    if not value_str:
        return None
    match = re.search(r"[-\d]+", value_str.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _parse_language_list(value) -> List[str]:
    items = _safe_json_load(str(value)) if isinstance(value, str) else value
    languages: List[str] = []
    if isinstance(items, list):
        for entry in items:
            if isinstance(entry, dict):
                code = entry.get("codeLanguage") or entry.get("language")
                level = entry.get("level")
                if code:
                    code_norm = str(code).strip().lower()
                    if level:
                        languages.append(f"{code_norm}:{str(level).lower()}")
                    else:
                        languages.append(code_norm)
            elif isinstance(entry, str):
                languages.append(entry.strip().lower())
    elif isinstance(items, str):
        languages.append(items.strip().lower())
    return [lang for lang in languages if lang]


def _parse_schedule(value) -> List[str]:
    schedules = _field_to_list(value)
    return [item.lower() for item in schedules]


def _parse_relocation(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text


def _parse_education_level(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    for key, level in _EDUCATION_LEVELS.items():
        if key in text:
            return level
    return None


def _parse_salary_range(row: pd.Series, prefix: str) -> Tuple[Optional[float], Optional[float]]:
    keys = [
        (f"{prefix}Min", f"{prefix}Max", prefix),
    ]
    if "_" in prefix:
        left, right = prefix.split("_", 1)
        keys.append((f"{left}Min_{right}", f"{left}Max_{right}", prefix))
        keys.append((f"{left}Min_{right}", f"{left}Max_{right}", f"{left}_{right}"))
        keys.append((f"{left}_{right}Min", f"{left}_{right}Max", prefix))
        keys.append((f"{left}_{right}Min", f"{left}_{right}Max", f"{left}_{right}"))
    for min_key, max_key, direct_key in keys:
        min_val = _parse_float(row.get(min_key))
        max_val = _parse_float(row.get(max_key))
        direct_val = _parse_float(row.get(direct_key))
        if min_val is None and direct_val is not None:
            min_val = direct_val
        if max_val is None and direct_val is not None:
            max_val = direct_val
        if min_val is not None or max_val is not None:
            return min_val, max_val
    return None, None


def _parse_iso_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _experience_stats(entries: Iterable[dict]) -> Tuple[Optional[float], Optional[float]]:
    total_days = 0
    last_end: Optional[datetime] = None
    now = datetime.utcnow()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        start = _parse_iso_date(entry.get("dateFrom", ""))
        end = _parse_iso_date(entry.get("dateTo", "")) or start
        if start and end:
            duration = (end - start).days
            if duration > 0:
                total_days += duration
            if last_end is None or (end and end > last_end):
                last_end = end
    years = total_days / 365.25 if total_days else None
    gap_years = None
    if last_end:
        gap_years = max((now - last_end).days / 365.25, 0.0)
    return years, gap_years


def _extract_resume_content(row: pd.Series, normalizer: RussianTextNormalizer) -> ResumeContent:
    raw_skills = _field_to_list(row.get("skills_cv"))
    raw_skills += _field_to_list(row.get("hardSkills_cv"))
    raw_skills += _field_to_list(row.get("softSkills_cv"))
    skills = SKILL_NORMALIZER.normalize_many(raw_skills)

    experience_entries = list(_safe_json_load(str(row.get("workExperienceList", ""))))
    experiences: List[str] = []
    for entry in experience_entries:
        if isinstance(entry, dict):
            for key in ("jobTitle", "demands", "description"):
                payload = entry.get(key)
                if isinstance(payload, str) and payload.strip():
                    experiences.append(payload.strip())

    education_entries = list(_safe_json_load(str(row.get("educationList", ""))))
    educations: List[str] = []
    for entry in education_entries:
        if isinstance(entry, dict):
            parts = [
                str(entry.get("instituteName", "")).strip(),
                str(entry.get("departmentName", "")).strip(),
                str(entry.get("educationLevel", "")).strip(),
            ]
            candidate = " ".join(filter(None, parts)).strip()
            if candidate:
                educations.append(candidate)

    text_sources = _field_to_list(row.get("positionName"))
    text_sources += _field_to_list(row.get("experience"))
    text_sources += _field_to_list(row.get("languageKnowledge_cv"))
    text_sources += skills + experiences + educations
    tokens = normalizer.normalize_tokens(" ".join(filter(None, text_sources)))

    salary_min, salary_max = _parse_salary_range(row, "salary_cv")
    languages = _parse_language_list(row.get("languageKnowledge_cv"))
    schedule_types = _parse_schedule(row.get("scheduleType_cv"))
    relocation_status = _parse_relocation(row.get("relocation"))
    travel_status = _parse_relocation(row.get("businessTrip"))
    education_level = _parse_education_level(row.get("education"))
    experience_years, gap_years = _experience_stats(experience_entries)

    sections = ParsedSections(
        {
            "skills": "\n".join(skills),
            "experience": "\n".join(experiences),
            "education": "\n".join(educations),
        }
    )

    return ResumeContent(
        path=Path(f"{row.get('idCv', 'cv')}.pdf"),
        sections=sections,
        tokens=tokens,
        skills=skills,
        experiences=experiences,
        educations=educations,
        salary_min=salary_min,
        salary_max=salary_max,
        location_code=str(row.get("locality") or row.get("localityName") or "").strip() or None,
        locality=str(row.get("localityName") or row.get("locality") or "").strip() or None,
        languages=languages,
        schedule_types=schedule_types,
        relocation_status=relocation_status,
        travel_status=travel_status,
        education_level=education_level,
        experience_years=experience_years,
        last_experience_years=gap_years,
        skill_text="; ".join(skills),
        experience_text="\n".join(experiences),
    )



def _extract_job_content(row: pd.Series, normalizer: RussianTextNormalizer) -> JobDescriptionContent:
    requirement_fields = [
        "positionRequirements",
        "experienceRequirements",
        "educationRequirements",
        "qualifications",
        "skills_vacancy",
        "hardSkills_vacancy",
        "softSkills_vacancy",
    ]
    requirements: List[str] = []
    for field in requirement_fields:
        if field in row:
            requirements += _field_to_list(row.get(field))

    responsibilities = _field_to_list(row.get("responsibilities"))

    conditions_fields = ["conditions", "otherVacancyBenefit", "careerPerspective", "benefit"]
    conditions: List[str] = []
    for field in conditions_fields:
        if field in row:
            conditions += _field_to_list(row.get(field))

    text_sources = _field_to_list(row.get("vacancyName"))
    text_sources += _field_to_list(row.get("professionalSphereName"))
    text_sources += requirements + responsibilities + conditions
    tokens = normalizer.normalize_tokens(" ".join(filter(None, text_sources)))

    salary_min, salary_max = _parse_salary_range(row, "salary_vacancy")
    languages_required = _parse_language_list(row.get("languageKnowledge_vacancy"))
    schedule_types = _parse_schedule(row.get("scheduleType_vacancy"))
    relocation_required = _parse_relocation(row.get("retrainingCapability_vacancy"))
    travel_required = _parse_relocation(row.get("businessTrip"))
    education_level_required = _parse_education_level(row.get("educationRequirements"))
    experience_required_years = _parse_float(row.get("experienceRequirements"))

    sections = ParsedSections(
        {
            "requirements": "\n".join(requirements),
            "responsibilities": "\n".join(responsibilities),
            "conditions": "\n".join(conditions),
        }
    )

    return JobDescriptionContent(
        path=Path(f"{row.get('idVacancy', 'vacancy')}.pdf"),
        sections=sections,
        tokens=tokens,
        requirements=requirements,
        responsibilities=responsibilities,
        conditions=conditions,
        salary_min=salary_min,
        salary_max=salary_max,
        location_code=str(row.get("stateRegionCode_vacancy") or row.get("regionName") or "").strip() or None,
        languages_required=languages_required,
        schedule_types=schedule_types,
        relocation_required=relocation_required,
        travel_required=travel_required,
        education_level_required=education_level_required,
        experience_required_years=experience_required_years,
        requirement_text="\n".join(requirements),
        responsibility_text="\n".join(responsibilities),
    )



@dataclass
class FeatureRow:
    vacancy_id: str
    cv_id: str
    relevance: int
    features: Dict[str, float]


def build_feature_table(
    frame: pd.DataFrame,
    *,
    max_vacancies: int | None = None,
    normalizer: RussianTextNormalizer | None = None,
    feature_extractor: FeatureExtractor | None = None,
    embedder: "SemanticEmbedder" | None = None,
) -> pd.DataFrame:
    """Construct a feature DataFrame ready for ranking model training."""

    normalizer = normalizer or RussianTextNormalizer()
    if feature_extractor is None:
        feature_extractor = FeatureExtractor(normalizer=normalizer, embedder=embedder)
    elif embedder is not None and getattr(feature_extractor, "embedder", None) is None:
        feature_extractor.embedder = embedder

    if "relevance" not in frame.columns:
        frame = frame.copy()
        frame["relevance"] = frame["cv_status"].map(STATUS_RELEVANCE_MAP).fillna(0).astype(int)

    groups = group_by_vacancy(frame)
    groups = sample_balanced_groups(groups, min_positive=1, min_negative=1)

    if max_vacancies is not None:
        groups = groups[:max_vacancies]

    feature_rows: List[FeatureRow] = []

    for vacancy_group in groups:
        group_frame = vacancy_group.frame
        first_row = group_frame.iloc[0]
        job_content = _extract_job_content(first_row, normalizer)

        resumes = []
        for _, row in group_frame.iterrows():
            resume = _extract_resume_content(row, normalizer)
            resumes.append((row, resume))

        resume_objects = [resume for _, resume in resumes if resume.tokens]
        if not resume_objects:
            continue

        try:
            index = BM25ResumeIndex(resume_objects)
        except ValueError:
            continue

        bm25_results = index.search(job_content, top_k=len(resumes))

        for row, resume in resumes:
            if not resume.tokens:
                continue
            cv_id = row.get("idCv")
            relevance = int(row.get("relevance", STATUS_RELEVANCE_MAP.get(row.get("cv_status"), 0)))

            result = next((res for res in bm25_results if res.resume is resume), None)
            bm25_score = result.score if result else None
            feature_vector = feature_extractor.extract(resume, job_content, bm25_score=bm25_score)
            feature_rows.append(
                FeatureRow(
                    vacancy_id=str(row.get("idVacancy")),
                    cv_id=str(cv_id),
                    relevance=relevance,
                    features=feature_vector.features,
                )
            )

    records: List[Dict[str, float | int | str]] = []
    for row in feature_rows:
        record: Dict[str, float | int | str] = {
            "idVacancy": row.vacancy_id,
            "idCv": row.cv_id,
            "relevance": row.relevance,
        }
        record.update(row.features)
        records.append(record)

    return pd.DataFrame.from_records(records)
