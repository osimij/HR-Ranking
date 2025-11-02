"""LLM-assisted parsing of resumes and job descriptions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from matcher.ingestion.gemini_client import GeminiClientError, generate_json_from_pdf
from matcher.ingestion.models import JobDescriptionContent, ResumeContent
from matcher.ingestion.pdf_reader import PdfDocument
from matcher.ingestion.section_parser import ParsedSections
from matcher.normalization.text import RussianTextNormalizer

LLMCompletionFn = Callable[[str], str]


class LLMExtractionError(RuntimeError):
    """Raised when an LLM response cannot be parsed into the expected schema."""


def _strip_code_fence(payload: str) -> str:
    text = payload.strip()
    if text.startswith("```"):
        fence_end = text.find("\n")
        text = text[fence_end + 1 :] if fence_end != -1 else ""
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise LLMExtractionError("LLM response did not return a JSON object.")


def _to_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else []
    items: List[str] = []
    if isinstance(value, Iterable):
        for entry in value:
            if isinstance(entry, str):
                candidate = entry.strip()
                if candidate:
                    items.append(candidate)
            elif isinstance(entry, Mapping):
                candidate = str(entry.get("text", "")).strip()
                if candidate:
                    items.append(candidate)
    return items


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("\xa0", " ").strip()
    if not text:
        return None
    numbers = re.findall(r"[\d]+(?:[.,]\d+)?", text)
    if not numbers:
        return None
    number = numbers[0].replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return None


def _parse_salary_range(value: Any) -> tuple[Optional[float], Optional[float]]:
    if isinstance(value, Mapping):
        return _parse_float(value.get("min")), _parse_float(value.get("max"))
    candidate = _parse_float(value)
    return candidate, candidate


def _parse_total_experience(text: Any) -> Optional[float]:
    if not text:
        return None
    candidate = str(text).lower()
    years = 0.0
    matched = False
    year_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:год|лет|года)", candidate)
    if year_match:
        years += float(year_match.group(1).replace(",", "."))
        matched = True
    month_match = re.search(r"(\d+(?:[.,]\d+)?)\s*месяц", candidate)
    if month_match:
        years += float(month_match.group(1).replace(",", ".")) / 12.0
        matched = True
    return years if matched else None


def _education_level_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    lookup = {
        "дошколь": 0,
        "началь": 1,
        "среднее": 2,
        "среднее профессион": 3,
        "неполное высш": 3,
        "бакалавр": 4,
        "специалист": 5,
        "магистр": 6,
        "магистрат": 6,
        "аспирант": 7,
        "phd": 7,
        "кандидат наук": 7,
    }
    text = str(value).lower()
    for key, level in lookup.items():
        if key in text:
            return level
    return None


def _format_experience_entry(entry: Mapping[str, Any]) -> str:
    parts: List[str] = []
    position = str(entry.get("position", "")).strip()
    company = str(entry.get("company", "")).strip()
    if position and company:
        parts.append(f"{position} @ {company}")
    elif position:
        parts.append(position)
    elif company:
        parts.append(company)
    period = str(entry.get("period", "")).strip()
    if period:
        parts.append(period)
    location = str(entry.get("location", "")).strip()
    if location:
        parts.append(location)
    responsibilities = entry.get("responsibilities")
    details = entry.get("details")
    detail_lines = []
    if isinstance(responsibilities, Iterable) and not isinstance(responsibilities, (str, bytes)):
        for item in responsibilities:
            candidate = str(item).strip()
            if candidate:
                detail_lines.append(candidate)
    if isinstance(details, Iterable) and not isinstance(details, (str, bytes)):
        for item in details:
            candidate = str(item).strip()
            if candidate:
                detail_lines.append(candidate)
    if detail_lines:
        parts.append("; ".join(detail_lines))
    return " — ".join(parts)


def _extract_languages(items: Any) -> List[str]:
    if items is None:
        return []
    normalized: List[str] = []
    if isinstance(items, Iterable) and not isinstance(items, (str, bytes)):
        for entry in items:
            if isinstance(entry, Mapping):
                language = str(entry.get("language", "")).strip().lower()
                level = str(entry.get("level", "")).strip().lower()
                if language and level:
                    normalized.append(f"{language}:{level}")
                elif language:
                    normalized.append(language)
            elif isinstance(entry, str):
                candidate = entry.strip().lower()
                if candidate:
                    normalized.append(candidate)
    elif isinstance(items, str):
        normalized.append(items.strip().lower())
    return normalized


def _extract_schedule(items: Any) -> List[str]:
    values = _to_string_list(items)
    return [value.lower() for value in values]


def _relocation_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text.lower() or None


def _flatten_to_strings(value: Any) -> List[str]:
    results: List[str] = []
    if value is None:
        return results
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
        if text:
            results.append(text)
        return results
    if isinstance(value, Mapping):
        for item in value.values():
            results.extend(_flatten_to_strings(item))
        return results
    if isinstance(value, Iterable):
        for item in value:
            results.extend(_flatten_to_strings(item))
        return results
    text = str(value).strip()
    if text:
        results.append(text)
    return results


@dataclass
class LLMResumeParser:
    completion: LLMCompletionFn
    normalizer: RussianTextNormalizer = field(default_factory=RussianTextNormalizer)
    prompt_template: str = field(default_factory=lambda: _DEFAULT_RESUME_PROMPT)

    def build_prompt(self, document: PdfDocument) -> str:
        return self.prompt_template.format(resume_text=document.text.strip())

    def parse(self, document: PdfDocument) -> ResumeContent:
        prompt = self.build_prompt(document)
        response = self.completion(prompt)
        payload = _strip_code_fence(response)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMExtractionError("LLM response was not valid JSON.") from exc
        resume_json = _ensure_dict(data)
        return self._resume_from_json(document, resume_json)

    def parse_pdf_bytes(self, pdf_bytes: bytes, *, source_path: Path | None = None) -> ResumeContent:
        prompt = self.prompt_template.format(resume_text="[ATTACHED_DOCUMENT]")
        try:
            response = generate_json_from_pdf(pdf_bytes, prompt)
        except GeminiClientError as exc:
            raise LLMExtractionError(f"Gemini request failed: {exc}") from exc
        payload = _strip_code_fence(response)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMExtractionError("Gemini response was not valid JSON.") from exc
        resume_json = _ensure_dict(data)
        synthetic_text = self._resume_text_from_json(resume_json)
        document = PdfDocument(path=source_path or Path("resume.pdf"), text=synthetic_text)
        return self._resume_from_json(document, resume_json)

    def _resume_from_json(self, document: PdfDocument, data: Dict[str, Any]) -> ResumeContent:
        candidate = _ensure_dict(data.get("candidate_info", {}))
        desired = _ensure_dict(data.get("desired_position", {})) if isinstance(data.get("desired_position"), Mapping) else {}
        experiences_raw = data.get("work_experience") or []
        if not isinstance(experiences_raw, Iterable) or isinstance(experiences_raw, (str, bytes)):
            experiences_raw = []
        experience_sections: List[str] = []
        for entry in experiences_raw:
            if isinstance(entry, Mapping):
                formatted = _format_experience_entry(entry)
                if formatted:
                    experience_sections.append(formatted)
        education_entries: List[str] = []
        education = data.get("education")
        if isinstance(education, Mapping):
            institution = str(education.get("institution", "")).strip()
            degree = str(education.get("degree", "")).strip()
            year = str(education.get("year", "")).strip()
            payload = " ".join(part for part in [degree, institution, year] if part)
            if payload:
                education_entries.append(payload)
        elif isinstance(education, Iterable):
            for item in education:
                if isinstance(item, str):
                    candidate_text = item.strip()
                    if candidate_text:
                        education_entries.append(candidate_text)
        salary_min, salary_max = _parse_salary_range(desired.get("salary"))
        experience_years = _parse_total_experience(data.get("work_experience_total"))
        skills = [skill.strip() for skill in data.get("skills", []) if isinstance(skill, str) and skill.strip()]
        languages = _extract_languages(data.get("languages"))
        schedule = _extract_schedule(desired.get("schedule_types"))

        tokens = self.normalizer.normalize_tokens(document.text)
        skill_text = "\n".join(skills)
        experience_text = "\n".join(experience_sections)

        return ResumeContent(
            path=document.path,
            sections=ParsedSections({}),
            tokens=tokens,
            skills=skills,
            experiences=experience_sections,
            educations=education_entries,
            salary_min=salary_min,
            salary_max=salary_max,
            location_code=str(candidate.get("residence") or "").strip() or None,
            locality=str(candidate.get("residence") or "").strip() or None,
            languages=languages,
            schedule_types=schedule,
            relocation_status=_relocation_text(candidate.get("relocation_readiness")),
            travel_status=_relocation_text(candidate.get("business_trip_readiness")),
            education_level=_education_level_to_int(education.get("degree") if isinstance(education, Mapping) else None),
            experience_years=experience_years,
            last_experience_years=None,
            skill_text=skill_text,
            experience_text=experience_text,
        )

    def _resume_text_from_json(self, data: Dict[str, Any]) -> str:
        parts = _flatten_to_strings(data)
        return "\n".join(part for part in parts if part)


@dataclass
class LLMJobParser:
    completion: LLMCompletionFn
    normalizer: RussianTextNormalizer = field(default_factory=RussianTextNormalizer)
    prompt_template: str = field(default_factory=lambda: _DEFAULT_JOB_PROMPT)

    def build_prompt(self, document: PdfDocument) -> str:
        return self.prompt_template.format(job_text=document.text.strip())

    def parse(self, document: PdfDocument) -> JobDescriptionContent:
        prompt = self.build_prompt(document)
        response = self.completion(prompt)
        payload = _strip_code_fence(response)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMExtractionError("LLM response was not valid JSON.") from exc
        job_json = _ensure_dict(data)
        return self._job_from_json(document, job_json)

    def parse_pdf_bytes(self, pdf_bytes: bytes, *, source_path: Path | None = None) -> JobDescriptionContent:
        prompt = self.prompt_template.format(job_text="[ATTACHED_DOCUMENT]")
        try:
            response = generate_json_from_pdf(pdf_bytes, prompt)
        except GeminiClientError as exc:
            raise LLMExtractionError(f"Gemini request failed: {exc}") from exc
        payload = _strip_code_fence(response)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMExtractionError("Gemini response was not valid JSON.") from exc
        job_json = _ensure_dict(data)
        synthetic_text = self._job_text_from_json(job_json)
        document = PdfDocument(path=source_path or Path("job.pdf"), text=synthetic_text)
        return self._job_from_json(document, job_json)

    def _job_from_json(self, document: PdfDocument, data: Dict[str, Any]) -> JobDescriptionContent:
        job_info = _ensure_dict(data.get("job_info", {}))
        salary_min, salary_max = _parse_salary_range(job_info.get("salary"))
        requirements = _to_string_list(data.get("requirements"))
        responsibilities = _to_string_list(data.get("responsibilities"))
        conditions = _to_string_list(data.get("conditions"))
        languages = _extract_languages(data.get("languages_required") or data.get("languages"))
        schedule = _extract_schedule(job_info.get("schedule_types") or data.get("schedule_types"))
        relocation = _relocation_text(job_info.get("relocation_required") or data.get("relocation_required"))
        travel = _relocation_text(job_info.get("travel_required") or data.get("travel_required"))
        education_level = _education_level_to_int(job_info.get("education_level") or data.get("education_level"))
        experience_years = _parse_total_experience(job_info.get("experience_required_years") or data.get("experience_required_years"))

        tokens = self.normalizer.normalize_tokens(document.text)
        requirement_text = "\n".join(requirements)
        responsibility_text = "\n".join(responsibilities)

        return JobDescriptionContent(
            path=document.path,
            sections=ParsedSections({}),
            tokens=tokens,
            requirements=requirements,
            responsibilities=responsibilities,
            conditions=conditions,
            salary_min=salary_min,
            salary_max=salary_max,
            location_code=str(job_info.get("location") or "").strip() or None,
            languages_required=languages,
            schedule_types=schedule,
            relocation_required=relocation,
            travel_required=travel,
            education_level_required=education_level,
            experience_required_years=experience_years,
            requirement_text=requirement_text,
            responsibility_text=responsibility_text,
        )

    def _job_text_from_json(self, data: Dict[str, Any]) -> str:
        parts = _flatten_to_strings(data)
        return "\n".join(part for part in parts if part)


_DEFAULT_RESUME_PROMPT = """You are a helpful assistant that extracts structured data from resumes.
Read the following resume text and return a single JSON object with the fields shown below.
Output only valid JSON, without additional commentary or Markdown fences.

Expected JSON structure:
{{
  "candidate_info": {{
    "name": string | null,
    "gender": string | null,
    "age": string | null,
    "birth_date": string | null,
    "phone": string | null,
    "email": string | null,
    "residence": string | null,
    "citizenship": string | null,
    "work_permit": string | null,
    "relocation_readiness": string | null,
    "business_trip_readiness": string | null
  }},
  "desired_position": {{
    "title": string | null,
    "salary": string | number | null,
    "specializations": [string],
    "employment_types": [string],
    "schedule_types": [string],
    "desired_commute_time": string | null
  }},
  "work_experience_total": string | null,
  "work_experience": [
    {{
      "period": string | null,
      "company": string | null,
      "location": string | null,
      "url": string | null,
      "industry": string | null,
      "position": string | null,
      "responsibilities": [string] | null,
      "details": [string] | null
    }}
  ],
  "education": object | [string] | null,
  "languages": [object|string] | null,
  "skills": [string] | null
}}

Resume text:
{resume_text}
"""


_DEFAULT_JOB_PROMPT = """You are a helpful assistant that extracts structured data from job descriptions.
Read the job description text and return a single JSON object with the fields shown below.
Output only valid JSON, without extra commentary or Markdown fences.

Expected JSON structure:
{{
  "job_info": {{
    "title": string | null,
    "company": string | null,
    "location": string | null,
    "salary": string | number | object | null,
    "employment_types": [string] | null,
    "schedule_types": [string] | null,
    "relocation_required": string | null,
    "travel_required": string | null,
    "education_level": string | null,
    "experience_required_years": string | number | null
  }},
  "requirements": [string] | null,
  "responsibilities": [string] | null,
  "conditions": [string] | null,
  "languages_required": [object|string] | null
}}

Job description text:
{job_text}
"""


def create_gemini_resume_parser() -> LLMResumeParser:
    """Instantiate an LLMResumeParser wired to the Gemini API."""
    from matcher.ingestion.gemini_client import gemini_completion

    return LLMResumeParser(completion=gemini_completion)


def create_gemini_job_parser() -> LLMJobParser:
    """Instantiate an LLMJobParser wired to the Gemini API."""
    from matcher.ingestion.gemini_client import gemini_completion

    return LLMJobParser(completion=gemini_completion)


__all__ = [
    "LLMResumeParser",
    "LLMJobParser",
    "LLMExtractionError",
    "create_gemini_resume_parser",
    "create_gemini_job_parser",
]
