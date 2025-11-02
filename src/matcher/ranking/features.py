"""Feature extraction utilities for resume-job pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from matcher.embeddings.semantic import SemanticEmbedder
from matcher.ingestion.models import JobDescriptionContent, ResumeContent
from matcher.normalization.text import RussianTextNormalizer
from matcher.retrieval.bm25 import BM25Result


def _normalize_items(items: Iterable[str], normalizer: RussianTextNormalizer) -> List[List[str]]:
    normalized_items: List[List[str]] = []
    for item in items:
        tokens = normalizer.normalize_tokens(item)
        if tokens:
            normalized_items.append(tokens)
    return normalized_items


def _flatten(items: Iterable[Iterable[str]]) -> List[str]:
    flattened: List[str] = []
    for sequence in items:
        flattened.extend(sequence)
    return flattened


def _compute_overlap(resume_tokens: List[str], job_tokens: List[str]) -> int:
    return len(set(resume_tokens) & set(job_tokens))


def _match_ratio(candidate: Iterable[str], required: Iterable[str]) -> float:
    candidate_set = {item.split(':')[0] for item in candidate if item}
    required_set = {item.split(':')[0] for item in required if item}
    if not required_set:
        return 0.0
    if not candidate_set:
        return 0.0
    intersection = candidate_set & required_set
    return len(intersection) / len(required_set)


def _schedule_match(candidate: Iterable[str], required: Iterable[str]) -> float:
    if not required:
        return 0.0
    candidate_norm = {item.strip().lower() for item in candidate if item}
    required_norm = {item.strip().lower() for item in required if item}
    if not candidate_norm:
        return 0.0
    return len(candidate_norm & required_norm) / len(required_norm)


def _boolean_alignment(candidate: Optional[str], required: Optional[str]) -> float:
    if required is None:
        return 0.0
    required_text = required.lower()
    candidate_text = (candidate or "").lower()
    if "не" in required_text and not required_text.strip().startswith("готов"):
        # requirement says no relocation/business trip needed
        return 1.0
    if "готов" in required_text:
        return 1.0 if "готов" in candidate_text and "не" not in candidate_text else 0.0
    if required_text:
        return 1.0 if candidate_text and required_text in candidate_text else 0.0
    return 0.0


def _requirements_coverage(
    resume_tokens: List[str],
    normalized_requirements: List[List[str]],
) -> float:
    if not normalized_requirements:
        return 0.0
    resume_token_set = set(resume_tokens)
    matched = 0
    for requirement_tokens in normalized_requirements:
        if resume_token_set.intersection(requirement_tokens):
            matched += 1
    return matched / len(normalized_requirements)


@dataclass
class FeatureVector:
    features: Dict[str, float]

    def __getitem__(self, item: str) -> float:
        return self.features[item]


class FeatureExtractor:
    """Derive numeric features from resume/job content."""

    def __init__(
        self,
        normalizer: RussianTextNormalizer | None = None,
        embedder: SemanticEmbedder | None = None,
    ) -> None:
        self.normalizer = normalizer or RussianTextNormalizer()
        self.embedder = embedder

    def _semantic_similarity(self, text_a: str, text_b: str) -> float:
        if not self.embedder:
            return 0.0
        if not text_a or not text_b:
            return 0.0
        vec_a = self.embedder.encode(text_a)
        vec_b = self.embedder.encode(text_b)
        return self.embedder.cosine_similarity(vec_a, vec_b)

    def extract(
        self,
        resume: ResumeContent,
        job: JobDescriptionContent,
        *,
        bm25_score: float | None = None,
    ) -> FeatureVector:
        resume_skills = _normalize_items(resume.skills, self.normalizer)
        job_skills = _normalize_items(job.requirements + job.responsibilities + job.conditions, self.normalizer)

        resume_skill_tokens = _flatten(resume_skills)
        job_skill_tokens = _flatten(job_skills)

        overlap = _compute_overlap(resume_skill_tokens, job_skill_tokens)
        resume_skill_count = len(resume_skill_tokens) or 1
        job_skill_count = len(job_skill_tokens) or 1

        skill_precision = overlap / resume_skill_count
        skill_recall = overlap / job_skill_count
        skill_f1 = (
            2 * skill_precision * skill_recall / (skill_precision + skill_recall)
            if skill_precision + skill_recall > 0
            else 0.0
        )

        resume_experiences = _normalize_items(resume.experiences, self.normalizer)
        job_requirements = _normalize_items(job.requirements, self.normalizer)

        coverage = _requirements_coverage(resume.tokens, job_requirements)

        resume_salary_min = resume.salary_min if resume.salary_min is not None else 0.0
        resume_salary_max = resume.salary_max if resume.salary_max is not None else resume_salary_min
        job_salary_min = job.salary_min if job.salary_min is not None else 0.0
        job_salary_max = job.salary_max if job.salary_max is not None else job_salary_min
        salary_gap_min = (
            (resume.salary_min or 0.0) - (job.salary_min or 0.0)
            if resume.salary_min is not None and job.salary_min is not None
            else 0.0
        )
        salary_gap_max = (
            (resume.salary_max or 0.0) - (job.salary_max or 0.0)
            if resume.salary_max is not None and job.salary_max is not None
            else 0.0
        )
        salary_ratio = 0.0
        if job_salary_max:
            salary_ratio = (resume.salary_min or resume_salary_max) / job_salary_max

        language_match_ratio = _match_ratio(resume.languages, job.languages_required)
        schedule_match_ratio = _schedule_match(resume.schedule_types, job.schedule_types)

        education_alignment = 0.0
        if job.education_level_required is not None:
            if resume.education_level is not None:
                education_alignment = 1.0 if resume.education_level >= job.education_level_required else 0.0
            else:
                education_alignment = 0.0

        experience_years = resume.experience_years or 0.0
        experience_vs_requirement = 0.0
        if resume.experience_years is not None and job.experience_required_years is not None:
            experience_vs_requirement = resume.experience_years - job.experience_required_years

        relocation_alignment = _boolean_alignment(resume.relocation_status, job.relocation_required)
        travel_alignment = _boolean_alignment(resume.travel_status, job.travel_required)

        location_match = 0.0
        if resume.location_code and job.location_code:
            location_match = 1.0 if resume.location_code == job.location_code else 0.0
        elif resume.locality and job.location_code:
            location_match = 1.0 if resume.locality.lower() in job.location_code.lower() else 0.0

        semantic_skill_similarity = self._semantic_similarity(resume.skill_text, job.requirement_text)
        semantic_experience_similarity = self._semantic_similarity(
            resume.experience_text or resume.skill_text,
            job.responsibility_text or job.requirement_text,
        )

        features: Dict[str, float] = {
            "resume_token_len": float(len(resume.tokens)),
            "job_token_len": float(len(job.tokens)),
            "skill_overlap": float(overlap),
            "skill_precision": float(skill_precision),
            "skill_recall": float(skill_recall),
            "skill_f1": float(skill_f1),
            "resume_experience_count": float(len(resume_experiences)),
            "resume_education_count": float(len(resume.educations)),
            "requirements_coverage": float(coverage),
            "salary_gap_min": float(salary_gap_min),
            "salary_gap_max": float(salary_gap_max),
            "salary_ratio": float(salary_ratio),
            "salary_candidate_min": float(resume_salary_min),
            "salary_candidate_max": float(resume_salary_max),
            "salary_job_min": float(job_salary_min),
            "salary_job_max": float(job_salary_max),
            "language_match_ratio": float(language_match_ratio),
            "schedule_match_ratio": float(schedule_match_ratio),
            "education_alignment": float(education_alignment),
            "experience_years": float(experience_years),
            "experience_vs_requirement": float(experience_vs_requirement),
            "experience_gap_years": float(resume.last_experience_years or 0.0),
            "relocation_alignment": float(relocation_alignment),
            "travel_alignment": float(travel_alignment),
            "location_match": float(location_match),
            "semantic_skill_similarity": float(semantic_skill_similarity),
            "semantic_experience_similarity": float(semantic_experience_similarity),
        }
        if bm25_score is not None:
            features["bm25_score"] = float(bm25_score)
        return FeatureVector(features)


def features_from_bm25(result: BM25Result, job: JobDescriptionContent, extractor: FeatureExtractor | None = None) -> FeatureVector:
    extractor = extractor or FeatureExtractor()
    return extractor.extract(result.resume, job, bm25_score=result.score)

