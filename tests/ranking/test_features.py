from pathlib import Path

from matcher.ingestion.models import (
    JobDescriptionContent,
    ResumeContent,
    parse_job_description,
    parse_resume,
)
from matcher.ingestion.pdf_reader import read_pdf_text
from matcher.ranking.features import FeatureExtractor, FeatureVector, features_from_bm25
from matcher.retrieval.bm25 import BM25ResumeIndex


def _make_resume(tokens, skills=None, experiences=None, educations=None) -> ResumeContent:
    skills = skills or []
    experiences = experiences or []
    educations = educations or []
    return ResumeContent(
        path=Path("resume.pdf"),
        sections=None,  # type: ignore[arg-type]
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
        skill_text="; ".join(skills),
        experience_text=" ".join(experiences),
    )


def _make_job(tokens, requirements=None, responsibilities=None, conditions=None) -> JobDescriptionContent:
    requirements = requirements or []
    responsibilities = responsibilities or []
    conditions = conditions or []
    return JobDescriptionContent(
        path=Path("job.pdf"),
        sections=None,  # type: ignore[arg-type]
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
        requirement_text=" ".join(requirements),
        responsibility_text=" ".join(responsibilities),
    )


def test_feature_extractor_computes_skill_overlap() -> None:
    resume = _make_resume(
        ["python", "django"],
        skills=["Python", "Django"],
        experiences=["Worked on Python backend"],
    )
    job = _make_job(
        ["python", "backend", "developer"],
        requirements=["Python developer with Django experience"],
    )
    extractor = FeatureExtractor()
    feature_vector = extractor.extract(resume, job, bm25_score=2.5)
    assert isinstance(feature_vector, FeatureVector)
    assert feature_vector["skill_overlap"] > 0
    assert feature_vector["skill_precision"] > 0
    assert feature_vector["requirements_coverage"] > 0
    assert feature_vector["bm25_score"] == 2.5


def test_features_from_real_documents() -> None:
    project_root = Path(__file__).resolve().parents[2]
    resume_doc = read_pdf_text(project_root / "CV" / "Osimi Jasur.pdf")
    job_doc = read_pdf_text(
        project_root
        / "JD"
        / (
            "Вакансия Графический дизайнер контента на маркетплейсах в Санкт-Петербурге, "
            "работа в компании «Procter & Gamble», Опытный специалист.pdf"
        )
    )

    resume = parse_resume(resume_doc)
    job = parse_job_description(job_doc)

    index = BM25ResumeIndex([resume])
    result = index.search(job, top_k=1)[0]
    feature_vector = features_from_bm25(result, job)
    assert feature_vector["resume_token_len"] > 0
    assert "bm25_score" in feature_vector.features
