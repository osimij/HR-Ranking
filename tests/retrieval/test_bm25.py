from pathlib import Path

from matcher.ingestion.models import JobDescriptionContent, ResumeContent, parse_job_description
from matcher.ingestion.pdf_reader import read_pdf_text
from matcher.retrieval.bm25 import BM25ResumeIndex, build_index_from_paths


def _make_resume(tokens: list[str], path: str = "resume.pdf") -> ResumeContent:
    return ResumeContent(
        path=Path(path),
        sections=None,  # type: ignore[arg-type]
        tokens=tokens,
        skills=[],
        experiences=[],
        educations=[],
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
        skill_text=" ".join(tokens),
        experience_text="",
    )


def _make_job(tokens: list[str], path: str = "job.pdf") -> JobDescriptionContent:
    return JobDescriptionContent(
        path=Path(path),
        sections=None,  # type: ignore[arg-type]
        tokens=tokens,
        requirements=[],
        responsibilities=[],
        conditions=[],
        salary_min=None,
        salary_max=None,
        location_code=None,
        languages_required=[],
        schedule_types=[],
        relocation_required=None,
        travel_required=None,
        education_level_required=None,
        experience_required_years=None,
        requirement_text="",
        responsibility_text="",
    )


def test_bm25_returns_top_result_in_expected_order() -> None:
    resumes = [
        _make_resume(["python", "django", "api"]),
        _make_resume(["designer", "figma", "photoshop"]),
    ]
    job = _make_job(["python", "api", "backend"])
    index = BM25ResumeIndex(resumes)

    results = index.search(job, top_k=2)
    assert results[0].resume is resumes[0]
    assert results[0].score >= results[1].score


def test_bm25_requires_non_empty_input() -> None:
    job = _make_job(["python"])
    try:
        BM25ResumeIndex([])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty resume list")

    resumes = [_make_resume(["python"])]
    index = BM25ResumeIndex(resumes)
    results = index.search(job, top_k=1)
    assert len(results) == 1


def test_build_index_from_real_resume(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    resume_path = project_root / "CV" / "Osimi Jasur.pdf"
    job_path = project_root / "JD" / (
        "Вакансия Графический дизайнер контента на маркетплейсах в Санкт-Петербурге, "
        "работа в компании «Procter & Gamble», Опытный специалист.pdf"
    )

    index = build_index_from_paths([resume_path])
    job_doc = read_pdf_text(job_path)
    job = parse_job_description(job_doc)
    results = index.search(job, top_k=1)
    assert results
    from numbers import Real

    assert isinstance(results[0].score, Real)
