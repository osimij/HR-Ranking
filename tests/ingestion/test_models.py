from pathlib import Path

from matcher.ingestion.models import parse_job_description, parse_resume
from matcher.ingestion.pdf_reader import read_pdf_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CV_PATH = PROJECT_ROOT / "CV" / "Osimi Jasur.pdf"
JD_PATH = PROJECT_ROOT / "JD" / (
    "Вакансия Графический дизайнер контента на маркетплейсах в Санкт-Петербурге, "
    "работа в компании «Procter & Gamble», Опытный специалист.pdf"
)


def test_parse_resume_extracts_skills_and_experience() -> None:
    resume_doc = read_pdf_text(CV_PATH)
    resume = parse_resume(resume_doc)
    assert resume.skills, "Expected skills to be populated"
    assert any("EPAM" in item or "EPAM Systems" in item for item in resume.experiences)
    assert resume.tokens, "Normalized tokens should not be empty"


def test_parse_job_description_sections_present() -> None:
    job_doc = read_pdf_text(JD_PATH)
    job = parse_job_description(job_doc)
    assert job.tokens, "Normalized tokens should not be empty"
    assert job.requirements or job.conditions, "At least one section should populate"
