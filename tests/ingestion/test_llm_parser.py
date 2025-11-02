from pathlib import Path

import pytest

from matcher.ingestion.llm_parser import LLMJobParser, LLMResumeParser
from matcher.ingestion.pdf_reader import PdfDocument


def test_llm_resume_parser_converts_json():
    response_payload = """
    {
      "candidate_info": {
        "name": "Osimi Jasur",
        "gender": "Мужчина",
        "age": "19 лет",
        "birth_date": "1 февраля 2006",
        "phone": "+7 (996) 9530336",
        "email": "osimijasur@gmail.com",
        "residence": "Таджикистан",
        "citizenship": "Таджикистан",
        "work_permit": "Россия, Таджикистан",
        "relocation_readiness": "Не готов к переезду",
        "business_trip_readiness": "не готов к командировкам"
      },
      "desired_position": {
        "title": "Копирайтер",
        "salary": null,
        "specializations": [
          "Копирайтер, редактор, корректор"
        ],
        "employment_types": [
          "полная занятость",
          "частичная занятость",
          "проектная работа"
        ],
        "schedule_types": [
          "удаленная работа"
        ],
        "desired_commute_time": "не имеет значения"
      },
      "work_experience_total": "2 года 3 месяца",
      "work_experience": [
        {
          "period": "Апрель 2023 — Октябрь 2023 (7 месяцев)",
          "company": "EPAM Systems",
          "location": "США",
          "url": "www.epam.com/",
          "industry": "Информационные технологии",
          "responsibilities": [
            "Разработка программного обеспечения"
          ],
          "position": "Research Intern",
          "details": [
            "Занимался исследованиями в сфере цифровых технологий."
          ]
        }
      ],
      "education": {
        "degree": "Бакалавр",
        "year": 2027,
        "institution": "Российско-Таджикский университет",
        "major": "Журналистика"
      },
      "languages": [
        {
          "language": "Русский",
          "level": "Родной"
        },
        {
          "language": "Английский",
          "level": "C1 Продвинутый"
        }
      ],
      "skills": [
        "Копирайтинг",
        "SEO-копирайтинг",
        "SMM"
      ]
    }
    """.strip()

    document = PdfDocument(path=Path("resume.pdf"), text="Dummy resume text")
    parser = LLMResumeParser(completion=lambda _: response_payload)

    resume = parser.parse(document)

    assert resume.skills == ["Копирайтинг", "SEO-копирайтинг", "SMM"]
    assert resume.education_level == 4
    assert resume.experience_years == pytest.approx(2.25, rel=1e-3)
    assert resume.languages == ["русский:родной", "английский:c1 продвинутый"]
    assert resume.schedule_types == ["удаленная работа"]
    assert resume.location_code == "Таджикистан"


def test_llm_job_parser_converts_json():
    response_payload = """
    {
      "job_info": {
        "title": "Senior Python Developer",
        "company": "Acme Corp",
        "location": "Москва",
        "salary": {
          "min": 150000,
          "max": 220000,
          "currency": "RUB"
        },
        "schedule_types": [
          "Полный день",
          "Удаленная работа"
        ],
        "relocation_required": "не требуется",
        "travel_required": "редкие командировки",
        "education_level": "Высшее образование",
        "experience_required_years": "3 года"
      },
      "requirements": ["Python", "Django", "PostgreSQL"],
      "responsibilities": ["Разрабатывать сервисы", "Проводить code review"],
      "conditions": ["Оформление по ТК", "ДМС"],
      "languages_required": [
        {
          "language": "Английский",
          "level": "B2"
        }
      ]
    }
    """.strip()

    document = PdfDocument(path=Path("job.pdf"), text="Dummy job description")
    parser = LLMJobParser(completion=lambda _: f"```json\n{response_payload}\n```")

    job = parser.parse(document)

    assert job.salary_min == 150000
    assert job.salary_max == 220000
    assert job.schedule_types == ["полный день", "удаленная работа"]
    assert job.languages_required == ["английский:b2"]
    assert job.experience_required_years == pytest.approx(3.0, rel=1e-3)
    assert job.relocation_required == "не требуется"
