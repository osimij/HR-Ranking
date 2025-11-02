from pathlib import Path

import pytest

from matcher.ingestion.pdf_reader import PdfDocument, read_pdf_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CV_PATH = PROJECT_ROOT / "CV" / "Osimi Jasur.pdf"
JD_PATH = PROJECT_ROOT / "JD" / (
    "Вакансия Графический дизайнер контента на маркетплейсах в Санкт-Петербурге, "
    "работа в компании «Procter & Gamble», Опытный специалист.pdf"
)


@pytest.mark.parametrize("path,expected_substring", [(CV_PATH, "Osimi Jasur"), (JD_PATH, "P&G")])
def test_read_pdf_text_contains_expected_content(path: Path, expected_substring: str) -> None:
    document = read_pdf_text(path)
    assert isinstance(document, PdfDocument)
    assert expected_substring in document.text


def test_lines_property_filters_empty_rows() -> None:
    document = read_pdf_text(CV_PATH)
    lines = list(document.lines)
    assert all(line.strip() for line in lines)
    assert any("Osimi Jasur" in line for line in lines)

