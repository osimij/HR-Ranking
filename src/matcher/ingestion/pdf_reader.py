"""Utilities for reading text-based PDF files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be parsed into text."""


@dataclass
class PdfDocument:
    """Simple container for a parsed PDF document."""

    path: Path
    text: str

    @property
    def lines(self) -> Iterable[str]:
        """Iterate over non-empty lines."""
        for line in self.text.splitlines():
            formatted = line.strip()
            if formatted:
                yield formatted


def read_pdf_text(path: Union[str, Path]) -> PdfDocument:
    """Read a text-based PDF and return its extracted content.

    Parameters
    ----------
    path:
        Path to the PDF file.

    Returns
    -------
    PdfDocument
        Container with the resolved path and the extracted text.
    """

    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF extraction. Install it via `pip install pypdf`."
        ) from exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pragma: no cover - library-specific errors
        raise PdfExtractionError(f"Failed to read PDF: {pdf_path}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - library-specific errors
            raise PdfExtractionError(
                f"Failed to extract text from page {index} of {pdf_path}"
            ) from exc
        pages.append(page_text)

    return PdfDocument(path=pdf_path, text="\n".join(pages))
