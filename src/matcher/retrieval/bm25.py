"""BM25 retrieval over normalized resume tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from matcher.ingestion.models import JobDescriptionContent, ResumeContent, parse_resume
from matcher.ingestion.pdf_reader import read_pdf_text
from matcher.normalization.text import RussianTextNormalizer


@dataclass
class BM25Result:
    resume: ResumeContent
    score: float


class BM25ResumeIndex:
    """Simple BM25 index that operates on resume tokens."""

    def __init__(self, resumes: Sequence[ResumeContent]) -> None:
        if not resumes:
            raise ValueError("At least one resume is required to build the index.")
        self._resumes = list(resumes)
        self._corpus = [resume.tokens for resume in self._resumes]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, job: JobDescriptionContent, *, top_k: int = 10) -> List[BM25Result]:
        """Return top_k resumes ranked by BM25 score for the given job description."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        scores = self._bm25.get_scores(job.tokens)
        paired: List[Tuple[ResumeContent, float]] = list(zip(self._resumes, scores))
        paired.sort(key=lambda item: item[1], reverse=True)
        return [BM25Result(resume=resume, score=score) for resume, score in paired[:top_k]]

    @property
    def resumes(self) -> Sequence[ResumeContent]:
        return self._resumes


def build_index_from_paths(
    resume_paths: Iterable[str | Path],
    *,
    normalizer: RussianTextNormalizer | None = None,
) -> BM25ResumeIndex:
    """Build a BM25ResumeIndex from resume PDF paths."""

    parsed_resumes: List[ResumeContent] = []
    for path in resume_paths:
        document = read_pdf_text(path)
        parsed_resumes.append(parse_resume(document, normalizer=normalizer))
    return BM25ResumeIndex(parsed_resumes)
