"""Utilities for loading and splitting the Kaggle vacancy-resume dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


CSV_DELIMITER = "|"
STATUS_RELEVANCE_MAP = {
    "Приглашение": 1,
    "Отказ": 0,
}

DEFAULT_COLUMNS: Sequence[str] = (
    "idCv",
    "idVacancy",
    "cv_status",
    "positionName",
    "vacancyName",
    "professionalSphereName",
    "skills_cv",
    "hardSkills_cv",
    "softSkills_cv",
    "experience",
    "workExperienceList",
    "educationList",
    "languageKnowledge_cv",
    "scheduleType_cv",
    "relocation",
    "businessTrip",
    "education",
    "salary_cv",
    "salaryMin_cv",
    "salaryMax_cv",
    "locality",
    "localityName",
    "skills_vacancy",
    "hardSkills_vacancy",
    "softSkills_vacancy",
    "positionRequirements",
    "experienceRequirements",
    "educationRequirements",
    "qualifications",
    "responsibilities",
    "conditions",
    "otherVacancyBenefit",
    "careerPerspective",
    "benefit",
    "languageKnowledge_vacancy",
    "scheduleType_vacancy",
    "retrainingCapability_vacancy",
    "salary_vacancy",
    "salaryMin_vacancy",
    "salaryMax_vacancy",
    "stateRegionCode_vacancy",
    "regionName",
)


@dataclass(frozen=True)
class VacancyGroup:
    vacancy_id: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_applications(
    csv_path: str | Path,
    *,
    columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load relevant columns from the Kaggle dataset and map relevance labels."""

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    usecols = list(columns or DEFAULT_COLUMNS)
    frame = pd.read_csv(
        path,
        sep=CSV_DELIMITER,
        usecols=usecols,
        nrows=limit,
        dtype=str,
        keep_default_na=False,
    )

    frame["relevance"] = frame["cv_status"].map(STATUS_RELEVANCE_MAP)
    frame = frame.dropna(subset=["relevance"]).copy()
    frame["relevance"] = frame["relevance"].astype(int)
    return frame


def group_by_vacancy(frame: pd.DataFrame) -> List[VacancyGroup]:
    """Group a DataFrame by vacancy ID into VacancyGroup objects."""
    if "idVacancy" not in frame.columns:
        raise KeyError("Column 'idVacancy' is required for grouping.")
    groups = []
    for vacancy_id, group in frame.groupby("idVacancy", sort=False):
        groups.append(VacancyGroup(vacancy_id=vacancy_id, frame=group))
    return groups


def split_dataset_by_vacancy(
    frame: pd.DataFrame,
    *,
    validation_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> DatasetSplit:
    """Perform a group-aware split on vacancy IDs."""

    if not 0 < validation_frac < 1:
        raise ValueError("validation_frac must be between 0 and 1.")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1.")
    if validation_frac + test_frac >= 1:
        raise ValueError("validation_frac + test_frac must be < 1.")

    vacancies = frame["idVacancy"].unique()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(vacancies)

    n = len(shuffled)
    val_cut = int(n * validation_frac)
    test_cut = val_cut + int(n * test_frac)

    val_ids = set(shuffled[:val_cut])
    test_ids = set(shuffled[val_cut:test_cut])

    validation = frame[frame["idVacancy"].isin(val_ids)].copy()
    test = frame[frame["idVacancy"].isin(test_ids)].copy()
    train = frame[~frame["idVacancy"].isin(val_ids | test_ids)].copy()

    return DatasetSplit(train=train, validation=validation, test=test)


def sample_balanced_groups(
    groups: Iterable[VacancyGroup],
    *,
    min_positive: int = 1,
    min_negative: int = 1,
) -> List[VacancyGroup]:
    """Filter groups ensuring minimum positive/negative counts."""

    filtered: List[VacancyGroup] = []
    for group in groups:
        positives = (group.frame["relevance"] > 0).sum()
        negatives = (group.frame["relevance"] == 0).sum()
        if positives >= min_positive and negatives >= min_negative:
            filtered.append(group)
    return filtered
