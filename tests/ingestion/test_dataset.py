import textwrap

import pandas as pd
import pytest

from matcher.ingestion.dataset import (
    DatasetSplit,
    load_applications,
    group_by_vacancy,
    sample_balanced_groups,
    split_dataset_by_vacancy,
)


CSV_SAMPLE = textwrap.dedent(
    """\
    idCv|idVacancy|cv_status|positionName|vacancyName
    cv1|vac1|Приглашение|Python Dev|Python Dev
    cv2|vac1|Отказ|Python Dev|Python Dev
    cv3|vac2|Отказ|Data Analyst|Data Analyst
    cv4|vac2|Приглашение|Data Analyst|Data Analyst
    cv5|vac3|Отказ|Designer|Designer
    """
)


def test_load_applications_maps_relevance(tmp_path) -> None:
    dataset_path = tmp_path / "sample.csv"
    dataset_path.write_text(CSV_SAMPLE, encoding="utf-8")

    frame = load_applications(dataset_path, columns=["idCv", "idVacancy", "cv_status"])
    assert "relevance" in frame.columns
    assert frame.loc[frame["idCv"] == "cv1", "relevance"].iloc[0] == 1
    assert frame.loc[frame["idCv"] == "cv2", "relevance"].iloc[0] == 0


def test_group_by_vacancy_returns_groups(tmp_path) -> None:
    dataset_path = tmp_path / "sample.csv"
    dataset_path.write_text(CSV_SAMPLE, encoding="utf-8")
    frame = load_applications(dataset_path, columns=["idCv", "idVacancy", "cv_status"])

    groups = group_by_vacancy(frame)
    assert len(groups) == 3
    assert groups[0].frame["idVacancy"].iloc[0] == "vac1"


def test_split_dataset_by_vacancy_produces_disjoint_sets(tmp_path) -> None:
    dataset_path = tmp_path / "sample.csv"
    dataset_path.write_text(CSV_SAMPLE, encoding="utf-8")
    frame = load_applications(dataset_path, columns=["idCv", "idVacancy", "cv_status"])

    split = split_dataset_by_vacancy(frame, validation_frac=0.2, test_frac=0.2, random_state=1)
    assert isinstance(split, DatasetSplit)
    train_ids = set(split.train["idVacancy"])
    val_ids = set(split.validation["idVacancy"])
    test_ids = set(split.test["idVacancy"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_sample_balanced_groups_filters_insufficient(tmp_path) -> None:
    dataset_path = tmp_path / "sample.csv"
    dataset_path.write_text(CSV_SAMPLE, encoding="utf-8")
    frame = load_applications(dataset_path, columns=["idCv", "idVacancy", "cv_status"])
    groups = group_by_vacancy(frame)

    balanced = sample_balanced_groups(groups, min_positive=1, min_negative=1)
    assert len(balanced) == 2  # vac1 and vac2 have both positive & negative
    assert all(len(group.frame) >= 2 for group in balanced)
