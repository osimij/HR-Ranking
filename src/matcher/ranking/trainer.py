"""LambdaRank training utilities built on LightGBM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb  # type: ignore
except (OSError, ImportError) as exc:  # pragma: no cover - environment dependent
    lgb = None
    _LGBM_IMPORT_ERROR = exc
else:
    _LGBM_IMPORT_ERROR = None


@dataclass(frozen=True)
class LightGBMArtifacts:
    model: lgb.Booster
    feature_names: List[str]
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]


def _group_sizes(frame: pd.DataFrame) -> List[int]:
    return frame.groupby("idVacancy").size().tolist()


def split_feature_table(
    table: pd.DataFrame,
    *,
    validation_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vacancies = table["idVacancy"].unique()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(vacancies)

    n = len(shuffled)
    val_cut = int(n * validation_frac)
    test_cut = val_cut + int(n * test_frac)

    val_ids = set(shuffled[:val_cut])
    test_ids = set(shuffled[val_cut:test_cut])

    validation = table[table["idVacancy"].isin(val_ids)].copy()
    test = table[table["idVacancy"].isin(test_ids)].copy()
    train = table[~table["idVacancy"].isin(val_ids | test_ids)].copy()

    if validation.empty and not train.empty:
        fallback_id = train["idVacancy"].iloc[0]
        mask = train["idVacancy"] == fallback_id
        validation = train[mask].copy()
        train = train[~mask].copy()

    if test.empty and not train.empty:
        fallback_id = train["idVacancy"].iloc[0]
        mask = train["idVacancy"] == fallback_id
        test = train[mask].copy()
        train = train[~mask].copy()

    return train, validation, test


def train_lambdarank(
    table: pd.DataFrame,
    *,
    validation_frac: float = 0.15,
    test_frac: float = 0.15,
    params: Dict[str, float | int | str] | None = None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 30,
) -> LightGBMArtifacts:
    if lgb is None:
        raise RuntimeError(
            "LightGBM is not available. Install lightgbm with OpenMP support."
        ) from _LGBM_IMPORT_ERROR
    if table.empty:
        raise ValueError("Feature table must not be empty.")

    feature_columns = [c for c in table.columns if c not in {"idCv", "idVacancy", "relevance"}]
    train, validation, test = split_feature_table(
        table,
        validation_frac=validation_frac,
        test_frac=test_frac,
    )

    def _prepare_dataset(frame: pd.DataFrame) -> lgb.Dataset:
        if frame.empty:
            raise ValueError("Split contains no rows; adjust validation/test fractions.")
        frame = frame.copy()
        frame = frame.sort_values("idVacancy")
        X = frame[feature_columns].astype(float)
        y = frame["relevance"].astype(int)
        group = _group_sizes(frame)
        if not group:
            raise ValueError("Group sizes are empty; ensure at least one vacancy per split.")
        return lgb.Dataset(X, label=y, group=group, feature_name=feature_columns)

    lgb_train = _prepare_dataset(train)
    lgb_val = _prepare_dataset(validation)

    training_params: Dict[str, float | int | str] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "ndcg_eval_at": [3, 5, 10],
        "verbosity": -1,
    }
    if params:
        training_params.update(params)

    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    
    model = lgb.train(
        training_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )

    best_iteration = model.best_iteration or num_boost_round
    
    # LightGBM 4.x: extract metrics from best_score dictionary
    train_metrics = {}
    validation_metrics = {}
    if hasattr(model, 'best_score') and model.best_score:
        for dataset_name, metrics in model.best_score.items():
            if dataset_name == "train":
                train_metrics = metrics
            elif dataset_name == "val":
                validation_metrics = metrics

    return LightGBMArtifacts(
        model=model,
        feature_names=feature_columns,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
    )


def predict_scores(model: lgb.Booster, frame: pd.DataFrame) -> np.ndarray:
    if lgb is None:
        raise RuntimeError(
            "LightGBM is not available. Install lightgbm with OpenMP support."
        ) from _LGBM_IMPORT_ERROR
    feature_columns = [c for c in frame.columns if c not in {"idCv", "idVacancy", "relevance"}]
    return model.predict(frame[feature_columns].astype(float), num_iteration=model.best_iteration or -1)
