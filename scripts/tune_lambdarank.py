#!/usr/bin/env python3
"""Hyperparameter sweep with group-aware 3-fold CV for LambdaRank."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features_full_v2.parquet"),
        help="Path to feature table (parquet or csv).",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Input format.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling vacancies.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of group folds.",
    )
    return parser.parse_args()


def load_table(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def group_kfold(vacancies: np.ndarray, folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    shuffled = rng.permutation(vacancies)
    return [arr for arr in np.array_split(shuffled, folds) if len(arr) > 0]


def prepare_dataset(frame: pd.DataFrame, feature_columns: List[str]) -> lgb.Dataset:
    sorted_frame = frame.sort_values("idVacancy")
    X = sorted_frame[feature_columns].astype(float)
    y = sorted_frame["relevance"].astype(int)
    group = sorted_frame.groupby("idVacancy").size().tolist()
    return lgb.Dataset(X, label=y, group=group, feature_name=feature_columns)


def run_fold(
    params: Dict[str, float | int | str],
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: List[str],
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> Tuple[float, int]:
    lgb_train = prepare_dataset(train_frame, feature_columns)
    lgb_val = prepare_dataset(val_frame, feature_columns)
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
    ]
    booster = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        valid_names=["val"],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )
    score = booster.best_score.get("val", {}).get("ndcg@5", float("nan"))
    best_iter = booster.best_iteration or num_boost_round
    return score, best_iter


def main() -> None:
    args = parse_args()
    table = load_table(args.features, args.format)
    print(f"Loaded table: {table.shape[0]} rows, {table.shape[1]} columns")  # noqa: T201

    feature_columns = [c for c in table.columns if c not in {"idCv", "idVacancy", "relevance"}]
    vacancies = table["idVacancy"].unique()
    rng = np.random.default_rng(args.seed)
    folds = group_kfold(vacancies, args.folds, rng)

    combos = list(
        itertools.product(
            [0.01, 0.03],  # learning_rate
            [63, 127],  # num_leaves
            [30, 50],  # min_data_in_leaf
            [1.0, 2.0],  # lambda_l1
            [1.0, 2.0],  # lambda_l2
        )
    )

    base_params: Dict[str, float | int | str] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "verbosity": -1,
    }

    results: List[Tuple[float, Dict[str, float | int | str]]] = []

    for idx, (lr, leaves, min_leaf, l1, l2) in enumerate(combos, start=1):
        params = base_params | {
            "learning_rate": lr,
            "num_leaves": leaves,
            "min_data_in_leaf": min_leaf,
            "lambda_l1": l1,
            "lambda_l2": l2,
        }
        fold_scores: List[float] = []
        print(  # noqa: T201
            f"Combo {idx}/{len(combos)} -> lr={lr}, leaves={leaves}, min_leaf={min_leaf}, l1={l1}, l2={l2}"
        )
        for fold_idx, val_ids in enumerate(folds, start=1):
            train_ids = np.concatenate([fold for fold in folds if not np.array_equal(fold, val_ids)])
            train_frame = table[table["idVacancy"].isin(train_ids)].copy()
            val_frame = table[table["idVacancy"].isin(val_ids)].copy()

            if val_frame.empty or train_frame.empty:
                print(f"  Fold {fold_idx}: skipped (empty split)")  # noqa: T201
                continue
            score, best_iter = run_fold(params, train_frame, val_frame, feature_columns)
            print(f"  Fold {fold_idx}: NDCG@5={score:.4f} (best_iter={best_iter})")  # noqa: T201
            fold_scores.append(score)
        mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        print(f"  -> Mean NDCG@5: {mean_score:.4f}")  # noqa: T201
        results.append((mean_score, params))

    best_score, best_params = max(results, key=lambda x: (x[0], -x[1]["learning_rate"]))
    print("Best params:")  # noqa: T201
    for key, value in best_params.items():
        print(f"  {key}: {value}")  # noqa: T201
    print(f"Best mean NDCG@5: {best_score:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
