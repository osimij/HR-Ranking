#!/usr/bin/env python3
"""Aggressive regularization sweep for LambdaRank on pruned features."""

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
        default=Path("data/features_pruned_top15.parquet"),
        help="Path to feature table (parquet or csv).",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Input format.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of vacancy-group folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling vacancies.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=400,
        help="Maximum boosting rounds per fold.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=60,
        help="Early stopping patience.",
    )
    return parser.parse_args()


def load_table(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def group_kfold(vacancies: np.ndarray, folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    shuffled = rng.permutation(vacancies)
    return [split for split in np.array_split(shuffled, folds) if len(split) > 0]


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
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[float, int]:
    lgb_train = prepare_dataset(train_frame, feature_columns)
    lgb_val = prepare_dataset(val_frame, feature_columns)
    booster = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        valid_names=["val"],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    score = booster.best_score.get("val", {}).get("ndcg@5", float("nan"))
    best_iter = booster.best_iteration or num_boost_round
    return score, best_iter


def main() -> None:
    args = parse_args()
    table = load_table(args.features, args.format)
    print(f"Loaded table: {table.shape[0]} rows, {table.shape[1]} columns")  # noqa: T201

    feature_columns = [c for c in table.columns if c not in {"idVacancy", "idCv", "relevance"}]
    vacancies = table["idVacancy"].unique()
    rng = np.random.default_rng(args.seed)
    folds = group_kfold(vacancies, args.folds, rng)

    param_grid = {
        "num_leaves": [20, 31, 50],
        "min_child_samples": [50, 100, 200],
        "lambda_l1": [0.0, 1.0, 5.0],
        "lambda_l2": [0.0, 1.0, 5.0],
        "feature_fraction": [0.7, 0.8, 1.0],
    }

    grid_keys = list(param_grid.keys())
    grid_values = list(param_grid.values())

    base_params: Dict[str, float | int | str] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.05,
        "verbosity": -1,
    }

    results: List[Tuple[float, Dict[str, float | int | str]]] = []

    for combo_index, values in enumerate(itertools.product(*grid_values), start=1):
        sweep_params = base_params | dict(zip(grid_keys, values))
        print(f"Combo {combo_index}: {sweep_params}")  # noqa: T201
        fold_scores: List[float] = []
        for fold_index, val_ids in enumerate(folds, start=1):
            train_ids = np.concatenate([fold for fold in folds if not np.array_equal(fold, val_ids)])
            train_frame = table[table["idVacancy"].isin(train_ids)].copy()
            val_frame = table[table["idVacancy"].isin(val_ids)].copy()
            if train_frame.empty or val_frame.empty:
                print(f"  Fold {fold_index}: skipped (empty split)")  # noqa: T201
                continue
            score, best_iter = run_fold(
                sweep_params,
                train_frame,
                val_frame,
                feature_columns,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping,
            )
            print(f"  Fold {fold_index}: NDCG@5={score:.4f} (best_iter={best_iter})")  # noqa: T201
            fold_scores.append(score)
        mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        print(f"  -> Mean NDCG@5: {mean_score:.4f}")  # noqa: T201
        results.append((mean_score, sweep_params))

    best_score, best_params = max(results, key=lambda item: item[0])
    print("Best params:")  # noqa: T201
    for key, value in best_params.items():
        print(f"  {key}: {value}")  # noqa: T201
    print(f"Best mean NDCG@5: {best_score:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
