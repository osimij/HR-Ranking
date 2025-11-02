#!/usr/bin/env python3
"""Train a LightGBM LambdaRank model on a generated feature table."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from matcher.ranking.trainer import (
    predict_scores,
    split_feature_table,
    train_lambdarank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features_sample.parquet"),
        help="Path to feature table in Parquet or CSV format.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Input file format.",
    )
    parser.add_argument(
        "--validation-frac",
        type=float,
        default=0.15,
        help="Fraction of vacancies to reserve for validation.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of vacancies to reserve for testing.",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=200,
        help="Maximum number of boosting rounds.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=30,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="Optional path to save the trained LightGBM model.",
    )
    return parser.parse_args()


def load_features(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def group_ndcg(frame: pd.DataFrame, predictions: np.ndarray, k: int = 5) -> float:
    """Compute mean NDCG@k over vacancy groups."""

    frame = frame.copy()
    frame = frame.assign(predictions=predictions)
    scores: Iterable[Tuple[np.ndarray, np.ndarray]] = (
        (
            group.sort_values("predictions", ascending=False)["relevance"].to_numpy(),
            group.sort_values("relevance", ascending=False)["relevance"].to_numpy(),
        )
        for _, group in frame.groupby("idVacancy")
    )

    ndcgs = []
    for preds, ideal in scores:
        ndcg = _ndcg_at_k(preds, ideal, k)
        if not math.isnan(ndcg):
            ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def _ndcg_at_k(preds: np.ndarray, ideal: np.ndarray, k: int) -> float:
    k = min(k, len(preds))
    if k == 0:
        return float("nan")
    gains = preds[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_gains = ideal[:k]
    ideal_dcg = float(np.sum(ideal_gains * discounts))
    if ideal_dcg == 0.0:
        return float("nan")
    return dcg / ideal_dcg


def main() -> None:
    args = parse_args()
    table = load_features(args.features, args.format)
    print(f"Loaded feature table with shape {table.shape}")  # noqa: T201

    try:
        artifacts = train_lambdarank(
            table,
            validation_frac=args.validation_frac,
            test_frac=args.test_frac,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping,
        )
    except RuntimeError as exc:  # LightGBM missing / misconfigured
        raise SystemExit(str(exc)) from exc

    print("Train metrics:", artifacts.train_metrics)  # noqa: T201
    print("Validation metrics:", artifacts.validation_metrics)  # noqa: T201

    train, validation, test = split_feature_table(
        table,
        validation_frac=args.validation_frac,
        test_frac=args.test_frac,
    )

    if not test.empty:
        preds = predict_scores(artifacts.model, test)
        ndcg = group_ndcg(test, preds, k=5)
        print(f"Test NDCG@5: {ndcg:.4f}")  # noqa: T201

    if args.output_model:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        artifacts.model.save_model(str(args.output_model))
        print(f"Saved model to {args.output_model}")  # noqa: T201


if __name__ == "__main__":
    main()
