#!/usr/bin/env python3
"""Generate feature table from the Kaggle resume-vacancy dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from matcher.ingestion.dataset import load_applications
from matcher.ranking.dataset_builder import build_feature_table
from matcher.embeddings.semantic import SemanticEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("train.csv"),
        help="Path to the Kaggle dataset CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Optional row limit when sampling the dataset.",
    )
    parser.add_argument(
        "--max-vacancies",
        type=int,
        default=500,
        help="Number of vacancy groups to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features.parquet"),
        help="Output file path for the generated feature table.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable semantic embedding features (faster).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_applications(args.dataset, limit=args.limit)
    embedder = None
    if not args.no_embeddings:
        try:
            embedder = SemanticEmbedder()
        except RuntimeError as exc:  # noqa: F841
            print("[warn] Semantic embeddings unavailable, continuing without them.")  # noqa: T201
            embedder = None
    table = build_feature_table(frame, max_vacancies=args.max_vacancies, embedder=embedder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    print(f"Saved {len(table)} rows with {len(table.columns)} columns to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
