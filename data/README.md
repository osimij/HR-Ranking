# Data Directory

This directory contains generated feature files for training.

## Required Data Files

Due to GitHub file size limits, data files are not included in the repository.

### Training Dataset
- **File:** `train.csv` (should be in project root)
- **Size:** ~973 MB
- **Content:** Resume-Job pairs with relevance labels
- **Source:** Contact project maintainer

### Generated Feature Files

These are created by running `scripts/generate_features.py`:

- `features_full.parquet` - All features from full dataset
- `features_enriched.parquet` - Enhanced feature set
- `features_pruned_top15.parquet` - Top 15 features only
- `features_rubert.parquet` - Features using ruBERT embeddings

## Generating Features

```bash
# Basic feature generation (multilingual embeddings)
python3 scripts/generate_features.py \
    --dataset ../train.csv \
    --output features.parquet

# With ruBERT embeddings
python3 scripts/generate_features.py \
    --dataset ../train.csv \
    --embedding-model ai-forever/sbert_large_nlu_ru \
    --output features_rubert.parquet

# Limit data for testing
python3 scripts/generate_features.py \
    --dataset ../train.csv \
    --limit 1000 \
    --output features_test.parquet
```

## Feature Files Structure

Parquet files contain:
- All features extracted by `FeatureExtractor`
- Columns: resume_id, vacancy_id, label, feature1, feature2, ...
- Ready for training with `train_lambdarank.py`

