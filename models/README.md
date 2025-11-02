# Models Directory

This directory should contain trained LightGBM models.

## Required Models

Due to GitHub file size limits, model files are not included in the repository.

### Main Version (RECOMMENDED - 78.2% accuracy)
- **File:** `lambdarank_full.txt`
- **Embeddings used:** paraphrase-multilingual-MiniLM-L12-v2
- **Performance:** NDCG@5 = 0.7820

### Experimental ruBERT Version (62.6% accuracy)
- **File:** `lambdarank_rubert.txt`
- **Embeddings used:** ai-forever/sbert_large_nlu_ru
- **Performance:** NDCG@5 = 0.6258

## How to Get Models

1. **Train your own:** Use the scripts in `/scripts` directory
2. **Contact the project maintainer:** Get pre-trained model files

## Training Your Own Model

```bash
# Generate features
python3 scripts/generate_features.py \
    --dataset train.csv \
    --output data/features.parquet

# Train model
python3 scripts/train_lambdarank.py \
    --features data/features.parquet \
    --output models/my_model.txt
```

## Model Selection

The app automatically looks for models in this order:
1. Path set in `LGBM_MODEL_PATH` environment variable
2. `lambdarank_rubert.txt` (experimental)
3. `lambdarank_pruned_baseline.txt` (recommended)

