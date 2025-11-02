# Models Directory

This directory contains trained LightGBM models.

## ✅ Included Pre-trained Models

### Main Version (RECOMMENDED - 78.2% accuracy)
- **File:** `lambdarank_full.txt` ✅ Included
- **Embeddings used:** paraphrase-multilingual-MiniLM-L12-v2
- **Performance:** NDCG@5 = 0.7820
- **Status:** Production-ready

### Other Available Models
- **File:** `lambdarank_pruned_baseline.txt` ✅ Included
- **Performance:** NDCG@5 = 0.6582
- **Status:** Baseline with 15 features

### Experimental Versions
- `lambdarank.txt` - Early version
- `lambdarank_full_v2.txt` - Alternative training run

**Note:** ruBERT model (`lambdarank_rubert.txt`) needs to be regenerated if you want to experiment with Russian-specific embeddings.

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

