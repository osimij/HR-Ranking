# ruBERT Experimental Branch

⚠️ **This is the experimental branch for Russian BERT embeddings**

## Performance Comparison

| Branch | Model | NDCG@5 | Status |
|--------|-------|--------|--------|
| **main** | Multilingual MiniLM | **78.2%** | ✅ Recommended |
| **rubert-experiment** | Russian SBERT | 62.6% | ⚠️ Experimental |

## What's Different?

This branch is configured to use Russian-specific BERT embeddings:
- **Embedding model:** `ai-forever/sbert_large_nlu_ru`
- **Expected advantage:** Better understanding of Russian workplace terminology
- **Actual result:** Lower performance (needs more tuning)

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Load ruBERT Configuration
```bash
source rubert_config.sh
```

### 3. Generate Features with ruBERT
```bash
python3 scripts/generate_features.py \
    --dataset train.csv \
    --embedding-model ai-forever/sbert_large_nlu_ru \
    --output data/features_rubert.parquet
```

### 4. Train ruBERT Model
```bash
python3 scripts/train_lambdarank.py \
    --features data/features_rubert.parquet \
    --output models/lambdarank_rubert.txt
```

### 5. Run the App
```bash
source rubert_config.sh
python3 -m streamlit run app.py
```

## Why Lower Performance?

Possible reasons the ruBERT version underperforms:
1. **Training data mismatch:** Model trained on features from multilingual embeddings
2. **Insufficient tuning:** ruBERT parameters need optimization
3. **Embedding quality:** The specific ruBERT model may not be optimal for HR matching
4. **Feature engineering:** May need different features for ruBERT semantic signals

## Switching Back to Main Branch

```bash
git checkout main
python3 -m streamlit run app.py
```

The main branch uses multilingual embeddings and performs better (78.2% vs 62.6%).

## Experimentation Ideas

To improve ruBERT performance:

1. **Try different ruBERT models:**
   ```bash
   export SEMANTIC_MODEL_NAME="DeepPavlov/rubert-base-cased"
   ```

2. **Adjust LightGBM hyperparameters** in `scripts/train_lambdarank.py`

3. **Feature engineering:** Add more Russian-specific features

4. **Ensemble approach:** Combine multilingual and ruBERT predictions

---

**Recommendation:** Use the `main` branch for production. This branch is for research and experimentation.

