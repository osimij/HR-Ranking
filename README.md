# HR-Ranking: Resume-Job Matching System

AI-powered system for matching CVs to job descriptions, optimized for Russian-language documents.

## 🎯 Project Overview

This system uses a 3-stage pipeline to rank resumes against job descriptions:
1. **BM25 Retrieval** - Fast keyword-based filtering
2. **Feature Engineering** - Extract 15+ intelligent signals
3. **LambdaRank Ranking** - LightGBM model for final scoring

## 📊 Two Versions Available

### 🔵 Main Branch (RECOMMENDED)
**Original multilingual model - BEST PERFORMANCE**

- **Test NDCG@5:** 0.7820 (78.2% accuracy)
- **Model:** `models/lambdarank_full.txt`
- **Embeddings:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Status:** ✅ Production-ready

**Use this version for:** Production deployment, best ranking accuracy

### 🔬 rubert-experiment Branch (EXPERIMENTAL)
**Russian-specific model - Lower performance**

- **Test NDCG@5:** 0.6258 (62.6% accuracy)  
- **Model:** `models/lambdarank_rubert.txt`
- **Embeddings:** ai-forever/sbert_large_nlu_ru (Russian BERT)
- **Status:** ⚠️ Experimental - needs more tuning

**Use this version for:** Experimenting with Russian-specific embeddings

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/osimij/HR-Ranking.git
cd HR-Ranking

# Install dependencies
pip install -r requirements.txt
```

### ⚠️ Important: Models and Data Files

Due to GitHub file size limits, the following files are **not included** in the repository:
- `train.csv` (973 MB) - Training dataset
- `data/*.parquet` - Generated feature files
- `models/*.txt` - Trained LightGBM models

**To run the system, you need to either:**

1. **Train your own model:**
   ```bash
   # Get training data (train.csv) from your data source
   # Then generate features and train
   python3 scripts/generate_features.py --dataset train.csv --output data/features.parquet
   python3 scripts/train_lambdarank.py --features data/features.parquet --output models/my_model.txt
   ```

2. **Or get pre-trained models** from the project maintainer

### Running the Web App

```bash
# Run the Streamlit interface
python3 -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Switching Between Versions

```bash
# Use the main (better) version
git checkout main

# Or try the ruBERT experimental version
git checkout rubert-experiment
```

## 📁 Project Structure

```
├── app.py                  # Streamlit web interface
├── src/matcher/            # Core matching system
│   ├── embeddings/         # Semantic embeddings
│   ├── ingestion/          # PDF parsing & extraction
│   ├── ranking/            # Feature extraction & training
│   └── retrieval/          # BM25 search
├── models/                 # Trained LightGBM models
├── data/                   # Feature datasets
├── scripts/                # Training scripts
└── tests/                  # Unit tests

```

## 🔧 Training Your Own Model

```bash
# Generate features from training data
python3 scripts/generate_features.py \
    --dataset train.csv \
    --output data/features.parquet

# Train LambdaRank model
python3 scripts/train_lambdarank.py \
    --features data/features.parquet \
    --output models/my_model.txt
```

## 📈 Performance Metrics

| Version | NDCG@5 | Training | Validation | Test |
|---------|--------|----------|------------|------|
| **Main (Multilingual)** | **78.2%** | 80.6% | 79.0% | **78.2%** |
| rubert-experiment | 62.6% | 78.2% | 59.0% | 62.6% |

## 🛠️ Tech Stack

- **Python 3.9+**
- **LightGBM** - LambdaRank for learning-to-rank
- **sentence-transformers** - Semantic embeddings
- **Streamlit** - Web interface
- **Google Gemini API** - Enhanced PDF parsing
- **BM25** - Fast text retrieval

## 📝 Environment Variables

```bash
# Optional: Set your Gemini API key for better PDF parsing
export GEMINI_API_KEY="your-api-key-here"

# Optional: Choose which model to use
export LGBM_MODEL_PATH="models/lambdarank_full.txt"

# Optional: Choose embedding model
export SEMANTIC_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## 🧪 Running Tests

```bash
python3 -m pytest
```

## 📄 License

MIT License

## 👥 Contributors

- Initial development and ML engineering
- Experimented with multilingual vs Russian-specific models

## 🔗 Links

- Repository: https://github.com/osimij/HR-Ranking
- Documentation: See `SYSTEM_EXPLAINED.md` for detailed technical explanation

---

**Recommendation:** Start with the `main` branch for best performance. The `rubert-experiment` branch is available for research and experimentation with Russian-specific language models.

