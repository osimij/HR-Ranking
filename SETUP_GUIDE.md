# Setup Guide for Collaborators

Quick guide for getting the HR-Ranking system running on your machine.

## 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/osimij/HR-Ranking.git
cd HR-Ranking
```

## 🐍 Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Step 3: Get the Data Files

The following large files are **not in the repository**. You need to get them from the project maintainer:

### Required Files:
1. **Training data:** `train.csv` (~973 MB) → Place in project root
2. **Pre-trained models:** 
   - `models/lambdarank_full.txt` (recommended, 78% accuracy)
   - `models/lambdarank_rubert.txt` (experimental, 62% accuracy)

**Ask the project maintainer for these files.**

## 🔑 Step 4: Set Up Gemini API (Optional but Recommended)

For better PDF parsing:

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Or add to your ~/.bashrc or ~/.zshrc
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.zshrc
```

Without this, the system will use fallback heuristic parsing (works but less accurate).

## ✅ Step 5: Test the Installation

```bash
# Run tests
python3 -m pytest

# Should see: 20 passed
```

## 🚀 Step 6: Run the Web App

```bash
python3 -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 🔬 Step 7: Train Your Own Model (Optional)

If you want to experiment:

```bash
# Generate features from training data
python3 scripts/generate_features.py \
    --dataset train.csv \
    --output data/features.parquet

# Train the model
python3 scripts/train_lambdarank.py \
    --features data/features.parquet \
    --output models/my_model.txt

# Use your model
export LGBM_MODEL_PATH="models/my_model.txt"
python3 -m streamlit run app.py
```

## 🎯 Two Model Versions Explained

### Version 1: Multilingual (Recommended) - 78.2% accuracy
- Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Best performance on test data
- Faster inference
- **This is what you should use for production**

### Version 2: ruBERT (Experimental) - 62.6% accuracy
- Uses `ai-forever/sbert_large_nlu_ru`
- Russian-specific language model
- Slower but theoretically better for Russian nuances
- **Currently underperforming, needs more tuning**

To switch between models:
```bash
# Use multilingual model (better)
export LGBM_MODEL_PATH="models/lambdarank_full.txt"

# Use ruBERT model (experimental)
export LGBM_MODEL_PATH="models/lambdarank_rubert.txt"
export SEMANTIC_MODEL_NAME="ai-forever/sbert_large_nlu_ru"
```

## 🐛 Troubleshooting

### "No module named 'google.genai'"
```bash
pip install google-genai
```

### "Model file not found"
Make sure you copied the model files to the `models/` directory.

### "streamlit: command not found"
```bash
# Use the full Python module path
python3 -m streamlit run app.py
```

### Gemini API errors
The system will automatically fall back to heuristic parsing. To fix:
- Check your `GEMINI_API_KEY` is set correctly
- Verify the API key is valid at https://aistudio.google.com/

## 📁 Project Structure

```
HR-Ranking/
├── app.py              # Streamlit web interface (START HERE)
├── train.csv           # Training data (GET FROM MAINTAINER)
├── requirements.txt    # Python dependencies
├── src/matcher/        # Core matching system
├── models/             # Trained models (GET FROM MAINTAINER)
├── data/               # Generated features
├── scripts/            # Training scripts
└── tests/              # Unit tests
```

## 🤝 Need Help?

- Check `README.md` for detailed documentation
- Check `SYSTEM_EXPLAINED.md` for technical details
- Contact the project maintainer
- Run tests: `python3 -m pytest -v` for detailed output

## 📝 Quick Commands Cheat Sheet

```bash
# Install everything
pip install -r requirements.txt

# Run tests
python3 -m pytest

# Start web app
python3 -m streamlit run app.py

# Generate features
python3 scripts/generate_features.py --dataset train.csv --output data/features.parquet

# Train model
python3 scripts/train_lambdarank.py --features data/features.parquet --output models/my_model.txt
```

---

**Ready to go!** Start with the web app (`python3 -m streamlit run app.py`) and upload some CVs to see it in action.

