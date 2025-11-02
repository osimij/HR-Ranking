#!/bin/bash
# Configuration for ruBERT Experiment Branch
# Source this file before running the app: source rubert_config.sh

# Use Russian BERT for semantic embeddings
export SEMANTIC_MODEL_NAME="ai-forever/sbert_large_nlu_ru"

# Point to ruBERT-trained model (needs to be generated)
export LGBM_MODEL_PATH="models/lambdarank_rubert.txt"

# Optional: Gemini API key
# export GEMINI_API_KEY="your_api_key_here"

echo "✅ ruBERT configuration loaded"
echo "   - Semantic model: $SEMANTIC_MODEL_NAME"
echo "   - LightGBM model: $LGBM_MODEL_PATH"

