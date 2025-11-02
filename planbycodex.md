# Resume Ranking System - Detailed Implementation Plan

## Project Overview

**Objective**: Build a scalable Python service that accepts a job description (JD) PDF and a batch of resume PDFs, extracts structured text, and returns per-resume fit scores with explanations optimized for Russian-language content.

**Key Requirements**:
- High-quality ranking with explainability
- Sub-second latency per JD after warmup
- Russian-language optimization
- Clean, production-ready code architecture
- Comprehensive evaluation metrics

---

## System Architecture

### Three-Stage Ranking Pipeline

#### Stage 1: BM25 Recall Layer
**Purpose**: Fast candidate retrieval using keyword matching on normalized text

**Implementation**:
- Normalize Russian text (lowercasing, lemmatization with pymorphy2)
- Build inverted index using `rank_bm25` library
- Tune BM25 parameters (k1, b) on validation set
- Return top-N candidates (N=50-100) for further processing
- **Goal**: High recall (>95%) to ensure good candidates aren't filtered out

**Why This Stage**:
- Extremely fast (microseconds)
- Reduces candidate pool from thousands to dozens
- Catches obvious keyword matches

#### Stage 2: Semantic Screening
**Purpose**: Re-rank candidates using deep semantic understanding

**Implementation**:
- Use pre-trained embedding models:
  - **ruBERT**: Russian-optimized BERT (Sberbank/DeepPavlov)
  - **LaBSE**: Multilingual sentence embeddings (Google)
- Generate embeddings for:
  - JD sections (requirements, responsibilities)
  - Resume sections (skills, experience, education)
- Compute cosine similarity between JD and resume embeddings
- Cache all resume embeddings for reuse across multiple JDs
- Return top-K candidates (K=20-30) with similarity scores

**Why This Stage**:
- Captures semantic similarity beyond keywords
- Handles synonyms ("разработчик" vs "программист")
- Understands context and related concepts

#### Stage 3: LightGBM LambdaRank
**Purpose**: Final precise ranking using engineered features

**Model**: LightGBM with LambdaRank objective (learning-to-rank)

**Feature Engineering**:

1. **Skill Features**:
   - Number of matched required skills
   - Number of matched preferred skills
   - Number of missing required skills
   - Skill category overlaps (languages, frameworks, tools)
   - Rare skill matches (weighted higher)

2. **Experience Features**:
   - Years of total experience
   - Years in relevant domain
   - Delta from JD requirements (e.g., JD wants 5y, candidate has 7y = +2)
   - Career progression indicators (promotions, scope growth)

3. **Education Features**:
   - Degree level match (Bachelor's, Master's, PhD)
   - Field relevance to job
   - Institution prestige (if available)

4. **Constraint Flags**:
   - Location compatibility
   - Work authorization/visa requirements
   - Salary expectations vs JD range (if available)
   - Required certifications presence

5. **Semantic Scores**:
   - Stage 2 embedding similarity scores
   - Section-level similarities (skills vs skills, experience vs responsibilities)

6. **Text Statistics**:
   - Resume length (total tokens)
   - Section completeness
   - Formatting quality indicators

**Training Approach**:
- LambdaRank optimizes ranking directly (not classification)
- Learns which features matter most for ranking
- Produces calibrated scores (0-100 range)

---

## Data Strategy

### Primary Training Data: Kaggle Dataset

**Dataset**: [Approved Kaggle resume-ranking dataset for Russian resumes]

**Split Strategy**:
- **Vacancy-grouped splitting**: All resumes for a given vacancy stay in same split (prevents data leakage)
- 70% training (train model)
- 15% validation (tune hyperparameters, select features)
- 15% test (final metric reporting)

### Holdout Validation Sets

**Set A: Organizer Samples**
- Keep completely separate from training
- Use for final pre-submission validation
- Ensures no overfitting to Kaggle data patterns

**Set B: Synthetic Edge Cases**
- Manually crafted examples:
  - Perfect candidate (matches all requirements)
  - Overqualified candidate
  - Underqualified but eager candidate
  - Wrong field entirely
  - Missing key constraint (location, visa)
- Tests system robustness and explanation quality

**Data Compliance**:
- Log organizer approval for external datasets
- Snapshot dataset license and attribution
- Document data provenance in README

---

## Explanation Generation

### Structured Explanation Components

**1. Overall Fit Score**: 0-100 with interpretation
- 85-100: Excellent match
- 70-84: Strong candidate
- 50-69: Moderate fit
- Below 50: Poor match

**2. Matched Requirements**:
```
✓ Python (5 years experience: requirement met)
✓ FastAPI framework
✓ PostgreSQL database experience
✓ Master's degree in Computer Science
```

**3. Missing Requirements**:
```
✗ Kubernetes experience (required)
⚠ Docker mentioned but only 6 months experience (requirement: 2+ years)
```

**4. Strengths**:
```
+ 8 years total experience (exceeds 5 year requirement)
+ Led team of 5 developers
+ Strong algorithmic background (competitive programming)
```

**5. Concerns**:
```
⚠ Currently located in Moscow, job is in Saint Petersburg
⚠ Last role change was 4 years ago (might prefer stability)
```

**6. Feature Importances**:
- Show which factors most influenced the score
- Extracted from LightGBM model's feature_importance

### Optional LLM Polishing

**When to Use**:
- Only if latency budget allows (<200ms overhead)
- For final top-5 candidates only
- To improve explanation readability

**Implementation**:
```
Prompt: "Rewrite this structured evaluation into a natural, professional 
Russian paragraph for a recruiter. Maintain all facts, just improve flow."

Input: [Structured bullet points]
Output: [Polished paragraph in Russian]
```

**Models to Consider**:
- GPT-4 API (best quality, higher cost)
- YandexGPT (Russian-optimized, local option)
- Gigachat (Russian LLM from Sberbank)

---

## Performance Optimization

### Caching Strategy

1. **PDF Parsing Cache**:
   - Hash PDF content → check cache before parsing
   - Store: {pdf_hash: parsed_text_dict}
   - Persistence: Redis or local SQLite

2. **Embedding Cache**:
   - Pre-compute all resume embeddings once
   - Store: {resume_id: embedding_vector}
   - Reload on service startup

3. **Feature Cache**:
   - Pre-extract static features (experience years, education, skills)
   - Only compute dynamic features (JD-specific overlaps) on-demand

### Batching

- **Embedding generation**: Batch process 32-64 resumes at once
- **Similarity computation**: Vectorized numpy operations
- **LightGBM inference**: Batch prediction for all candidates

### Target Latency

**Cold Start** (first JD):
- PDF parsing: ~500ms
- Embedding generation: ~300ms
- Ranking: ~200ms
- **Total**: ~1 second

**Warm Start** (subsequent JDs with cached resumes):
- JD parsing: ~500ms
- Retrieval (BM25): ~50ms
- Semantic screening: ~100ms (cached resume embeddings)
- LightGBM ranking: ~50ms
- **Total**: ~700ms

**Scaling**:
- Linear time complexity with number of resumes (O(n))
- BM25 index scales to 100K+ resumes
- Embedding cache memory: ~1KB per resume (100K resumes = 100MB)

---

## Code Architecture

### Module Structure

```
resume_ranker/
├── ingestion/
│   ├── pdf_parser.py          # PDF → structured text
│   ├── text_cleaner.py         # Layout noise removal
│   └── section_extractor.py   # Identify resume sections
├── normalization/
│   ├── text_normalizer.py      # Lowercasing, lemmatization
│   ├── skill_extractor.py      # NER + pattern matching for skills
│   └── experience_parser.py    # Parse dates, companies, roles
├── retrieval/
│   ├── bm25_index.py           # Build and query BM25 index
│   └── embedding_generator.py  # ruBERT/LaBSE wrapper
├── ranking/
│   ├── feature_engineer.py     # Compute all features
│   ├── lightgbm_ranker.py      # Train and inference
│   └── scorer.py               # Final score calibration
├── explanation/
│   ├── structured_explainer.py # Generate bullet points
│   └── llm_polisher.py         # Optional LLM polishing
├── api/
│   ├── fastapi_app.py          # REST API endpoints
│   ├── schemas.py              # Pydantic models
│   └── demo_interface.py       # HTML demo page
├── evaluation/
│   ├── metrics.py              # NDCG@k, MRR, MAP
│   ├── ablation.py             # Component contribution analysis
│   └── test_suite.py           # Synthetic test cases
├── utils/
│   ├── caching.py              # Redis/SQLite cache manager
│   ├── config.py               # Configuration management
│   └── logging.py              # Structured logging
└── tests/
    ├── test_parsing.py
    ├── test_normalization.py
    ├── test_ranking.py
    └── test_api.py
```

### Code Quality Standards

**Type Hints**: All functions fully annotated
```python
def parse_resume(pdf_path: Path) -> ResumeData:
    ...
```

**Linting**: Black + Flake8 + mypy
- Black: code formatting
- Flake8: style guide enforcement
- mypy: static type checking

**Testing**:
- Unit tests: 80%+ coverage of core logic
- Integration tests: Full pipeline on sample data
- Regression tests: Synthetic edge cases

**Documentation**:
- Docstrings: Google style for all public functions
- README: Setup, usage, architecture
- API docs: Auto-generated from FastAPI

---

## Evaluation & Metrics

### Primary Metric: NDCG@k

**NDCG@10** (Normalized Discounted Cumulative Gain at position 10)
- Measures ranking quality with position decay
- Gold standard for ranking evaluation
- Target: NDCG@10 > 0.75 on test set

**Why NDCG**:
- Rewards getting best candidates at the top
- Handles graded relevance (not just binary)
- Standard in information retrieval

### Secondary Metrics

- **MRR** (Mean Reciprocal Rank): Average position of first relevant result
- **Precision@5**: What % of top-5 are actually good candidates
- **Recall@20**: Did we capture all good candidates in top-20?

### Ablation Studies

**Measure contribution of each component**:

1. **Baseline**: Random ranking → NDCG ≈ 0.3
2. **BM25 only** → NDCG ≈ 0.55
3. **BM25 + Embeddings** → NDCG ≈ 0.68
4. **Full pipeline** → NDCG ≈ 0.78

**Present as table in demo**:
```
Component          | NDCG@10 | Δ from Previous
-------------------|---------|----------------
Random             | 0.32    | -
BM25               | 0.56    | +0.24
+ Semantic         | 0.69    | +0.13
+ LightGBM         | 0.78    | +0.09
```

### Error Analysis

**Track failure modes**:
- False positives: Why did a bad candidate rank high?
- False negatives: Why did a good candidate rank low?
- Document patterns for presentation

---

## Presentation Strategy

### Demo Flow (5 minutes)

**1. Setup** (30 seconds):
- Show input: 1 JD PDF + 20 resume PDFs
- "We'll rank these candidates in under 1 second"

**2. Live Execution** (1 minute):
- Run command: `python demo.py --jd jobs/python_dev.pdf --resumes candidates/*.pdf`
- Show real-time processing logs
- Display results table

**3. Results Walkthrough** (2 minutes):
- Top candidate: Show full explanation
- Highlight matched skills, experience alignment
- Point out missing requirements flagged
- Compare to #2 and #3 candidates

**4. Technical Deep Dive** (1.5 minutes):
- Architecture diagram on slide
- Explain 3-stage funnel with example numbers
  - "Started with 100 candidates → BM25 → 50 → Embeddings → 20 → LightGBM → Final ranking"
- Show Russian-language handling (lemmatization example)

**5. Performance Metrics** (30 seconds):
- NDCG@10: 0.78
- Ablation chart showing each stage's contribution
- Latency: 650ms average

### Slide Deck Outline

**Slide 1: Title**
- Project name
- Team
- Tagline: "AI-powered resume ranking for Russian job market"

**Slide 2: Problem**
- Recruiters manually review 100+ resumes per position
- Keyword search misses semantic matches
- Need explainable, fast, accurate ranking

**Slide 3: Solution Architecture**
- Visual diagram of 3-stage pipeline
- Highlight Russian-language optimizations

**Slide 4: Data & Training**
- Kaggle dataset stats
- Training methodology
- Validation strategy

**Slide 5: Results - Metrics**
- NDCG@10 chart
- Ablation table
- Latency benchmarks

**Slide 6: Results - Demo**
- Live demo or screencast
- Example explanations

**Slide 7: Russian-Language Handling**
- Challenges (morphology, synonyms)
- Solutions (lemmatization, ruBERT)
- Example: "программист" vs "разработчик" captured

**Slide 8: Technical Highlights**
- Clean architecture
- Scalability (caching, batching)
- Code quality (tests, typing)

**Slide 9: Future Work**
- Real-time feedback from recruiters
- Active learning to improve rankings
- Multi-lingual support

**Slide 10: Thank You**
- GitHub repo link
- Team contact

### Backup Plan

**If live demo fails**:
- Pre-recorded screencast (2 minutes)
- Static results screenshots
- Walk through code on laptop

**Practice**:
- Full rehearsal 3x before presentation
- Time each section
- Prepare for Q&A: common questions list

---

## Execution Timeline (40 hours)

### Phase 1: Foundation (7 hours)

**Task 1.1: Scoping & Compliance** (2h)
- Contact organizers for dataset approval
- Download and verify Kaggle dataset
- Check data quality and format
- Document license and attribution
- Define train/val/test splits

**Task 1.2: Repository Setup** (1h)
- Initialize git repo
- Create module structure
- Setup requirements.txt with versions:
  ```
  pdfminer.six==20231228
  pypdf==3.17.0
  pymorphy2==0.9.1
  rank-bm25==0.2.2
  transformers==4.35.0
  sentence-transformers==2.2.2
  lightgbm==4.1.0
  fastapi==0.104.1
  uvicorn==0.24.0
  numpy==1.24.0
  pandas==2.1.0
  scikit-learn==1.3.2
  ```
- Setup linting config (pyproject.toml)

**Task 1.3: Sample Data Collection** (4h)
- Gather 10+ Russian resume PDFs (various formats)
- Gather 3-5 Russian JD PDFs
- Manually create ground truth rankings
- Document edge cases found

### Phase 2: Ingestion Pipeline (5 hours)

**Task 2.1: PDF Parser** (3h)
- Implement pdfminer.six extraction
- Add fallback to pypdf for stubborn PDFs
- Handle encoding issues (UTF-8, Windows-1251)
- Clean layout artifacts (headers, footers, page numbers)
- Test on 10 sample resumes

**Task 2.2: Section Extraction** (2h)
- Identify resume sections (skills, experience, education)
- Identify JD sections (requirements, responsibilities, nice-to-have)
- Use keyword patterns + heuristics
- Validate on samples

### Phase 3: Normalization (4 hours)

**Task 3.1: Text Normalization** (2h)
- Implement pymorphy2 lemmatization
- Handle special cases (English + Russian mixed text)
- Lowercase and clean punctuation
- Unit tests for normalization

**Task 3.2: Entity Extraction** (2h)
- Skill extraction (pattern matching + NER)
- Date parsing for experience (regex + dateparser)
- Education level detection
- Test extraction accuracy on samples

### Phase 4: Stage 1 - BM25 Recall (4 hours)

**Task 4.1: BM25 Index** (2h)
- Build document index with rank_bm25
- Index both full resume and section-specific text
- Test on validation set

**Task 4.2: Parameter Tuning** (2h)
- Tune k1 (term frequency saturation): try [1.2, 1.5, 2.0]
- Tune b (length normalization): try [0.5, 0.75, 1.0]
- Evaluate recall@50 on validation
- Lock parameters for test set

### Phase 5: Stage 2 - Embeddings (4 hours)

**Task 5.1: Embedding Generation** (2h)
- Load pre-trained model (decide: ruBERT vs LaBSE)
- Implement batched embedding generation
- Generate embeddings for all resume sections
- Save to cache (pickle or numpy)

**Task 5.2: Similarity & Ranking** (2h)
- Compute cosine similarity (JD vs candidates)
- Tune top-k cutoff for Stage 3 (try k=20,30,50)
- Benchmark latency
- Validate on samples

### Phase 6: Stage 3 - LightGBM (6 hours)

**Task 6.1: Feature Engineering** (3h)
- Implement all feature categories:
  - Skill overlaps (matched, missing, bonus)
  - Experience features (years, deltas)
  - Education features
  - Constraint flags
  - Semantic scores from Stage 2
- Create feature extraction pipeline
- Debug feature calculation on examples

**Task 6.2: Model Training** (3h)
- Prepare training data format for LambdaRank
  - Group by vacancy_id
  - Label: relevance scores (0-4 scale)
- Train LightGBM with LambdaRank objective:
  ```python
  params = {
      'objective': 'lambdarank',
      'metric': 'ndcg',
      'ndcg_eval_at': [5, 10, 20],
      'learning_rate': 0.05,
      'num_leaves': 31,
      'feature_fraction': 0.8,
  }
  ```
- Hyperparameter tuning on validation set
- Save final model
- Lock test set metrics

### Phase 7: Evaluation (4 hours)

**Task 7.1: Metrics Implementation** (2h)
- Implement NDCG@k calculator
- Compute metrics on val/test sets
- Track metrics per stage for ablation

**Task 7.2: Ablation Studies** (2h)
- Run pipeline with different configurations:
  - BM25 only
  - BM25 + Embeddings
  - Full pipeline
- Generate comparison table
- Document insights

### Phase 8: Explanation & API (4 hours)

**Task 8.1: Structured Explainer** (2h)
- Generate matched/missing skills lists
- Compute experience alignment
- Check constraints
- Extract feature importances from model
- Format as structured JSON

**Task 8.2: API & Demo** (2h)
- Build FastAPI app with endpoints:
  - POST /rank: Upload JD + resumes → get rankings
  - GET /explain/{result_id}: Get detailed explanation
- Create simple HTML demo page
- Test end-to-end with curl/Postman

### Phase 9: Polish (3 hours)

**Task 9.1: Code Quality** (2h)
- Add type hints to all functions
- Run Black formatter
- Fix Flake8 warnings
- Run mypy and fix type errors
- Write docstrings

**Task 9.2: Testing** (1h)
- Write unit tests for parsers
- Write integration test for full pipeline
- Test edge cases (empty resume, malformed PDF)
- Verify test coverage >70%

### Phase 10: Performance Optimization (2 hours)

**Task 10.1: Profiling** (1h)
- Profile pipeline with cProfile
- Identify bottlenecks
- Measure latency per component

**Task 10.2: Caching Implementation** (1h)
- Add PDF parse cache (SQLite)
- Add embedding cache (pickle + dict)
- Verify latency improvement

### Phase 11: Presentation Prep (4 hours)

**Task 11.1: Slide Creation** (2h)
- Design slides (Google Slides or PowerPoint)
- Create architecture diagram (draw.io)
- Generate metric charts (matplotlib)
- Add ablation table
- Polish visuals

**Task 11.2: Demo Script & Rehearsal** (2h)
- Write demo script with timings
- Prepare sample data for demo
- Rehearse full presentation 3x
- Time each section
- Record backup screencast
- Prepare Q&A answers

### Phase 12: Buffer (8 hours)

**Reserved for**:
- PDF parsing issues (encoding, layout quirks)
- Model not converging (hyperparameter search)
- Feature bugs (off-by-one errors, NaN values)
- Demo technical difficulties
- Last-minute presentation changes
- Sleep and breaks

---

## Risk Mitigation

### Risk 1: PDF Parsing Fails on Complex Layouts
**Mitigation**:
- Test multiple parsing libraries early (pdfminer.six, pypdf, pdfplumber)
- Create fallback chain
- Accept plain text upload as backup
- Document known failures

### Risk 2: Model Doesn't Learn/Converge
**Mitigation**:
- Start with simple baseline (weighted average of features)
- Use pre-tuned LightGBM defaults
- Have a "rules-based ranker" as fallback
- Validate on small sample first

### Risk 3: Latency Too High
**Mitigation**:
- Profile early (Phase 10)
- Cut optional features if needed
- Use smaller embedding model (distilled ruBERT)
- Reduce BM25 candidate pool size

### Risk 4: Kaggle Dataset Doesn't Have Labels
**Mitigation**:
- Check dataset format in Phase 1
- If no labels, create synthetic rankings:
  - Use rule-based scorer to generate labels
  - Train LightGBM on synthetic labels
  - Still demonstrates pipeline

### Risk 5: Demo Fails During Presentation
**Mitigation**:
- Pre-recorded screencast backup
- Static results screenshots
- Cached results for instant display
- Practice on different machines

---

## Success Criteria

### Minimum Viable Product (Must Have)
- ✅ Accepts PDF inputs (JD + resumes)
- ✅ Returns ranked list with scores
- ✅ Generates explanations (matched/missing skills)
- ✅ NDCG@10 > 0.65 on test set
- ✅ Working demo (API or CLI)
- ✅ Clean code with tests

### Target Product (Should Have)
- ✅ All 3 pipeline stages working
- ✅ NDCG@10 > 0.75
- ✅ Latency < 1 second per JD
- ✅ Ablation study showing component contributions
- ✅ Russian-language optimization documented
- ✅ FastAPI with HTML demo

### Stretch Goals (Nice to Have)
- ✅ LLM-polished explanations
- ✅ Real-time API deployment (Docker + cloud)
- ✅ A/B test showing improvement over baseline
- ✅ Beautiful UI with charts
- ✅ NDCG@10 > 0.80

---

## Post-Hackathon Improvements

### If We Win or Want to Productionize

**1. Production Deployment**:
- Dockerize application
- Deploy to cloud (AWS Lambda / Google Cloud Run)
- Add authentication and rate limiting
- Monitor with logging and alerts

**2. User Feedback Loop**:
- Let recruiters adjust rankings
- Collect implicit feedback (clicks, hires)
- Retrain model weekly with new data

**3. Advanced Features**:
- Multi-language support (English, German, etc.)
- Diversity-aware ranking (de-bias)
- Salary prediction
- Culture fit scoring
- Interview question suggestions

**4. Scale Optimizations**:
- Vector database for embeddings (Pinecone, Qdrant)
- Distributed processing (Ray, Dask)
- GPU acceleration for embedding generation
- Cache warming strategies

---

## Key Takeaways for Judges

### 1. Technical Sophistication
- Modern ML stack (transformers, LightGBM, ranking algorithms)
- Proper train/val/test methodology
- No data leakage

### 2. Russian-Language Optimization
- Lemmatization with pymorphy2
- Russian BERT embeddings
- Handles morphological complexity

### 3. Explainability
- Not a black box
- Clear structured explanations
- Feature importance from model

### 4. Scalability
- Sub-second latency
- Caching strategy
- Batch processing
- Can handle thousands of resumes

### 5. Code Quality
- Clean architecture
- Type hints and tests
- Production-ready

### 6. Evaluation Rigor
- Multiple metrics
- Ablation studies
- Holdout validation

---

## Contact & Resources

**GitHub Repository**: [To be created]

**Key Libraries**:
- pdfminer.six: PDF parsing
- pymorphy2: Russian lemmatization
- sentence-transformers: Embedding models
- LightGBM: Gradient boosting
- FastAPI: Web framework

**Useful Links**:
- ruBERT model: https://huggingface.co/DeepPavlov/rubert-base-cased
- LaBSE model: https://huggingface.co/sentence-transformers/LaBSE
- LightGBM docs: https://lightgbm.readthedocs.io/
- LambdaRank paper: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/

**Team Roles** (if applicable):
- ML Engineer: Model training and evaluation
- Backend Engineer: API and pipeline
- Data Engineer: Parsing and feature extraction
- Presenter: Demo and slides

---

*This plan is ambitious but achievable with AI-assisted coding. Focus on getting a working MVP by hour 20, then polish and optimize. Good luck! 🚀*

