# 🎯 Resume-Job Matching System - Web Interface

## Quick Start Guide

### 1. Install Dependencies

```bash
pip install streamlit
```

(Or install all requirements: `pip install -r requirements.txt`)

### 2. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Using the Interface

1. **Upload Job Description**
   - Click "Upload Job Description (PDF)" in the sidebar
   - Select your vacancy/job description PDF file

2. **Upload Resumes**
   - Click "Upload Candidate Resumes (PDF)" in the sidebar
   - Select one or multiple resume PDF files
   - You can select multiple files at once (Ctrl/Cmd + Click)

3. **Rank Candidates**
   - Click the blue "🚀 Rank Candidates" button
   - Wait a few seconds while the system processes

4. **View Results**
   - See ranked candidates with relevance scores
   - Top 3 candidates are automatically expanded
   - Click any candidate to see details:
     - Skills and experience
     - Salary expectations
     - Feature analysis (optional)

5. **Download Results**
   - Click "📥 Download Results as CSV" at the bottom
   - Get a CSV file with rankings

---

## Features

### What You See:

- **🥇 Rank Badges**: Top 3 candidates get gold/silver/bronze
- **Relevance Score**: LightGBM prediction (0-1 scale)
- **BM25 Score**: Text similarity from retrieval stage
- **Resume Summary**: Key details (skills, experience, education, salary)
- **Feature Details** (optional): All 15 engineered features

### Options:

- **Show feature details**: See the 15 numeric features that drive the ranking
- **Show BM25 scores**: Display the initial retrieval scores

---

## Example Workflow

```
1. Upload:  vacancy.pdf + resume1.pdf, resume2.pdf, resume3.pdf

2. Process: System runs in ~5 seconds
   - Extracts text from PDFs
   - Normalizes Russian text
   - Runs BM25 retrieval
   - Extracts 15 features per candidate
   - Predicts relevance with LightGBM

3. Results:
   🥇 resume2.pdf - Score: 0.8234
   🥈 resume1.pdf - Score: 0.6781
   🥉 resume3.pdf - Score: 0.4512

4. Download: ranking_results.csv
```

---

## Troubleshooting

### "Model not found" error
- Make sure `models/lambdarank_pruned_baseline.txt` exists
- Run training scripts first if needed

### "Error processing files"
- Ensure PDFs are readable (not scanned images)
- Check PDFs contain Russian text
- Try re-uploading the files

### Slow performance
- Processing 10+ resumes may take 10-30 seconds
- Consider using fewer files for demo

### Missing features
- Install all dependencies: `pip install -r requirements.txt`
- Ensure you have LightGBM with OpenMP support

---

## Demo Tips

### For Presentation:

1. **Prepare Sample Files**:
   - 1 job description PDF
   - 5-10 candidate resume PDFs
   - Mix of good and bad matches

2. **Story Arc**:
   - "Here's a real vacancy from our dataset"
   - "We have 10 candidates who applied"
   - *Upload and rank*
   - "Our system ranks them in 5 seconds"
   - "Notice how the top candidate has high skill overlap and salary alignment"

3. **Show Features** (optional):
   - Enable "Show feature details"
   - Point out key features for top candidate:
     - `skill_f1`: 0.85 (strong skill match)
     - `salary_ratio`: 0.92 (realistic expectations)
     - `bm25_score`: 45.2 (good keyword match)

4. **Compare Rankings**:
   - "Without our system, you'd review all 10 manually"
   - "With our system, focus on the top 3 and save 70% of time"

---

## Technical Details

### What Happens Behind the Scenes:

```python
# 1. Parse documents
job = parse_job_description(job_pdf)
resumes = [parse_resume(pdf) for pdf in resume_pdfs]

# 2. BM25 retrieval (fast filtering)
bm25_results = BM25ResumeIndex(resumes).search(job)

# 3. Feature extraction (15 features per candidate)
for resume, bm25_score in bm25_results:
    features = extract_features(resume, job, bm25_score)
    
# 4. LightGBM ranking
relevance_scores = model.predict(features)

# 5. Sort and display
ranked = sort_by_score(candidates)
```

### Model Used:
- **Algorithm**: LightGBM LambdaRank
- **Features**: 15 engineered signals
- **Performance**: NDCG@5 = 0.7820
- **File**: `models/lambdarank_pruned_baseline.txt`

---

## Customization

### Change Colors:
Edit CSS in `app.py`:
```python
.rank-1 { background-color: #FFD700; }  # Gold
.rank-2 { background-color: #C0C0C0; }  # Silver
.rank-3 { background-color: #CD7F32; }  # Bronze
```

### Change Default Options:
```python
show_features = st.sidebar.checkbox("Show feature details", value=True)  # Default ON
show_bm25 = st.sidebar.checkbox("Show BM25 scores", value=False)  # Default OFF
```

### Adjust Top K:
```python
# In main(), change expanded candidates
expanded=(rank <= 5)  # Show top 5 instead of 3
```

---

## Next Steps

After successful demo:
1. **API version**: Wrap this in FastAPI for production
2. **Batch processing**: Handle 100+ resumes at once
3. **Database**: Store results for analytics
4. **Advanced UI**: Add filters, search, comparison views

---

**Ready to demo! 🚀**

