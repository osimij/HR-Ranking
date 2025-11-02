#!/usr/bin/env python3
"""Simple web interface for Resume-Job Matching System."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import lightgbm as lgb
import pandas as pd
import streamlit as st

from matcher.ingestion.gemini_client import GeminiClientError, gemini_completion
from matcher.ingestion.llm_parser import (
    LLMExtractionError,
    LLMJobParser,
    LLMResumeParser,
)
from matcher.ingestion.models import parse_job_description, parse_resume
from matcher.ingestion.pdf_reader import PdfDocument, read_pdf_text
from matcher.normalization.text import RussianTextNormalizer
from matcher.ranking.features import FeatureExtractor
from matcher.retrieval.bm25 import BM25ResumeIndex


# Temporary API key setup for Gemini integration (TODO: move to secure storage)
DEFAULT_GEMINI_API_KEY = "AIzaSyCJgnONL0x4RVOJ6XDhmFwuAYMkYWIfiEc"
os.environ.setdefault("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY)

# Page config
st.set_page_config(
    page_title="Resume-Job Matcher",
    page_icon="🎯",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .rank-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        color: white;
    }
    .rank-1 { background-color: #FFD700; color: #000; }
    .rank-2 { background-color: #C0C0C0; color: #000; }
    .rank-3 { background-color: #CD7F32; color: #fff; }
    .rank-other { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained LightGBM model."""
    model_path = Path(__file__).resolve().parent / "models" / "lambdarank_pruned_baseline.txt"
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        st.stop()
    booster = lgb.Booster(model_file=str(model_path))
    # Get the feature names the model expects
    feature_names = booster.feature_name()
    return booster, feature_names


@st.cache_resource
def initialize_components():
    """Initialize normalizer, feature extractor, and LLM parsers."""
    normalizer = RussianTextNormalizer()
    
    # Try to initialize embedder for semantic features
    try:
        from matcher.embeddings.semantic import SemanticEmbedder
        embedder = SemanticEmbedder()
        extractor = FeatureExtractor(normalizer=normalizer, embedder=embedder)
    except Exception:
        # Fall back to no embedder (semantic features will be 0)
        extractor = FeatureExtractor(normalizer=normalizer, embedder=None)
    
    llm_resume_parser = None
    llm_job_parser = None
    llm_error = None
    try:
        llm_resume_parser = LLMResumeParser(completion=gemini_completion, normalizer=normalizer)
        llm_job_parser = LLMJobParser(completion=gemini_completion, normalizer=normalizer)
    except GeminiClientError as exc:
        llm_error = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        llm_error = f"Unexpected Gemini initialization error: {exc}"
    
    return normalizer, extractor, llm_resume_parser, llm_job_parser, llm_error


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to temporary location."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        return Path(tmp_file.name)


def extract_features_for_pair(resume_content, job_content, extractor, bm25_score=None):
    """Extract features for a single resume-job pair."""
    features = extractor.extract(resume_content, job_content, bm25_score=bm25_score)
    return features.features


def rank_badge(rank):
    """Generate HTML for rank badge."""
    if rank == 1:
        return '<span class="rank-badge rank-1">🥇 #1</span>'
    elif rank == 2:
        return '<span class="rank-badge rank-2">🥈 #2</span>'
    elif rank == 3:
        return '<span class="rank-badge rank-3">🥉 #3</span>'
    else:
        return f'<span class="rank-badge rank-other">#{rank}</span>'


def main():
    st.markdown('<div class="main-header">🎯 Resume-Job Matching System</div>', unsafe_allow_html=True)
    st.markdown("**Upload a job description and candidate resumes to see AI-powered rankings**")
    
    # Load model and components
    model, model_features = load_model()
    normalizer, extractor, llm_resume_parser, llm_job_parser, llm_error = initialize_components()
    # Ensure cached parsers expose the latest PDF parsing helpers; clear cache otherwise.
    missing_methods = [
        name
        for name, parser in (
            ("resume", llm_resume_parser),
            ("job", llm_job_parser),
        )
        if parser is not None and not hasattr(parser, "parse_pdf_bytes")
    ]
    if missing_methods:
        initialize_components.clear()
        normalizer, extractor, llm_resume_parser, llm_job_parser, llm_error = initialize_components()
        missing_methods = [
            name
            for name, parser in (
                ("resume", llm_resume_parser),
                ("job", llm_job_parser),
            )
            if parser is not None and not hasattr(parser, "parse_pdf_bytes")
        ]
        if missing_methods:
            llm_error = (
                f"Updated Gemini parsers unavailable ({', '.join(missing_methods)} parser lacks PDF support). "
                "Restart the app to refresh LLM components."
            )
            llm_resume_parser = None
            llm_job_parser = None
    llm_available = llm_resume_parser is not None and llm_job_parser is not None
    if llm_error:
        st.warning(f"Gemini parsing unavailable. Falling back to heuristic parser. Details: {llm_error}")
    
    # Sidebar for uploads
    st.sidebar.header("📁 Upload Files")
    
    # Job description upload
    job_file = st.sidebar.file_uploader(
        "Upload Job Description (PDF)",
        type=["pdf"],
        help="Upload the job vacancy description as a PDF file"
    )
    
    # Resume uploads
    resume_files = st.sidebar.file_uploader(
        "Upload Candidate Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more candidate resumes as PDF files"
    )
    
    # Options
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Options")
    show_features = st.sidebar.checkbox("Show feature details", value=False)
    show_bm25 = st.sidebar.checkbox("Show BM25 scores", value=True)
    use_gemini = False
    if llm_available:
        use_gemini = st.sidebar.checkbox(
            "Use Gemini parsing (LLM)",
            value=True,
            help="LLM-powered extraction for richer structured data",
        )
    else:
        st.sidebar.info("Gemini parsing disabled; using fallback parser.")
    
    # Process button
    if st.sidebar.button("🚀 Rank Candidates", type="primary", use_container_width=True):
        if not job_file:
            st.error("⚠️ Please upload a job description first!")
            return
        
        if not resume_files:
            st.error("⚠️ Please upload at least one resume!")
            return
        
        # Show progress
        with st.spinner("Processing documents..."):
            try:
                fallback_messages: list[str] = []
                job_llm_mode = "none"
                resumes_llm_pdf = 0
                resumes_llm_text = 0
                
                # Parse job description
                job_content = None
                job_doc: Optional[PdfDocument] = None
                job_path = save_uploaded_file(job_file)
                try:
                    job_bytes = job_path.read_bytes()
                    if use_gemini and llm_job_parser:
                        try:
                            job_content = llm_job_parser.parse_pdf_bytes(job_bytes, source_path=job_path)
                            job_llm_mode = "pdf"
                        except (LLMExtractionError, GeminiClientError) as exc:
                            fallback_messages.append(
                                f"Gemini PDF parsing failed for job description. Trying text fallback. ({exc})"
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            fallback_messages.append(
                                f"Unexpected Gemini PDF error for job description: {exc}. Trying text fallback."
                            )
                    if job_content is None:
                        if job_doc is None:
                            job_doc = read_pdf_text(job_path)
                        if use_gemini and llm_job_parser:
                            try:
                                job_content = llm_job_parser.parse(job_doc)
                                job_llm_mode = "text"
                            except (LLMExtractionError, GeminiClientError) as exc:
                                fallback_messages.append(
                                    f"Gemini text parsing failed for job description. Using heuristic parser. ({exc})"
                                )
                            except Exception as exc:  # pragma: no cover - defensive
                                fallback_messages.append(
                                    f"Unexpected Gemini text error for job description: {exc}. Using heuristic parser."
                                )
                    if job_content is None:
                        if job_doc is None:
                            job_doc = read_pdf_text(job_path)
                        job_content = parse_job_description(job_doc, normalizer=normalizer)
                finally:
                    job_path.unlink(missing_ok=True)
                
                # Parse all resumes
                resumes = []
                resume_names = []
                for resume_file in resume_files:
                    resume_content = None
                    resume_doc: Optional[PdfDocument] = None
                    resume_path = save_uploaded_file(resume_file)
                    try:
                        resume_bytes = resume_path.read_bytes()
                        if use_gemini and llm_resume_parser:
                            try:
                                resume_content = llm_resume_parser.parse_pdf_bytes(resume_bytes, source_path=resume_path)
                                resumes_llm_pdf += 1
                            except (LLMExtractionError, GeminiClientError) as exc:
                                fallback_messages.append(
                                    f"Gemini PDF parsing failed for resume '{resume_file.name}'. Trying text fallback. ({exc})"
                                )
                            except Exception as exc:  # pragma: no cover - defensive
                                fallback_messages.append(
                                    f"Unexpected Gemini PDF error for resume '{resume_file.name}': {exc}. Trying text fallback."
                                )
                        if resume_content is None:
                            if resume_doc is None:
                                resume_doc = read_pdf_text(resume_path)
                            if use_gemini and llm_resume_parser:
                                try:
                                    resume_content = llm_resume_parser.parse(resume_doc)
                                    resumes_llm_text += 1
                                except (LLMExtractionError, GeminiClientError) as exc:
                                    fallback_messages.append(
                                        f"Gemini text parsing failed for resume '{resume_file.name}'. Using heuristic parser. ({exc})"
                                    )
                                except Exception as exc:  # pragma: no cover - defensive
                                    fallback_messages.append(
                                        f"Unexpected Gemini text error for resume '{resume_file.name}': {exc}. Using heuristic parser."
                                    )
                        if resume_content is None:
                            if resume_doc is None:
                                resume_doc = read_pdf_text(resume_path)
                            resume_content = parse_resume(resume_doc, normalizer=normalizer)
                    finally:
                        resume_path.unlink(missing_ok=True)
                    
                    resumes.append(resume_content)
                    resume_names.append(resume_file.name)
                
                st.success(f"✅ Processed 1 job description and {len(resumes)} resumes")
                
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                return
        
        if use_gemini and llm_available:
            summary_bits = []
            if job_llm_mode == "pdf":
                summary_bits.append("job description via Gemini (PDF)")
            elif job_llm_mode == "text":
                summary_bits.append("job description via Gemini (text)")
            else:
                summary_bits.append("job description via heuristic parser")

            if resumes_llm_pdf or resumes_llm_text:
                pdf_part = f"{resumes_llm_pdf} PDF" if resumes_llm_pdf else None
                text_part = f"{resumes_llm_text} text" if resumes_llm_text else None
                modes = ", ".join(part for part in [pdf_part, text_part] if part)
                summary_bits.append(f"resumes via Gemini ({modes})")
            else:
                summary_bits.append("resumes via heuristic parser")
            st.info("LLM parsing summary: " + ", ".join(summary_bits))
        for message in fallback_messages:
            st.warning(message)
        
        # BM25 retrieval
        with st.spinner("Running BM25 retrieval..."):
            try:
                bm25_index = BM25ResumeIndex(resumes)
                bm25_results = bm25_index.search(job_content, top_k=len(resumes))
            except Exception as e:
                st.error(f"❌ Error in BM25 retrieval: {str(e)}")
                return
        
        # Feature extraction and ranking
        with st.spinner("Extracting features and ranking..."):
            try:
                results = []
                missing_feature_names = set()
                for result in bm25_results:
                    resume_idx = resumes.index(result.resume)
                    
                    # Extract features
                    features = extract_features_for_pair(
                        result.resume,
                        job_content,
                        extractor,
                        bm25_score=result.score
                    )
                    
                    # Align features with model expectations, defaulting missing values to 0
                    feature_row = [features.get(name, 0.0) for name in model_features]
                    missing = [name for name in model_features if name not in features]
                    if missing:
                        missing_feature_names.update(missing)
                    feature_vector = pd.DataFrame([feature_row], columns=model_features)
                    
                    # Predict relevance score
                    relevance_score = model.predict(feature_vector)[0]
                    
                    results.append({
                        'name': resume_names[resume_idx],
                        'bm25_score': result.score,
                        'relevance_score': relevance_score,
                        'features': features,
                        'resume': result.resume
                    })
                
                # Sort by relevance score
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                
            except Exception as e:
                st.error(f"❌ Error in ranking: {str(e)}")
                st.exception(e)
                return
        
        if missing_feature_names:
            st.warning(
                "Some model features were missing from the extractor output and were set to 0 "
                f"for scoring: {', '.join(sorted(missing_feature_names))}"
            )
        
        # Display results
        st.markdown("---")
        st.header("📊 Ranking Results")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Candidates", len(results))
        with col2:
            st.metric("Top Score", f"{results[0]['relevance_score']:.4f}")
        with col3:
            avg_score = sum(r['relevance_score'] for r in results) / len(results)
            st.metric("Average Score", f"{avg_score:.4f}")
        
        st.markdown("---")
        
        # Display each candidate
        for rank, result in enumerate(results, 1):
            with st.expander(
                f"{rank}. **{result['name']}** - Score: {result['relevance_score']:.4f}",
                expanded=(rank <= 3)
            ):
                # Rank badge
                st.markdown(rank_badge(rank), unsafe_allow_html=True)
                
                # Scores
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Relevance Score (LightGBM)**")
                    score = result['relevance_score']
                    progress_value = max(0.0, min(score, 1.0))
                    st.progress(progress_value)
                    st.write(f"`{score:.4f}`")
                
                if show_bm25:
                    with col2:
                        st.markdown("**🔍 BM25 Retrieval Score**")
                        st.write(f"`{result['bm25_score']:.2f}`")
                
                # Resume details
                st.markdown("**📄 Resume Summary**")
                resume = result['resume']
                
                details_col1, details_col2 = st.columns(2)
                with details_col1:
                    st.write(f"**Skills:** {len(resume.skills)} listed")
                    if resume.skills[:3]:
                        st.write(f"Top: {', '.join(resume.skills[:3])}")
                    
                    st.write(f"**Experience:** {len(resume.experiences)} positions")
                    if resume.experience_years:
                        st.write(f"Total: {resume.experience_years} years")
                
                with details_col2:
                    st.write(f"**Education:** {len(resume.educations)} entries")
                    if resume.salary_min or resume.salary_max:
                        salary_str = f"{resume.salary_min or 0:,.0f}"
                        if resume.salary_max and resume.salary_max != resume.salary_min:
                            salary_str += f" - {resume.salary_max:,.0f}"
                        st.write(f"**Salary Expectation:** {salary_str} RUB")
                
                # Feature details
                if show_features:
                    st.markdown("**🔬 Feature Details**")
                    features_df = pd.DataFrame([result['features']]).T
                    features_df.columns = ['Value']
                    features_df = features_df.sort_values('Value', ascending=False)
                    
                    # Show top features
                    st.dataframe(
                        features_df.head(10),
                        use_container_width=True,
                        height=300
                    )
        
        # Download results
        st.markdown("---")
        results_df = pd.DataFrame([
            {
                'Rank': rank,
                'Candidate': r['name'],
                'Relevance_Score': f"{r['relevance_score']:.4f}",
                'BM25_Score': f"{r['bm25_score']:.2f}"
            }
            for rank, r in enumerate(results, 1)
        ])
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            "ranking_results.csv",
            "text/csv",
            use_container_width=True
        )
    
    else:
        # Instructions when no files uploaded
        st.info("👆 Upload files in the sidebar and click 'Rank Candidates' to start!")
        
        st.markdown("---")
        st.markdown("### 📖 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**1️⃣ Retrieval**")
            st.write("BM25 algorithm quickly filters candidates by keyword match")
        
        with col2:
            st.markdown("**2️⃣ Feature Engineering**")
            st.write("Extract 15 smart features: skills, salary, experience, location, etc.")
        
        with col3:
            st.markdown("**3️⃣ Ranking**")
            st.write("LightGBM LambdaRank model produces final relevance scores")
        
        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        st.write("**Test NDCG@5:** 0.7820 (78% perfect ranking accuracy)")
        st.write("**Features:** 15 engineered signals")
        st.write("**Algorithm:** LightGBM LambdaRank")


if __name__ == "__main__":
    main()
