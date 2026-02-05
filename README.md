# Resume Matching Engine

> An AI-powered system for scoring and ranking resumes against job descriptions

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Technical Approach](#technical-approach)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Results](#evaluation-results)
- [Limitations & Future Work](#limitations--future-work)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## Overview

### Problem Statement

Our Talent Acquisition team struggles with manual resume screening:
- **Slow**: Hours spent per position reviewing resumes
- **Inconsistent**: Different reviewers have different criteria
- **Incomplete**: Qualified candidates may be overlooked

### Solution

This proof-of-concept implements an AI-powered matching engine that:
1. Takes a job description and resumes as input
2. Produces relevance scores (0.0 to 1.0) for each resume
3. Ranks candidates and explains the match quality

### Key Features

- **Semantic Matching**: Understands meaning beyond keywords
- **Skill Extraction**: Identifies technical and soft skills automatically
- **Explainable Scores**: Shows matched/missing skills for each candidate
- **Production-Ready**: Clean code, comprehensive tests, easy deployment

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd thp

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for advanced NLP)
python -m spacy download en_core_web_sm
```

### Run Demo

```bash
# Run the full demo with sample data
python main.py demo
```

### Basic Usage

```bash
# Score a single resume
python main.py match --jd data/job_descriptions/senior_ai_engineer.txt \
                     --resume data/resumes/resume_01_senior_ml_engineer.txt

# Rank all resumes
python main.py rank --jd data/job_descriptions/senior_ai_engineer.txt \
                    --resume-dir data/resumes/

# Evaluate against labeled data
python main.py evaluate --jd data/job_descriptions/senior_ai_engineer.txt \
                        --resume-dir data/resumes/ \
                        --labels data/evaluation_labels.json
```

### Python API

```python
from src.matching_engine import ResumeMatchingEngine

# Initialize engine
engine = ResumeMatchingEngine()

# Score a single resume
result = engine.score_resume(job_description, resume_text)
print(f"Score: {result.final_score:.2f}")
print(f"Matched Skills: {result.matched_skills}")
print(f"Missing Skills: {result.missing_skills}")

# Rank multiple resumes
resumes = {"candidate_1": resume1, "candidate_2": resume2}
ranked = engine.rank_resumes(job_description, resumes)
for r in ranked:
    print(f"#{r.rank}: {r.resume_id} - Score: {r.match_result.final_score:.2f}")
```

---
