# AIO Projects Portfolio

A curated portfolio of AI/ML projects completed during the AIO learning journey, including both individual and team-based work.  
Each project is self-contained in its own folder with implementation code and project-level documentation.

---

## Table of Contents

- [Repository Goals](#repository-goals)
- [Project Index](#project-index)
- [Detailed Project Descriptions](#detailed-project-descriptions)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Run Examples by Project](#run-examples-by-project)
- [Documentation Rules](#documentation-rules)
- [Environment Notes](#environment-notes)
- [License](#license)

---

## Repository Goals

- Track project progress along a clear learning timeline.
- Demonstrate growth from core ML/NLP to MLOps and advanced video action recognition.
- Provide reusable references for experimentation, evaluation, and deployment.

---

## Project Index

- `250712 Project 1.2 - Team GrID034`
- `250717 Project 2.2`
- `250730 Project 2.2 - Team GrID034`
- `250814 Project 3.1`
- `250818 Project 3.1 - Collab`
- `250914 Project 4`
- `251012 Project 5`
- `251201 Project 6`
- `260101 Project 7`

---

## Detailed Project Descriptions

### 1) Project 1.2 - RAG Chatbot
**Folder:** `250712 Project 1.2 - Team GrID034`

**Problem**
- Build a chatbot that answers from internal documents, not only from the model's pre-trained knowledge.

**Core Techniques**
- Retrieval-Augmented Generation (RAG) with chunking, embedding, and vector retrieval.
- Context injection from retrieved passages into LLM prompts.
- Document ingestion and indexing pipeline.

**Deliverables**
- A document-grounded conversational assistant.
- Reduced hallucination risk compared to non-retrieval chatbot workflows.

---

### 2) Project 2.2 - Email Classification (Individual)
**Folder:** `250717 Project 2.2`

**Problem**
- Classify incoming emails (e.g., spam/ham or custom categories) to reduce manual filtering work.

**Core Techniques**
- Text vectorization and classical ML classifiers.
- Structured training and evaluation flow for model comparison.
- Flexible setup for testing preprocessing and feature engineering strategies.

**Deliverables**
- Baseline email classification pipeline.
- Individual benchmark for comparing with team implementation.

---

### 3) Project 2.2 - Email Classification (Team)
**Folder:** `250730 Project 2.2 - Team GrID034`

**Problem**
- Build a practical, end-to-end email classifier integrated with real Gmail workflows.

**Core Techniques**
- Dual-classifier design: KNN (FAISS) + TF-IDF.
- Gmail API integration with OAuth 2.0.
- Priority caching strategy (`corrections > original`) to learn from user feedback.
- Streamlit UI for monitoring, correction, and evaluation.

**Deliverables**
- End-to-end system (CLI + web app).
- Feedback-aware classification behavior with improved operational efficiency.

---

### 4) Project 3.1 - Topic Modeling & Text Classification
**Folder:** `250814 Project 3.1`

**Problem**
- Analyze and classify academic abstracts (ArXiv-style) across multiple scientific topics.

**Core Techniques**
- Multiple feature representations: BoW, TF-IDF, and embeddings.
- Model suite: KNN, Decision Tree, Naive Bayes, SVM, Random Forest.
- Ensemble methods (voting, stacking) for robustness.
- Guided multi-step workflow via Streamlit.

**Deliverables**
- Full text-classification experimentation toolkit.
- Repeatable process for model selection based on evaluation criteria.

---

### 5) Project 3.1 - Collaborative Extension
**Folder:** `250818 Project 3.1 - Collab`

**Problem**
- Extend Project 3.1 for team collaboration and shared experimentation.

**Core Techniques**
- Standardized training/evaluation flow across team members.
- Modularized structure for easier feature/model expansion.
- Improved reproducibility through clearer project organization.

**Deliverables**
- Team-ready variant for parallel development and result comparison.
- Better handoff and reviewability across contributors.

---

### 6) Project 4 - Comprehensive Machine Learning Platform
**Folder:** `250914 Project 4`

**Problem**
- Build an all-in-one ML platform for multi-dataset training, evaluation, and inference.

**Core Techniques**
- Broad model coverage: classification, clustering, ensemble learning.
- 7-step guided workflow from data loading to inference.
- Hyperparameter optimization and experiment management.
- Automation scripts for repeatable training and evaluation.

**Deliverables**
- Centralized ML experimentation platform.
- Faster benchmarking through structured workflow and visual feedback.

---

### 7) Project 5 - Production-Ready MLOps System
**Folder:** `251012 Project 5`

**Problem**
- Promote ML workflows from notebook-level experimentation to deployment-ready operations.

**Core Techniques**
- MLflow for experiment tracking and model registry.
- DVC for data/model versioning.
- FastAPI for serving and Streamlit for user-facing dashboards.
- Docker Compose local infrastructure (e.g., Postgres, MinIO, MLflow).
- Monitoring stack with Prometheus/Grafana (plus drift monitoring if configured).

**Deliverables**
- End-to-end MLOps architecture with reproducibility and traceability.
- Practical deployment baseline for production-oriented ML development.

---

### 8) Project 6 - Time Series Forecasting with PatchTST
**Folder:** `251201 Project 6`

**Problem**
- Forecast stock price trajectories (FPT) in a multi-step time-series setting.

**Core Techniques**
- PatchTST model for long-horizon forecasting.
- Walk-forward validation with `TimeSeriesSplit`.
- Linear-regression post-processing for bias correction.
- Metric-driven evaluation and visual comparative analysis.

**Deliverables**
- End-to-end forecasting pipeline with realistic validation setup.
- Strong metric improvements over baseline (as documented in project files).

---

### 9) Project 7 - SOTA Video Action Recognition Pipeline
**Folder:** `260101 Project 7`

**Problem**
- Recognize actions from videos with a high-performance, competition-ready training pipeline.

**Core Techniques**
- ViT/VideoMAE backbones with EMA, CutMix, Focal Loss, Progressive Resize, and TTA.
- Multi-model training and weighted ensembling.
- Advanced Alpha variants:
  - Alpha 1: two-phase training with reserved heads for weak classes.
  - Alpha 2: CatBoost meta-decision routing for prediction strategy selection.
- Optional object-focused preprocessing flow using detection-based cropping.

**Deliverables**
- Flexible, high-performance video classification framework.
- Scalable setup for leaderboard-focused experiments and large-scale benchmarks.

---

## Repository Structure

```text
AIO/
|-- 250712 Project 1.2 - Team GrID034/
|-- 250717 Project 2.2/
|-- 250730 Project 2.2 - Team GrID034/
|-- 250814 Project 3.1/
|-- 250818 Project 3.1 - Collab/
|-- 250914 Project 4/
|-- 251012 Project 5/
|-- 251201 Project 6/
|-- 260101 Project 7/
|-- .gitignore
|-- LICENSE
`-- README.md
```

---

## Technology Stack

**Core Languages & Frameworks**
- Python
- Streamlit, FastAPI

**Machine Learning & Data**
- scikit-learn, Transformers, FAISS
- NeuralForecast, PatchTST
- PyTorch, timm, CatBoost

**MLOps & Engineering**
- MLflow, DVC, Prefect
- Docker, GitHub Actions
- Prometheus, Grafana

---

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd AIO
   ```
2. Go to the target project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Follow that project's local `README.md` for exact run instructions and configs.

---

## Run Examples by Project

### Project 1.2 (RAG Chatbot)
```bash
cd "250712 Project 1.2 - Team GrID034"
pip install -r requirements.txt
python rag_chatbot_app.py
```

### Project 2.2 (Email Classification)
```bash
cd "250730 Project 2.2 - Team GrID034"
pip install -r requirements.txt
python main.py
streamlit run app.py
```

### Project 3.1 (Topic Modeling)
```bash
cd "250814 Project 3.1"
pip install -r requirements.txt
streamlit run app.py
```

### Project 4 (Comprehensive ML Platform)
```bash
cd "250914 Project 4"
pip install -r requirements.txt
streamlit run app.py
```

### Project 5 (MLOps)
```bash
cd "251012 Project 5"
pip install -r requirements.txt
docker compose -f infra/docker-compose.dev.yml up -d --build
```

### Project 6 (Time Series Forecasting)
```bash
cd "251201 Project 6"
pip install neuralforecast scikit-learn scipy pandas numpy matplotlib
```

### Project 7 (Video Action Recognition)
```bash
cd "260101 Project 7"
pip install -r requirements.txt
python train.py
```

---

## Documentation Rules

- Each project folder has its own `README.md` and should be treated as the source of truth.
- If any details differ between this root README and project-level docs, prioritize project-level documentation.
- Project-specific metrics, model versions, and experiment settings are maintained within each project folder.

---

## Environment Notes

- Recommended: Python 3.10+.
- Use a dedicated virtual environment per project.
- For GPU projects, ensure compatible CUDA and drivers are installed.

---

## License

This repository is distributed under the terms defined in `LICENSE`.
