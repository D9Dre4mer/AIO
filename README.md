# AIO Projects

This repository contains a collection of projects that I have either developed independently or collaborated on as part of the AIO curriculum. Each project addresses specific requirements or challenges set by the AIO program, showcasing a range of skills and technologies.

Below is a brief overview of the projects included in this repository:

## 📁 Project Overview

### **[Project 1.2]**: Chatbot with RAG System
- **Technology Stack**: Python, LangChain, Vector Database, LLM Integration
- **Key Features**: 
  - Retrieval-Augmented Generation (RAG)
  - Vector-based document search
  - Conversational AI interface
  - Document processing and indexing
- **Status**: ✅ Completed

### **[Project 2.2]**: Email Classification System
- **Technology Stack**: Python, Machine Learning, Gmail API, Streamlit, FAISS
- **Key Features**:
  - **Dual Classifier System**: KNN (FAISS) + TF-IDF
  - **Real-time Gmail Integration**: OAuth 2.0, automatic email classification
  - **Advanced Cache Management**: Priority system (corrections > original)
  - **User Corrections Handling**: Learn from user feedback
  - **FAISS Index Management**: Efficient similarity search with caching
  - **Terminal Logging**: Cache verification for debugging
  - **Multi-language Support**: Multilingual transformer models
  - **Web Interface**: Streamlit dashboard with interactive features
  - **Performance Optimization**: GPU acceleration, batch processing
- **Status**: ✅ Completed with advanced features

### **[Project 3.1]**: Topic Modeling & Text Classification
- **Technology Stack**: Python, Streamlit, scikit-learn, Transformers, FAISS
- **Key Features**:
  - **Topic Modeling**: ArXiv abstracts classification
  - **Multiple Models**: KNN, Decision Tree, Naive Bayes, SVM, Random Forest
  - **Text Vectorization**: BoW, TF-IDF, Word Embeddings
  - **Interactive Wizard**: 7-step guided workflow
  - **Ensemble Learning**: Voting and Stacking classifiers
  - **GPU Acceleration**: CUDA support for embeddings
  - **Cache Management**: Advanced caching system
  - **Real-time Visualization**: Training progress and results
- **Status**: ✅ Completed with comprehensive features

### **[Project 4]**: Comprehensive Machine Learning Platform
- **Technology Stack**: Python, Streamlit, scikit-learn, PyTorch, Optuna, GPU Acceleration
- **Key Features**:
  - **Multiple Datasets**: Heart disease, spam detection, large-scale text classification
  - **15+ ML Models**: Classification, clustering, ensemble learning
  - **Interactive Wizard**: 7-step guided workflow with real-time visualization
  - **Advanced Features**: GPU acceleration, hyperparameter optimization, ensemble learning
  - **Modular Architecture**: Extensible design for easy model addition
  - **Automated Testing**: Comprehensive testing scripts for all datasets
  - **Performance Optimization**: Memory management, caching, garbage collection
- **Status**: ✅ Completed with comprehensive features

### **[Project 5]**: Production-Ready MLOps System
- **Technology Stack**: Python, MLflow, DVC, FastAPI, Docker, Prefect, Prometheus, Grafana, Optuna
- **Key Features**:
  - **Experiment Tracking**: MLflow with Postgres backend & S3 artifact storage
  - **Data Management**: DVC with S3 remote for version control
  - **Model Serving**: FastAPI with MLflow Model Registry integration
  - **Hyperparameter Optimization**: Optuna with MLflow nested runs
  - **Orchestration**: Prefect flows for automated pipelines
  - **Monitoring**: Prometheus/Grafana with Evidently drift detection
  - **CI/CD**: GitHub Actions with canary deployment & rollback
  - **Environment Management**: conda-lock for reproducibility
  - **Streamlit UI**: Interactive dashboard with MLflow integration
  - **Data Validation**: Great Expectations and Pandera
- **Status**: ✅ Completed - Production Ready

### **[Project 6]**: Time Series Forecasting with PatchTST
- **Technology Stack**: Python, NeuralForecast, PatchTST, Optuna, scikit-learn
- **Key Features**:
  - **Advanced Model**: PatchTST (Patch-based Time Series Transformer)
  - **Stock Price Forecasting**: FPT stock price prediction for 100 days ahead
  - **Hyperparameter Optimization**: Optuna with fixed optimal parameters
  - **Post-Processing**: Linear Regression with TimeSeriesSplit for bias correction
  - **Smooth Bias Correction**: 20% smooth transition for improved reliability
  - **Performance Metrics**: 97.62% MSE improvement over baseline
  - **Visualization**: Automatic comparison plots and residual analysis
  - **Walk-Forward Validation**: TimeSeriesSplit for realistic evaluation
- **Status**: ✅ Completed with advanced forecasting techniques

## 🚀 Key Highlights

### **Project 6 - Time Series Forecasting**
This project demonstrates advanced time series forecasting techniques with PatchTST:

#### **📈 Forecasting Components**
- **PatchTST Model**: Transformer-based architecture for time series
- **Fixed Optimal Parameters**: Pre-optimized hyperparameters (no Optuna needed)
- **Post-Processing Regression**: Bias correction using Linear Regression
- **Smooth Correction**: 20% smooth transition for reliability
- **Long-term Forecasting**: 100-day ahead predictions

#### **📊 Performance Results**
- **MSE**: 15.26 (97.62% improvement over baseline)
- **RMSE**: 3.91 (84.57% improvement)
- **MAE**: 3.70 (84.45% improvement)
- **Bias**: 0.91 (96.22% reduction)
- **MAPE**: 3.54% (84.84% improvement)

#### **🔬 Technical Methods**
- **TimeSeriesSplit**: Walk-forward validation (3 folds)
- **Optuna Optimization**: Hyperparameter tuning (20 trials)
- **Bias Correction**: Learning correction formula from validation folds
- **Smooth Transition**: Balancing reliability and accuracy

### **Project 5 - Production MLOps System**
This project demonstrates production-ready MLOps practices:

#### **🔄 MLOps Components**
- **MLflow Integration**: Experiment tracking, model registry, artifact storage
- **DVC Pipeline**: Data version control with S3 remote
- **FastAPI Serving**: RESTful API with health checks and metrics
- **Prefect Orchestration**: Automated pipeline scheduling
- **Monitoring Stack**: Prometheus + Grafana dashboards
- **Drift Detection**: Evidently for data quality monitoring

#### **🚀 Deployment Features**
- **Docker Compose**: Development stack (Postgres, MinIO, MLflow)
- **CI/CD Pipeline**: GitHub Actions with automated testing
- **Canary Deployment**: Gradual rollout with rollback capability
- **Environment Management**: Conda-lock for reproducibility

#### **📊 Data & Model Management**
- **Data Validation**: Great Expectations + Pandera
- **Model Registry**: Versioned models with staging/production
- **Artifact Storage**: S3-compatible MinIO for model artifacts
- **Streamlit UI**: Interactive dashboard with MLflow integration

### **Project 4 - Comprehensive ML Platform**
This project demonstrates advanced machine learning techniques with multiple datasets and comprehensive model evaluation:

#### **🤖 AI/ML Components**
- **12 Classification Models**: KNN, Decision Tree, Naive Bayes, Logistic Regression, SVM, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Ensemble Learning**: Voting Classifier, Stacking Classifier, Ensemble Manager
- **Clustering**: K-Means with optimal K detection
- **Hyperparameter Optimization**: Optuna integration for automatic tuning

#### **📊 Multiple Datasets**
- **Heart Disease Dataset**: Cardiovascular disease prediction (~1,000 samples)
- **Spam Detection Dataset**: SMS spam/ham classification (~11,000 messages)
- **Large Text Dataset**: Large-scale text classification (300,000+ samples)

#### **🎨 Interactive Web Interface**
- **7-Step Wizard**: Guided workflow from data loading to model inference
- **Real-time Visualization**: Live training progress and performance metrics
- **Model Comparison**: Side-by-side performance analysis
- **Export Capabilities**: Results, models, and visualizations
- **Session Management**: Save and resume work sessions

#### **⚡ Performance Features**
- **GPU Acceleration**: CUDA 12.6+ support for deep learning models
- **Memory Management**: Efficient data processing and garbage collection
- **Caching System**: Advanced cache management for faster loading
- **Automated Testing**: Comprehensive testing scripts for all datasets

### **Project 3.1 - Topic Modeling & Text Classification**
This project demonstrates comprehensive text classification with ArXiv abstracts:

#### **📚 Text Processing**
- **ArXiv Dataset**: Large-scale academic paper abstracts
- **Text Vectorization**: BoW, TF-IDF, Word Embeddings
- **Preprocessing**: Text cleaning, tokenization, normalization
- **Multi-class Classification**: Scientific paper categorization

#### **🤖 Machine Learning Models**
- **Traditional ML**: KNN, Decision Tree, Naive Bayes, SVM, Random Forest
- **Ensemble Methods**: Voting and Stacking classifiers
- **Performance Optimization**: GPU acceleration for embeddings
- **Cross-validation**: Robust model evaluation

#### **🎨 User Interface**
- **Interactive Wizard**: 7-step guided workflow
- **Real-time Training**: Live progress monitoring
- **Visualization**: Confusion matrices and performance plots
- **Export Features**: Results and model artifacts

### **Project 2.2 - Email Classification System**
This project demonstrates advanced machine learning techniques for email spam classification with the following innovative features:

#### **🤖 AI/ML Components**
- **KNN Classifier**: Using FAISS for efficient similarity search
- **TF-IDF Classifier**: Baseline comparison system
- **Transformer Embeddings**: Multilingual model support
- **Cache Priority System**: Intelligent cache management

#### **📧 Gmail Integration**
- **Real-time Processing**: Automatic email classification
- **Label Management**: Custom Gmail labels (Inbox_Custom, Spam_Custom)
- **Local Storage**: Email backup and management
- **OAuth 2.0**: Secure authentication

#### **💾 Advanced Caching System**
- **Separate Caches**: Original vs. corrections datasets
- **FAISS Index Caching**: Persistent similarity search indices
- **Merged Dataset Caching**: Stable corrections dataset storage
- **Cache Priority Logic**: Corrections > Original for better accuracy

#### **🔄 User Corrections Handling**
- **Feedback Integration**: Learn from user corrections
- **Data Merging**: Combine CSV dataset with JSON corrections
- **Stable Caching**: Persistent merged dataset storage
- **Consistency Management**: Robust data validation

#### **📊 Evaluation & Monitoring**
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Performance plots and confusion matrices
- **Terminal Logging**: Real-time cache usage verification
- **Error Analysis**: Detailed performance breakdown

#### **🎨 User Interfaces**
- **Command Line Interface**: Full-featured CLI with arguments
- **Streamlit Web Interface**: Interactive dashboard
- **Real-time Updates**: Live email scanning and classification

## 📚 Documentation

Each project includes comprehensive documentation:

### **Project 2.2 Documentation**
- **📋 MIND_MAP.md**: Comprehensive system architecture overview
- **🚀 QUICK_GUIDE.md**: Fast-start guide for users
- **🏗️ ARCHITECTURE_DIAGRAM.md**: Visual system design diagrams
- **📖 README.md**: Detailed project documentation
- **💻 Source Code**: Well-documented Python modules

## 🔧 Technical Stack

### **Core Technologies**
- **Python 3.10+**: Primary programming language
- **Machine Learning**: scikit-learn, transformers, faiss-cpu, neuralforecast
- **Web Framework**: Streamlit, FastAPI for interactive interfaces
- **API Integration**: Google Gmail API, OAuth 2.0
- **Data Processing**: pandas, numpy, nltk
- **MLOps**: MLflow, DVC, Prefect, Docker
- **Time Series**: PatchTST, NeuralForecast, Optuna

### **AI/ML Libraries**
- **Transformers**: Hugging Face models for embeddings
- **FAISS**: Facebook AI Similarity Search for KNN
- **scikit-learn**: Traditional ML algorithms (TF-IDF, Naive Bayes)
- **Vector Operations**: Efficient similarity computations
- **NeuralForecast**: Time series forecasting models (PatchTST, NLinear, DLinear)
- **Optuna**: Hyperparameter optimization framework
- **MLflow**: Experiment tracking and model registry

### **Development Tools**
- **Version Control**: Git with detailed commit history, DVC for data versioning
- **Documentation**: Markdown with comprehensive guides
- **Testing**: Modular architecture, pytest, unit tests
- **Performance**: GPU acceleration and caching optimization
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions with automated pipelines
- **Monitoring**: Prometheus, Grafana, Evidently

## 🎯 Learning Outcomes

### **Technical Skills Developed**
- **Machine Learning**: KNN, TF-IDF, Transformer models, Time Series Forecasting
- **API Integration**: Gmail API, OAuth authentication, FastAPI RESTful services
- **Caching Systems**: Advanced cache management strategies
- **Performance Optimization**: GPU acceleration, batch processing
- **User Feedback Integration**: Corrections handling and learning
- **System Architecture**: Modular design and component interaction
- **MLOps**: Experiment tracking, model serving, CI/CD pipelines
- **Time Series**: PatchTST, hyperparameter optimization, bias correction

### **Software Engineering Practices**
- **Modular Architecture**: Clean separation of concerns
- **Configuration Management**: Centralized settings, environment variables
- **Error Handling**: Robust exception management
- **Logging Systems**: Comprehensive debugging support
- **Documentation**: Detailed technical documentation
- **Performance Monitoring**: Real-time system verification, Prometheus metrics
- **DevOps**: Docker containerization, CI/CD pipelines, automated testing
- **Data Versioning**: DVC for data and model version control

## 🚀 Getting Started

### **Project 4 - Comprehensive ML Platform**
```bash
# Navigate to project directory
cd "250914 Project 4"

# Install dependencies
pip install -r requirements.txt

# Launch web application
streamlit run app.py

# Run automated testing scripts
python auto_train_heart_dataset.py
python auto_train_spam_ham.py
python auto_train_large_dataset.py

# Run command line tools
python main.py
python comprehensive_evaluation.py
python training_pipeline.py
```

### **Project 3.1 - Topic Modeling**
```bash
# Navigate to project directory
cd "250814 Project 3.1"

# Install dependencies
pip install -r requirements.txt

# Launch web application
streamlit run app.py

# Run comprehensive evaluation
python comprehensive_evaluation.py

# Run training pipeline
python training_pipeline.py
```

### **Project 2.2 - Email Classification**
```bash
# Navigate to project directory
cd "250730 Project 2.2 - Team GrID034"

# Install dependencies
pip install -r requirements.txt

# Basic usage
python main.py

# Evaluate model performance
python main.py --evaluate

# Run Gmail email classifier
python main.py --run-email-classifier

# Web interface
streamlit run app.py
```

### **Project 6 - Time Series Forecasting**
```bash
# Navigate to project directory
cd "251201 Project 6"

# Install dependencies
pip install neuralforecast scikit-learn scipy pandas numpy matplotlib

# Open and run the main notebook
# patchtst_fixed_params.ipynb - Uses pre-optimized parameters
```

### **Project 5 - MLOps Production System**
```bash
# Navigate to project directory
cd "251012 Project 5"

# Setup environment
conda activate <your_env_name>
pip install -r requirements.txt

# Start infrastructure (Postgres + MinIO + MLflow)
docker compose -f infra/docker-compose.dev.yml up -d --build

# Run training with MLflow tracking
python -m src.train.main

# Start FastAPI server
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit UI
streamlit run app.py --server.port 8501

# Access services
# MLflow UI: http://localhost:5000
# FastAPI Docs: http://localhost:8000/docs
# Streamlit UI: http://localhost:8501
```

### **Project 1.2 - RAG Chatbot**
```bash
# Navigate to project directory
cd "250712 Project 1.2 - Team GrID034"

# Install dependencies
pip install -r requirements.txt

# Run the chatbot application
python rag_chatbot_app.py
```

### **Documentation Navigation**
- **📋 MIND_MAP.md**: Start here for system overview
- **🚀 QUICK_GUIDE.md**: Fast implementation guide
- **🏗️ ARCHITECTURE_DIAGRAM.md**: Visual system design
- **💻 Source Code**: Detailed implementation in Python files

## 📈 Project Status

| Project | Status | Key Features | Documentation |
|---------|--------|--------------|---------------|
| Project 1.2 | ✅ Completed | RAG System, Vector Search | README.md |
| Project 2.2 | ✅ Completed | Advanced ML, Gmail API, Cache System | Comprehensive Docs |
| Project 3.1 | ✅ Completed | Topic Modeling, Text Classification | Comprehensive Docs |
| Project 4 | ✅ Completed | Multi-dataset ML Platform, Interactive Wizard | Comprehensive Docs |
| Project 5 | ✅ Completed | MLOps, MLflow, FastAPI, Docker, CI/CD | Comprehensive Docs |
| Project 6 | ✅ Completed | Time Series Forecasting, PatchTST, 97.62% Improvement | Comprehensive Docs |

## 🤝 Collaboration

These projects were developed as part of the AIO curriculum, showcasing:
- **Independent Development**: Self-directed project implementation
- **Team Collaboration**: Group project coordination
- **Technical Innovation**: Advanced feature implementation
- **Documentation Excellence**: Comprehensive technical writing

---

Feel free to explore each project folder for detailed documentation, source code, and instructions on how to run or use the projects. Each project demonstrates different aspects of AI/ML development and software engineering best practices.