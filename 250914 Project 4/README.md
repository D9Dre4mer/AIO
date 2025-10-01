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

## 🚀 Key Highlights

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

### **Project 4 Documentation**
- **📋 README.md**: Comprehensive project overview and usage guide
- **🏗️ Modular Architecture**: Well-structured codebase with clear separation of concerns
- **💻 Source Code**: Well-documented Python modules with extensive comments
- **🧪 Testing Scripts**: Automated testing for all datasets and models

### **Project 2.2 Documentation**
- **📋 MIND_MAP.md**: Comprehensive system architecture overview
- **🚀 QUICK_GUIDE.md**: Fast-start guide for users
- **🏗️ ARCHITECTURE_DIAGRAM.md**: Visual system design diagrams
- **📖 README.md**: Detailed project documentation
- **💻 Source Code**: Well-documented Python modules

## 🔧 Technical Stack

### **Core Technologies**
- **Python 3.8+**: Primary programming language
- **Machine Learning**: scikit-learn, transformers, faiss-cpu, PyTorch
- **Web Framework**: Streamlit for interactive interfaces
- **API Integration**: Google Gmail API, OAuth 2.0
- **Data Processing**: pandas, numpy, nltk
- **Optimization**: Optuna for hyperparameter tuning

### **AI/ML Libraries**
- **Transformers**: Hugging Face models for embeddings
- **FAISS**: Facebook AI Similarity Search for KNN
- **scikit-learn**: Traditional ML algorithms (TF-IDF, Naive Bayes, SVM, etc.)
- **PyTorch**: Deep learning capabilities with GPU acceleration
- **Ensemble Learning**: Voting, Stacking, and advanced ensemble methods
- **Vector Operations**: Efficient similarity computations

### **Development Tools**
- **Version Control**: Git with detailed commit history
- **Documentation**: Markdown with comprehensive guides
- **Testing**: Modular architecture for easy testing
- **Performance**: GPU acceleration and caching optimization

## 🎯 Learning Outcomes

### **Technical Skills Developed**
- **Machine Learning**: KNN, TF-IDF, Transformer models, Ensemble learning
- **API Integration**: Gmail API, OAuth authentication
- **Caching Systems**: Advanced cache management strategies
- **Performance Optimization**: GPU acceleration, batch processing
- **User Feedback Integration**: Corrections handling and learning
- **System Architecture**: Modular design and component interaction
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Multi-dataset Handling**: Comprehensive data processing pipelines

### **Software Engineering Practices**
- **Modular Architecture**: Clean separation of concerns
- **Configuration Management**: Centralized settings
- **Error Handling**: Robust exception management
- **Logging Systems**: Comprehensive debugging support
- **Documentation**: Detailed technical documentation
- **Performance Monitoring**: Real-time system verification
- **Testing Automation**: Comprehensive testing frameworks

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

### **Documentation Navigation**
- **📋 README.md**: Start here for project overview
- **🚀 Quick Start**: Fast implementation guide
- **🏗️ Architecture**: System design and structure
- **💻 Source Code**: Detailed implementation in Python files

## 📈 Project Status

| Project | Status | Key Features | Documentation |
|---------|--------|--------------|---------------|
| Project 1.2 | ✅ Completed | RAG System, Vector Search | README.md |
| Project 2.2 | ✅ Completed | Advanced ML, Gmail API, Cache System | Comprehensive Docs |
| Project 4 | ✅ Completed | Multi-dataset ML Platform, Interactive Wizard | Comprehensive Docs |

## 🤝 Collaboration

These projects were developed as part of the AIO curriculum, showcasing:
- **Independent Development**: Self-directed project implementation
- **Team Collaboration**: Group project coordination
- **Technical Innovation**: Advanced feature implementation
- **Documentation Excellence**: Comprehensive technical writing
- **Multi-domain Expertise**: Text classification, numerical analysis, ensemble learning

---

Feel free to explore each project folder for detailed documentation, source code, and instructions on how to run or use the projects. Each project demonstrates different aspects of AI/ML development and software engineering best practices.