# Topic Modeling Project 🚀 **COMPREHENSIVE ML PLATFORM v5.0.0**

A comprehensive topic modeling and text classification platform featuring advanced machine learning algorithms, interactive wizard interface, and ensemble learning capabilities for document classification using ArXiv abstracts dataset.

## 📊 **PROJECT STATUS: COMPREHENSIVE ML PLATFORM v5.0.0 COMPLETED**
- **15+ model-embedding combinations** with advanced algorithms ✅
- **Interactive Wizard UI** with 7-step guided workflow ✅
- **Ensemble Learning System** with stacking and voting ✅
- **Modular Architecture v4.0.0** with extensible design ✅
- **Streamlit Web Application** with responsive interface ✅
- **Comprehensive Evaluation System** with cross-validation ✅
- **GPU Acceleration Support** for deep learning models ✅
- **Session Management** with progress tracking ✅

## 🎯 Project Overview

This comprehensive platform implements and compares various text classification approaches with advanced features:

### Core Components
- **Text Vectorization Methods**: Bag of Words (BoW), TF-IDF, and Word Embeddings
- **Machine Learning Models**: 7+ algorithms including clustering, classification, and ensemble methods
- **Interactive Wizard Interface**: 7-step guided workflow for easy model configuration
- **Ensemble Learning**: Advanced stacking and voting classifiers
- **Dataset**: ArXiv abstracts with scientific paper categories

### Advanced Features
- **GPU Acceleration**: CUDA support for deep learning models
- **Session Management**: Progress tracking and state persistence
- **Real-time Monitoring**: Training progress and performance metrics
- **Export Capabilities**: Results, models, and visualizations
- **Responsive Design**: Modern web interface with mobile support

## 🏗️ Project Structure

```
250814 Project 3.1/
├── app.py                    # Main Streamlit web application
├── config.py                 # Configuration and constants
├── data_loader.py            # Dataset loading and preprocessing
├── text_encoders.py          # Text vectorization methods
├── main.py                   # Command-line execution script
├── comprehensive_evaluation.py # Comprehensive evaluation system
├── training_pipeline.py      # Training pipeline orchestration
├── visualization.py          # Plotting and visualization functions
├── models/                   # Modular ML architecture v4.0.0
│   ├── base/                # Base classes and interfaces
│   ├── classification/      # Classification models (7 algorithms)
│   ├── clustering/          # Clustering models
│   ├── ensemble/            # Ensemble learning system
│   ├── utils/               # Utilities and managers
│   └── new_model_trainer.py # Advanced model trainer
├── wizard_ui/               # Interactive wizard interface
│   ├── core.py              # Wizard management system
│   ├── session_manager.py   # Session state management
│   ├── validation.py        # Step validation system
│   ├── navigation.py        # Navigation controller
│   ├── components/          # UI components
│   └── steps/               # Individual wizard steps
├── cache/                   # Dataset cache and backups
├── pdf/Figures/             # Generated visualizations
├── info/                    # Documentation and wireframes
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Features

### 🎯 Interactive Wizard Interface
- **7-Step Guided Workflow**: From dataset selection to model inference
- **Real-time Validation**: Input validation and error handling
- **Progress Tracking**: Visual progress indicators and step completion
- **Session Management**: Save and resume work sessions
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### 📊 Text Vectorization Methods
1. **Bag of Words (BoW)**: Simple word frequency counting
2. **TF-IDF**: Term frequency-inverse document frequency with optimization
3. **Word Embeddings**: Pre-trained sentence transformers with GPU acceleration

### 🤖 Machine Learning Models

#### Classification Models
1. **K-Nearest Neighbors (KNN)**: Instance-based learning with optimal K selection
2. **Decision Tree**: Interpretable tree-based classification with pruning
3. **Naive Bayes**: Probabilistic classifier with multiple variants
4. **Logistic Regression**: Linear classifier with regularization
5. **Linear SVM**: Support Vector Machine with linear kernel
6. **SVM**: Support Vector Machine with RBF kernel

#### Clustering Models
1. **K-Means Clustering**: Unsupervised clustering with optimal K detection

#### Ensemble Learning
1. **Stacking Classifier**: Advanced ensemble with meta-learning
2. **Voting Classifier**: Majority voting ensemble
3. **Automatic Ensemble**: Smart model combination based on performance

### 📈 Advanced Features
- **Cross-Validation**: 5-fold CV with overfitting detection
- **Hyperparameter Optimization**: Grid search and random search
- **Performance Metrics**: Accuracy, precision, recall, F1-score, confusion matrices
- **GPU Acceleration**: CUDA support for deep learning models
- **Model Persistence**: Save and load trained models
- **Export Capabilities**: Results, visualizations, and model artifacts

### 📚 Dataset
- **Source**: HuggingFace UniverseTBD/arxiv-abstracts-large
- **Categories**: astro-ph, cond-mat, cs, math, physics
- **Samples**: Configurable (1000-500,000+ abstracts)
- **Split**: Configurable train/validation/test splits
- **Preprocessing**: Text cleaning, tokenization, and normalization

## 📋 Requirements

### System Requirements
- **Python**: 3.8+ (recommended: 3.9+)
- **Memory**: 8GB+ RAM (16GB+ recommended for large datasets)
- **Storage**: 5GB+ free space for datasets and models
- **GPU**: Optional but recommended for word embeddings (CUDA 12.6+)

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies

#### Core Data Science & ML
- `numpy` >= 2.2.6 - Numerical computing
- `pandas` >= 2.3.2 - Data manipulation
- `matplotlib` >= 3.10.1 - Plotting and visualization
- `seaborn` >= 0.13.2 - Statistical visualization
- `scikit-learn` >= 1.7.1 - Machine learning algorithms
- `scipy` >= 1.16.1 - Scientific computing

#### Deep Learning & NLP (GPU-Enabled)
- `torch` >= 2.8.0+cu126 - PyTorch with CUDA 12.6 support
- `sentence-transformers` >= 5.1.0 - Pre-trained embeddings
- `transformers` >= 4.55.4 - HuggingFace transformers
- `datasets` >= 4.0.0 - Dataset loading and processing

#### Web Application
- `streamlit` >= 1.49.0 - Web interface framework
- `plotly` >= 6.3.0 - Interactive visualizations

#### Text Processing
- `nltk` >= 3.9.1 - Natural language processing
- `regex` >= 2025.7.34 - Advanced text processing

#### GPU Acceleration (Optional)
- `cupy-cuda12x` >= 13.6.0 - GPU-accelerated computing

## 🎮 Usage

### 🌐 Web Application (Recommended)

Launch the interactive Streamlit web application:

```bash
streamlit run app.py
```

This opens a modern web interface with:
- **Interactive Wizard**: 7-step guided workflow
- **Real-time Visualization**: Live training progress and results
- **Model Comparison**: Side-by-side performance analysis
- **Export Features**: Download results, models, and visualizations
- **Session Management**: Save and resume your work

### 💻 Command Line Interface

#### Run the Complete Pipeline

```bash
python main.py
```

This executes the entire topic modeling pipeline:
1. Load and explore the ArXiv dataset
2. Select and preprocess samples
3. Apply different text vectorization methods
4. Train and test multiple ML models
5. Generate confusion matrices and visualizations
6. Save results to `pdf/Figures/` directory

#### Run Comprehensive Evaluation

```bash
python comprehensive_evaluation.py
```

This runs a comprehensive evaluation of all model-embedding combinations with cross-validation.

#### Run Training Pipeline

```bash
python training_pipeline.py
```

This runs the advanced training pipeline with ensemble learning and hyperparameter optimization.

### 🧙‍♂️ Wizard Interface Guide

The interactive wizard provides a guided 7-step workflow for topic modeling:

### Step 1: Dataset Selection & Upload
- Choose between ArXiv dataset or upload custom data
- Configure dataset parameters (sample size, categories)
- Validate data format and structure

### Step 2: Data Preprocessing & Sampling
- Configure text preprocessing options
- Set sampling parameters and data splits
- Preview processed data samples

### Step 3: Column Selection & Validation
- Select text and label columns
- Validate data types and content
- Preview column statistics

### Step 4: Model Configuration & Vectorization
- Choose vectorization methods (BoW, TF-IDF, Embeddings)
- Select machine learning models
- Configure ensemble learning options

### Step 5: Training Execution & Monitoring
- Execute model training with real-time progress
- Monitor performance metrics
- Handle training errors and warnings

### Step 6: Results Analysis & Export
- Analyze model performance
- Generate visualizations and reports
- Export results and trained models

### Step 7: Text Classification & Inference
- Classify new text samples
- Test model predictions
- Save inference results

## 🔧 Configuration

Edit `config.py` to customize:
- Cache directory for datasets
- Categories to select
- Number of samples
- Model parameters
- Output directories
- GPU settings

## 📊 Output

The project generates comprehensive results and visualizations:

### 📈 Performance Metrics
- **Confusion matrices** for each model-vectorization combination
- **Model comparison plots** showing accuracy across methods
- **Performance summaries** with detailed metrics (accuracy, precision, recall, F1-score)
- **Cross-validation results** with overfitting analysis
- **Ensemble performance** comparisons

### 📁 Generated Files
- **Visualizations**: High-quality plots saved to `pdf/Figures/`
- **Model Artifacts**: Trained models saved for future use
- **Results Data**: CSV files with detailed performance metrics
- **Session Backups**: Wizard session data for resuming work
- **Dataset Backups**: Processed datasets in CSV format

### 🎯 Export Options
- **PDF Reports**: Comprehensive analysis reports
- **CSV Data**: Raw results and metrics for further analysis
- **Model Files**: Pickled models for deployment
- **Visualizations**: PNG/PDF figures for presentations

## 🔧 Configuration

Edit `config.py` to customize:
- Cache directory for datasets
- Categories to select
- Number of samples
- Model parameters
- Output directories

## 📈 Results

Typical performance across vectorization methods:
- **Bag of Words**: Moderate performance (60-75% accuracy)
- **TF-IDF**: Good performance (70-85% accuracy), especially with KNN and SVM
- **Word Embeddings**: Best performance (80-95% accuracy) across all models

## 🧠 Model Descriptions

### Classification Models

#### K-Nearest Neighbors (KNN)
- **Purpose**: Instance-based classification with optimal K selection
- **Use Case**: Text classification, interpretable predictions
- **Advantages**: Simple, no training required, interpretable, handles non-linear patterns
- **Performance**: Excellent with TF-IDF and embeddings

#### Decision Tree
- **Purpose**: Tree-based classification with pruning
- **Use Case**: Interpretable models, feature importance analysis
- **Advantages**: Interpretable, handles mixed data, no scaling needed, feature selection
- **Performance**: Good baseline, excellent interpretability

#### Naive Bayes
- **Purpose**: Probabilistic classification with multiple variants
- **Use Case**: Text classification, high-dimensional data
- **Advantages**: Fast, works well with text, handles high dimensions, probabilistic outputs
- **Performance**: Excellent for text data, especially with TF-IDF

#### Logistic Regression
- **Purpose**: Linear classifier with regularization
- **Use Case**: Binary and multiclass classification, feature importance
- **Advantages**: Fast, interpretable, handles overfitting, probabilistic outputs
- **Performance**: Good baseline, excellent with embeddings

#### Linear SVM
- **Purpose**: Support Vector Machine with linear kernel
- **Use Case**: High-dimensional text classification
- **Advantages**: Memory efficient, works well with sparse data, good generalization
- **Performance**: Excellent with TF-IDF, good with embeddings

#### SVM (RBF Kernel)
- **Purpose**: Support Vector Machine with RBF kernel
- **Use Case**: Non-linear classification problems
- **Advantages**: Handles non-linear patterns, good generalization
- **Performance**: Good with embeddings, requires careful tuning

### Clustering Models

#### K-Means Clustering
- **Purpose**: Unsupervised clustering with optimal K detection
- **Use Case**: Exploratory data analysis, baseline clustering, document grouping
- **Advantages**: Simple, fast, interpretable, automatic K selection
- **Performance**: Good for exploratory analysis, requires preprocessing

### Ensemble Learning

#### Stacking Classifier
- **Purpose**: Advanced ensemble with meta-learning
- **Use Case**: Combining multiple models for better performance
- **Advantages**: Often outperforms individual models, robust predictions
- **Performance**: Typically 5-15% improvement over best individual model

#### Voting Classifier
- **Purpose**: Majority voting ensemble
- **Use Case**: Simple ensemble approach, combining diverse models
- **Advantages**: Simple, robust, reduces overfitting
- **Performance**: Good improvement over individual models

## 🔍 Text Vectorization Methods

### Bag of Words (BoW)
- Counts word frequencies in documents
- Ignores grammar and word order
- Simple but effective for many tasks

### TF-IDF
- Considers word frequency and document rarity
- Reduces weight of common words
- Better than BoW for most applications

### Word Embeddings
- Dense vector representations capturing semantic meaning
- Pre-trained on large corpora
- Best performance but requires more computational resources

## 📁 File Descriptions

### **Core Application**
- **`app.py`**: Main Streamlit web application with wizard interface
- **`main.py`**: Command-line execution script for the complete pipeline
- **`comprehensive_evaluation.py`**: Advanced evaluation system with cross-validation
- **`training_pipeline.py`**: Training pipeline with ensemble learning

### **Data Processing**
- **`data_loader.py`**: Dataset loading, preprocessing, and text cleaning
  - Creates CSV backup files with comprehensive dataset information
  - Automatically generates `arxiv_dataset_backup.csv` when loading datasets
  - Includes ALL samples without limits for complete data export
  - Creates separate statistics files for easy analysis
- **`text_encoders.py`**: Text vectorization methods (BoW, TF-IDF, Embeddings)
- **`config.py`**: Centralized configuration and constants

### **Machine Learning Models**
- **`models/`**: Modular ML architecture with extensible design
  - **`base/`**: Abstract base classes and interfaces
  - **`classification/`**: 6 classification algorithms (KNN, Decision Tree, Naive Bayes, Logistic Regression, Linear SVM, SVM)
  - **`clustering/`**: K-Means clustering with optimal K detection
  - **`ensemble/`**: Ensemble learning with stacking and voting
  - **`utils/`**: Model factory, registry, and validation managers
  - **`new_model_trainer.py`**: Advanced model trainer with cross-validation

### **Wizard Interface**
- **`wizard_ui/`**: Interactive wizard system
  - **`core.py`**: Wizard management and step coordination
  - **`session_manager.py`**: Session state and progress tracking
  - **`validation.py`**: Input validation and error handling
  - **`navigation.py`**: Navigation controls and step transitions
  - **`components/`**: Reusable UI components
  - **`steps/`**: Individual wizard step implementations

### **Visualization & Output**
- **`visualization.py`**: Plotting and visualization functions
- **`pdf/Figures/`**: Generated visualizations and reports
- **`cache/`**: Dataset cache and backup files
  - Contains `arxiv_dataset_backup.csv` with ALL dataset samples
  - Contains `arxiv_dataset_statistics.csv` with summary metrics
  - Contains `arxiv_categories_distribution.csv` with category counts

### **Documentation**
- **`info/`**: Project documentation and wireframes
- **`README.md`**: This comprehensive documentation file
- **`requirements.txt`**: Python dependencies with version specifications

## 🚨 Important Notes

### System Requirements
- **First Run**: Downloads ArXiv dataset (~2.3M abstracts) - may take 10-15 minutes
- **Memory**: Word embeddings require 8GB+ RAM, 16GB+ recommended for large datasets
- **GPU**: CUDA 12.6+ recommended for optimal performance with embeddings
- **Storage**: 5GB+ free space for datasets, models, and results

### Performance Tips
- **Small Datasets**: Use BoW or TF-IDF for faster processing
- **Large Datasets**: Use word embeddings for better accuracy
- **GPU Available**: Automatically detected and utilized for embeddings
- **Results**: Saved to `pdf/Figures/` directory with high-quality visualizations

### Troubleshooting
- **Memory Issues**: Reduce sample size or use BoW/TF-IDF instead of embeddings
- **GPU Issues**: Install CUDA toolkit or use CPU-only mode
- **Session Issues**: Use wizard session backup to resume interrupted work
- **Model Loading**: Ensure all dependencies are installed correctly

## 🤝 Contributing

We welcome contributions to improve the platform:

### Development Areas
- **New Models**: Implement additional ML algorithms
- **Vectorization**: Add new text preprocessing methods
- **UI/UX**: Enhance wizard interface and visualizations
- **Performance**: Optimize training and inference speed
- **Datasets**: Add support for new data sources
- **Documentation**: Improve guides and tutorials

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License by Group GrID034. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

- **HuggingFace**: For the ArXiv abstracts dataset
- **Streamlit**: For the web application framework
- **scikit-learn**: For machine learning algorithms
- **PyTorch**: For deep learning capabilities
- **Sentence Transformers**: For pre-trained embeddings
