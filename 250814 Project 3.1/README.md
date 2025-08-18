# Topic Modeling Project

A comprehensive topic modeling project that demonstrates different text vectorization methods and machine learning algorithms for document classification using ArXiv abstracts dataset.

## 🎯 Project Overview

This project implements and compares various text classification approaches:
- **Text Vectorization Methods**: Bag of Words (BoW), TF-IDF, and Word Embeddings
- **Machine Learning Models**: K-Means Clustering, K-Nearest Neighbors, Decision Tree, and Naive Bayes
- **Dataset**: ArXiv abstracts with scientific paper categories

## 🏗️ Project Structure

```
250814 Project 3.1/
├── config.py              # Configuration and constants
├── data_loader.py         # Dataset loading and preprocessing
├── text_encoders.py       # Text vectorization methods
├── models.py              # Machine learning models
├── visualization.py       # Plotting and visualization functions
├── main.py               # Main execution script
├── demo.py               # Demo script for text encoders
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Features

### Text Vectorization Methods
1. **Bag of Words (BoW)**: Simple word frequency counting
2. **TF-IDF**: Term frequency-inverse document frequency
3. **Word Embeddings**: Pre-trained sentence transformers

### Machine Learning Models
1. **K-Means Clustering**: Unsupervised clustering approach
2. **K-Nearest Neighbors (KNN)**: Instance-based learning
3. **Decision Tree**: Interpretable tree-based classification
4. **Naive Bayes**: Probabilistic classifier

### Dataset
- **Source**: HuggingFace UniverseTBD/arxiv-abstracts-large
- **Categories**: astro-ph, cond-mat, cs, math, physics
- **Samples**: 1000 abstracts with single labels
- **Split**: 80% training, 20% testing

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Key Dependencies
- `numpy` >= 1.21.0
- `matplotlib` >= 3.5.0
- `seaborn` >= 0.11.0
- `scikit-learn` >= 1.0.0
- `datasets` >= 2.0.0
- `sentence-transformers` >= 2.0.0
- `torch` >= 1.9.0

## 🎮 Usage

### Run the Complete Pipeline

```bash
python main.py
```

This will execute the entire topic modeling pipeline:
1. Load and explore the ArXiv dataset
2. Select and preprocess samples
3. Apply different text vectorization methods
4. Train and test multiple ML models
5. Generate confusion matrices and visualizations
6. Save results to `pdf/Figures/` directory

### Run Text Encoders Demo

```bash
python demo.py
```

This demonstrates the different text encoding methods with example documents.

## 📊 Output

The project generates:
- **Confusion matrices** for each model-vectorization combination
- **Model comparison plots** showing accuracy across methods
- **Performance summaries** with detailed metrics
- **Saved figures** in PDF format

## 🔧 Configuration

Edit `config.py` to customize:
- Cache directory for datasets
- Categories to select
- Number of samples
- Model parameters
- Output directories

## 📈 Results

Typical performance across vectorization methods:
- **Bag of Words**: Moderate performance
- **TF-IDF**: Good performance, especially with KNN
- **Word Embeddings**: Best performance across all models

## 🧠 Model Descriptions

### K-Means Clustering
- **Purpose**: Unsupervised clustering
- **Use Case**: Exploratory data analysis, baseline clustering
- **Advantages**: Simple, fast, interpretable

### K-Nearest Neighbors (KNN)
- **Purpose**: Instance-based classification
- **Use Case**: Text classification, interpretable predictions
- **Advantages**: Simple, no training required, interpretable

### Decision Tree
- **Purpose**: Tree-based classification
- **Use Case**: Interpretable models, mixed data types
- **Advantages**: Interpretable, handles mixed data, no scaling needed

### Naive Bayes
- **Purpose**: Probabilistic classification
- **Use Case**: Text classification, high-dimensional data
- **Advantages**: Fast, works well with text, handles high dimensions

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

### **Core Modules**
- **`main.py`**: Main execution script orchestrating the entire pipeline
- **`data_loader.py`**: Handles dataset loading, preprocessing, and text cleaning
  - **NEW!** Now creates CSV backup files with comprehensive dataset information
  - Automatically generates `arxiv_dataset_backup.csv` when loading datasets
  - Includes ALL samples without limits for complete data export
  - Creates separate statistics files for easy analysis
- **`text_encoders.py`**: Implements different text vectorization methods
- **`models.py`**: Handles different machine learning models and their training/testing
- **`visualization.py`**: Handles plotting and visualization functions
- **`config.py`**: Centralized configuration and constants

### **Configuration & Data**
- **`config.py`**: Project configuration (cache directory, sample limits, etc.)
- **`requirements.txt`**: Python dependencies for the project
- **`cache/`**: Directory for storing downloaded datasets and CSV backups
  - **NEW!** Contains `arxiv_dataset_backup.csv` with ALL dataset samples
  - **NEW!** Contains `arxiv_dataset_statistics.csv` with summary metrics
  - **NEW!** Contains `arxiv_categories_distribution.csv` with category counts

### **Execution Scripts**
- **`run_project.bat`**: Windows batch script to run the project
- **`run_project.ps1`**: PowerShell script to run the project
- **`demo.py`**: Demo script for testing individual components
- **`test_modules.py`**: Unit tests for project modules

## 🚨 Notes

- The first run will download the ArXiv dataset (~2.3M abstracts)
- Word embeddings require significant computational resources
- Results are saved to `pdf/Figures/` directory
- GPU acceleration is automatically detected for embeddings

## 🤝 Contributing

Feel free to:
- Add new text vectorization methods
- Implement additional ML models
- Improve visualization functions
- Optimize performance
- Add new datasets

## 📄 License

This project is for educational and research purposes.
