# Cấu Trúc Dự Án - Comprehensive Machine Learning Platform

## Tổng Quan Dự Án

**Project 4** là một nền tảng Machine Learning toàn diện với khả năng xử lý nhiều loại dataset và triển khai 15+ mô hình ML khác nhau. Dự án được thiết kế với kiến trúc modular, hỗ trợ GPU acceleration và giao diện web tương tác.

### Thông Tin Cơ Bản
- **Tên dự án**: Comprehensive Machine Learning Platform
- **Công nghệ chính**: Python, Streamlit, scikit-learn, PyTorch, Optuna
- **Môi trường**: Conda environment `PJ3.1`
- **Trạng thái**: ✅ Hoàn thành với đầy đủ tính năng

---

## Cấu Trúc Cây Thư Mục Toàn Dự Án

```
250914 Project 4/
├── 📁 __pycache__/                           # Python cache files
├── 📄 app.py                                 # Streamlit web application (5,625 lines)
├── 📄 auto_train_heart_dataset.py            # Automated training for heart dataset
├── 📄 auto_train_large_dataset.py            # Automated training for large dataset
├── 📄 auto_train_spam_ham.py                 # Automated training for spam dataset
├── 📄 cache_manager.py                       # Cache management system
├── 📄 catboost_info/                         # CatBoost training information
│   ├── 📄 catboost_training.json
│   ├── 📄 learn_error.tsv
│   ├── 📄 time_left.tsv
│   ├── 📁 learn/
│   │   └── 📄 events.out.tfevents
│   └── 📁 tmp/
├── 📄 comprehensive_evaluation.py            # Comprehensive evaluation system (3,105 lines)
├── 📄 config.py                              # Project configuration
├── 📄 confusion_matrix_cache.py              # Confusion matrix caching
├── 📄 data_loader.py                         # Data loading and preprocessing (1,266 lines)
├── 📄 debug_cache/                           # Debug cache directory
├── 📄 estimate_training_time.py               # Training time estimation
├── 📄 gpu_config_manager.py                  # GPU configuration management
├── 📄 main.py                                # Main execution script (340 lines)
├── 📄 manage_embedding_cache.py              # Embedding cache management
├── 📄 optuna_optimizer.py                    # Hyperparameter optimization
├── 📄 README.md                              # Project overview documentation
├── 📄 requirements.txt                       # Python dependencies
├── 📄 shap_cache_manager.py                  # SHAP cache management
├── 📄 text_encoders.py                       # Text vectorization (377 lines)
├── 📄 training_pipeline.py                    # Training pipeline (3,046 lines)
├── 📄 visualization.py                        # Visualization functions (792 lines)
│
├── 📁 cache/                                 # Cache system
│   ├── 📁 confusion_matrices/
│   │   └── 📄 adaboost_numeric_dataset_MinMaxScaler.png
│   ├── 📁 models/                            # Cached trained models
│   │   ├── 📁 adaboost/
│   │   ├── 📁 catboost/
│   │   ├── 📁 decision_tree/
│   │   ├── 📁 gradient_boosting/
│   │   ├── 📁 knn/
│   │   ├── 📁 lightgbm/
│   │   ├── 📁 logistic_regression/
│   │   ├── 📁 naive_bayes/
│   │   ├── 📁 random_forest/
│   │   ├── 📁 svm/
│   │   ├── 📁 xgboost/
│   │   └── 📁 stacking_ensemble_logistic_regression/
│   ├── 📁 shap/                              # SHAP explanations
│   └── 📁 training_results/                  # Training results cache
│       ├── 📁 042fc2402025ef21/
│       ├── 📁 17f504f1ee84c0ff/
│       ├── 📁 1d9999e685db46c2/
│       ├── 📁 b2638be352c7c84c/
│       └── 📁 d4cfa841d59ac832/
│
├── 📁 data/                                  # Dataset files
│   ├── 📄 20250822-004129_sample-300_000Samples.csv
│   ├── 📄 2cls_spam_text_cls.csv
│   ├── 📄 archive.zip
│   ├── 📄 arxiv_dataset_backup.csv
│   ├── 📄 cache_metadata.json
│   ├── 📄 Heart_disease_cleveland_new.csv
│   ├── 📄 heart.csv
│   └── 📄 heart+disease.zip
│
│
├── 📁 models/                                # Machine learning models
│   ├── 📁 __pycache__/
│   ├── 📄 __init__.py
│   ├── 📄 README.md
│   ├── 📄 register_models.py
│   │
│   ├── 📁 base/                              # Base classes and interfaces
│   │   ├── 📁 __pycache__/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_model.py                  # Abstract base model class
│   │   ├── 📄 interfaces.py                  # Model interfaces
│   │   └── 📄 metrics.py                     # Metrics and evaluation
│   │
│   ├── 📁 classification/                     # Classification models
│   │   ├── 📁 __pycache__/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 adaboost_model.py
│   │   ├── 📄 catboost_model.py
│   │   ├── 📄 decision_tree_model.py
│   │   ├── 📄 gradient_boosting_model.py
│   │   ├── 📄 knn_model.py
│   │   ├── 📄 lightgbm_model.py
│   │   ├── 📄 linear_svc_model.py
│   │   ├── 📄 logistic_regression_model.py
│   │   ├── 📄 naive_bayes_model.py
│   │   ├── 📄 random_forest_model.py
│   │   ├── 📄 svm_model.py
│   │   └── 📄 xgboost_model.py
│   │
│   ├── 📁 clustering/                        # Clustering models
│   │   ├── 📁 __pycache__/
│   │   ├── 📄 __init__.py
│   │   └── 📄 kmeans_model.py
│   │
│   ├── 📁 ensemble/                           # Ensemble learning
│   │   ├── 📁 __pycache__/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 ensemble_manager.py
│   │   └── 📄 stacking_classifier.py
│   │
│   └── 📁 utils/                              # Model utilities
│       ├── 📁 __pycache__/
│       ├── 📄 __init__.py
│       ├── 📄 model_factory.py               # Factory pattern
│       ├── 📄 model_registry.py              # Registry pattern
│       └── 📄 validation_manager.py           # Validation management
│
│
├── 📁 Root Code/                             # Source code references
│   ├── 📁 Project 4.1/
│   │   └── 📄 [Code-Hint]-Project-4.1-Topic-Modeling-Ensemble-Learning.ipynb
│   └── 📁 Project 4.2/                       # 122 files total
│       ├── 📄 *.png                          # 51 PNG files
│       ├── 📄 *.joblib                       # 15 joblib files
│       ├── 📄 *.md                           # 11 markdown files
│       └── 📄 *.py                           # Python scripts
│
├── 📁 utils/                                 # Utility modules
│   ├── 📄 progress_tracker.py
│   └── 📄 rapids_detector.py
│
└── 📁 wizard_ui/                             # Streamlit wizard interface
    ├── 📁 __pycache__/
    ├── 📄 __init__.py
    ├── 📄 core.py                            # Core wizard management
    ├── 📄 main.py                            # Wizard main controller
    ├── 📄 navigation.py                      # Navigation logic
    ├── 📄 responsive/
    │   └── 📄 __init__.py
    ├── 📄 session_backup.json                # Session backup data
    ├── 📄 session_manager.py                 # Session management
    ├── 📄 validation.py                       # Validation logic
    ├── 📁 components/                        # UI components
    │   ├── 📁 __pycache__/
    │   ├── 📄 __init__.py
    │   ├── 📄 dataset_preview.py
    │   └── 📄 file_upload.py
    ├── 📁 steps/                             # Wizard steps
    │   ├── 📁 __pycache__/
    │   ├── 📄 __init__.py
    │   ├── 📄 step1_dataset.py
    │   ├── 📄 step3_optuna_stacking.py
    │   └── 📄 step5_shap_visualization.py
    └── 📁 windows/                           # Window components
        └── 📄 __init__.py
```

---

## Cấu Trúc Thư Mục Chi Tiết

### 📁 Thư Mục Gốc

```
250914 Project 4/
├── 📄 app.py                          # Giao diện Streamlit chính
├── 📄 main.py                         # Script thực thi chính (command line)
├── 📄 config.py                       # File cấu hình toàn dự án
├── 📄 requirements.txt                # Dependencies và thư viện
├── 📄 README.md                       # Tài liệu tổng quan dự án
├── 📄 training_pipeline.py            # Pipeline huấn luyện cho Streamlit
├── 📄 comprehensive_evaluation.py     # Hệ thống đánh giá toàn diện
├── 📄 data_loader.py                  # Module tải và xử lý dữ liệu
├── 📄 text_encoders.py                # Module mã hóa văn bản
├── 📄 visualization.py                # Module trực quan hóa
├── 📄 cache_manager.py                # Quản lý cache hệ thống
├── 📄 shap_cache_manager.py           # Quản lý cache SHAP
├── 📄 confusion_matrix_cache.py      # Cache ma trận nhầm lẫn
├── 📄 optuna_optimizer.py             # Tối ưu hóa hyperparameter
├── 📄 gpu_config_manager.py           # Quản lý cấu hình GPU
├── 📄 estimate_training_time.py       # Ước tính thời gian huấn luyện
├── 📄 manage_embedding_cache.py       # Quản lý cache embedding
└── 📄 auto_train_*.py                 # Scripts tự động huấn luyện
```

### 📁 Thư Mục Models (`models/`)

Kiến trúc modular cho các mô hình ML:

```
models/
├── 📄 __init__.py                     # Package initialization
├── 📄 register_models.py              # Đăng ký tất cả models
├── 📄 README.md                       # Tài liệu models
│
├── 📁 base/                           # Lớp cơ sở
│   ├── 📄 base_model.py               # Abstract base class
│   ├── 📄 interfaces.py               # Interfaces định nghĩa
│   └── 📄 metrics.py                  # Metrics và đánh giá
│
├── 📁 classification/                 # Models phân loại
│   ├── 📄 logistic_regression_model.py
│   ├── 📄 decision_tree_model.py
│   ├── 📄 naive_bayes_model.py
│   ├── 📄 knn_model.py
│   ├── 📄 svm_model.py
│   ├── 📄 random_forest_model.py
│   ├── 📄 adaboost_model.py
│   ├── 📄 gradient_boosting_model.py
│   ├── 📄 xgboost_model.py
│   ├── 📄 lightgbm_model.py
│   ├── 📄 catboost_model.py
│   └── 📄 linear_svc_model.py
│
├── 📁 clustering/                     # Models clustering
│   └── 📄 kmeans_model.py
│
├── 📁 ensemble/                       # Ensemble learning
│   ├── 📄 ensemble_manager.py
│   └── 📄 stacking_classifier.py
│
└── 📁 utils/                          # Utilities
    ├── 📄 model_factory.py            # Factory pattern
    ├── 📄 model_registry.py           # Registry pattern
    └── 📄 validation_manager.py       # Quản lý validation
```

### 📁 Thư Mục Wizard UI (`wizard_ui/`)

Giao diện wizard 7 bước:

```
wizard_ui/
├── 📄 __init__.py
├── 📄 core.py                         # Core wizard management
├── 📄 main.py                         # Wizard main controller
├── 📄 navigation.py                   # Điều hướng wizard
├── 📄 session_manager.py              # Quản lý session
├── 📄 validation.py                   # Validation logic
├── 📄 session_backup.json             # Backup session data
│
├── 📁 components/                     # UI components
│   ├── 📄 dataset_preview.py          # Preview dataset
│   └── 📄 file_upload.py              # Upload files
│
├── 📁 steps/                          # Các bước wizard
│   ├── 📄 step1_dataset.py            # Bước 1: Chọn dataset
│   ├── 📄 step3_optuna_stacking.py    # Bước 3: Optuna & Stacking
│   └── 📄 step5_shap_visualization.py # Bước 5: SHAP visualization
│
├── 📁 responsive/                     # Responsive design
└── 📁 windows/                        # Window components
```

### 📁 Thư Mục Data (`data/`)

```
data/
├── 📄 Heart_disease_cleveland_new.csv # Dataset bệnh tim
├── 📄 heart.csv                       # Dataset tim (backup)
├── 📄 2cls_spam_text_cls.csv          # Dataset spam detection
├── 📄 20250822-004129_sample-300_000Samples.csv # Large dataset
├── 📄 arxiv_dataset_backup.csv        # ArXiv dataset backup
├── 📄 cache_metadata.json             # Metadata cache
├── 📄 archive.zip                     # Archive files
└── 📄 heart+disease.zip               # Heart disease archive
```

### 📁 Thư Mục Cache (`cache/`)

Hệ thống cache phân cấp:

```
cache/
├── 📁 models/                         # Cached trained models
│   ├── 📁 adaboost/
│   ├── 📁 catboost/
│   ├── 📁 decision_tree/
│   ├── 📁 gradient_boosting/
│   ├── 📁 knn/
│   ├── 📁 lightgbm/
│   ├── 📁 logistic_regression/
│   ├── 📁 naive_bayes/
│   ├── 📁 random_forest/
│   ├── 📁 svm/
│   ├── 📁 xgboost/
│   └── 📁 stacking_ensemble_*/
│
├── 📁 training_results/               # Kết quả huấn luyện
│   └── 📁 [hash_folders]/             # Folders theo hash
│
├── 📁 confusion_matrices/             # Ma trận nhầm lẫn
│   └── 📄 *.png                       # Confusion matrix images
│
└── 📁 shap/                          # SHAP explanations
    └── 📁 [model_folders]/           # SHAP cho từng model
```

### 📁 Thư Mục Utils (`utils/`)

```
utils/
├── 📄 progress_tracker.py             # Theo dõi tiến trình
└── 📄 rapids_detector.py              # Phát hiện RAPIDS
```

### 📁 Thư Mục Info (`info/`)

Tài liệu và hướng dẫn:

```
info/
├── 📄 README.md                       # Tài liệu chính
├── 📄 COMPLETE_GUIDE.md              # Hướng dẫn đầy đủ
├── 📄 CACHE_SYSTEM_GUIDE.md           # Hướng dẫn cache
├── 📄 AUTO_TRAIN_SETUP_GUIDE.md       # Setup tự động
├── 📄 RECENT_CHANGES_SUMMARY.md       # Tóm tắt thay đổi
├── 📄 STEP4_README.md                 # Hướng dẫn Step 4
├── 📄 STEP5_GUIDE.md                  # Hướng dẫn Step 5
├── 📄 UI_IMPROVEMENTS_SUMMARY.md      # Cải tiến UI
├── 📄 HEART_DATASET_*.md              # Tài liệu dataset tim
├── 📄 DUPLICATE_REMOVAL_*.md          # Xử lý duplicate
├── 📄 Plan/                           # Kế hoạch phát triển
├── 📄 Blog/                           # Blog và kết quả
├── 📄 Presentation/                   # Thuyết trình
└── 📄 Current UI/                     # UI hiện tại
```

### 📁 Thư Mục PDF (`pdf/`)

```
pdf/
└── 📁 Figures/                        # Biểu đồ và hình ảnh
    └── 📄 *.pdf                       # PDF outputs
```

### 📁 Thư Mục Root Code (`Root Code/`)

```
Root Code/
├── 📁 Project 4.1/                   # Project 4.1 code
│   └── 📄 [Code-Hint]-Project-4.1-Topic-Modeling-Ensemble-Learning.ipynb
└── 📁 Project 4.2/                   # Project 4.2 code
    ├── 📄 *.png                       # Images
    ├── 📄 *.joblib                    # Model files
    ├── 📄 *.md                        # Documentation
    └── 📄 *.py                        # Python scripts
```

---

## Kiến Trúc Hệ Thống

### 🏗️ Kiến Trúc Modular

Dự án sử dụng kiến trúc modular với các thành phần chính:

1. **Base Layer** (`models/base/`)
   - `BaseModel`: Abstract base class cho tất cả models
   - `ModelInterface`: Interface định nghĩa contract
   - `ModelMetrics`: Metrics và đánh giá performance

2. **Model Layer** (`models/classification/`, `models/clustering/`, `models/ensemble/`)
   - Các implementation cụ thể của models
   - Mỗi model kế thừa từ `BaseModel`
   - Hỗ trợ GPU acceleration và optimization

3. **Utility Layer** (`models/utils/`)
   - `ModelFactory`: Factory pattern để tạo models
   - `ModelRegistry`: Registry pattern để quản lý models
   - `ValidationManager`: Quản lý validation và cross-validation

### 🔄 Data Flow

```
Data Input → DataLoader → TextVectorizer → Models → Evaluation → Visualization
     ↓           ↓            ↓           ↓         ↓           ↓
  Raw Data → Preprocessed → Vectorized → Trained → Metrics → Charts/Reports
```

### 🎯 Core Components

1. **DataLoader** (`data_loader.py`)
   - Tải và xử lý datasets
   - Dynamic category detection
   - Preprocessing và cleaning

2. **TextVectorizer** (`text_encoders.py`)
   - Bag of Words (BoW)
   - TF-IDF
   - Word Embeddings (sentence-transformers)

3. **Model Training** (`models/`)
   - 12+ classification models
   - Ensemble learning (Voting, Stacking)
   - Clustering (K-Means)

4. **Evaluation** (`comprehensive_evaluation.py`)
   - Cross-validation
   - Performance metrics
   - Overfitting detection

5. **Visualization** (`visualization.py`)
   - Confusion matrices
   - Performance charts
   - Model comparison

---

## Tính Năng Chính

### 🤖 Machine Learning Models

**Classification Models (12 models):**
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Linear SVC

**Ensemble Learning:**
- Voting Classifier (Hard/Soft)
- Stacking Classifier
- Ensemble Manager

**Clustering:**
- K-Means với optimal K detection

### 📊 Datasets Hỗ Trợ

1. **Heart Disease Dataset** (~1,000 samples)
   - Cardiovascular disease prediction
   - Numerical features

2. **Spam Detection Dataset** (~11,000 messages)
   - SMS spam/ham classification
   - Text features

3. **Large Text Dataset** (300,000+ samples)
   - Large-scale text classification
   - ArXiv abstracts

### 🎨 Interactive Web Interface

**7-Step Wizard:**
1. Dataset Selection & Upload
2. Data Preprocessing & Sampling
3. Model Selection & Configuration
4. Training & Validation
5. Evaluation & Metrics
6. Visualization & Analysis
7. Export & Save Results

**Features:**
- Real-time progress tracking
- Interactive visualizations
- Session management
- Export capabilities
- Responsive design

### ⚡ Performance Features

**GPU Acceleration:**
- CUDA 12.6+ support
- PyTorch GPU integration
- RAPIDS cuML support
- Automatic device detection

**Memory Management:**
- Efficient data processing
- Garbage collection
- Sparse matrix support
- Batch processing

**Caching System:**
- Model caching
- Training results caching
- SHAP explanations caching
- Confusion matrix caching

**Optimization:**
- Hyperparameter optimization (Optuna)
- Cross-validation
- Early stopping
- Multi-threading support

---

## Cấu Hình và Dependencies

### 🔧 Core Dependencies

```python
# Core Data Science & Machine Learning
numpy>=1.26.4
pandas>=2.3.3
matplotlib>=3.7.2
seaborn>=0.12.2
scikit-learn>=1.7.2
scipy>=1.16.2

# Deep Learning & NLP (GPU-Enabled - CUDA 12.6)
torch>=2.8.0+cu126
torchvision>=0.23.0+cu126
torchaudio>=2.8.0+cu126
sentence-transformers>=5.1.0
transformers>=4.55.4

# GPU Acceleration Libraries
cupy-cuda12x>=13.6.0
faiss-gpu>=1.8.0
faiss-cpu>=1.12.0
cuml-cpu>=24.08.0

# Enhanced ML Models & Optimization
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.4.0
shap>=0.42.0

# Web Application Framework
streamlit>=1.49.0
```

### ⚙️ Configuration (`config.py`)

**Key Settings:**
- Cache directories và paths
- Model parameters và thresholds
- GPU optimization settings
- RAPIDS cuML configuration
- Optuna optimization settings
- SHAP configuration
- Data processing settings

---

## Cách Sử Dụng

### 🚀 Quick Start

```bash
# Kích hoạt môi trường conda
conda activate PJ3.1

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy giao diện web
streamlit run app.py

# Hoặc chạy command line
python main.py
```

### 📋 Automated Testing

```bash
# Test với Heart Disease dataset
python auto_train_heart_dataset.py

# Test với Spam Detection dataset
python auto_train_spam_ham.py

# Test với Large dataset
python auto_train_large_dataset.py
```

### 🔧 Advanced Usage

```bash
# Comprehensive evaluation
python comprehensive_evaluation.py

# Training pipeline
python training_pipeline.py

# Estimate training time
python estimate_training_time.py
```

---

## Tài Liệu Bổ Sung

### 📚 Documentation Files

- `README.md`: Tổng quan dự án
- `info/COMPLETE_GUIDE.md`: Hướng dẫn đầy đủ
- `info/CACHE_SYSTEM_GUIDE.md`: Hướng dẫn cache system
- `info/AUTO_TRAIN_SETUP_GUIDE.md`: Setup tự động
- `models/README.md`: Tài liệu models

### 🎯 Key Features Documentation

- **Modular Architecture**: Dễ dàng thêm models mới
- **GPU Acceleration**: Hỗ trợ CUDA và RAPIDS
- **Caching System**: Tối ưu performance
- **Interactive UI**: Wizard interface thân thiện
- **Comprehensive Evaluation**: Đánh giá toàn diện
- **Export Capabilities**: Xuất kết quả và models

---

## Kết Luận

Dự án **Comprehensive Machine Learning Platform** là một nền tảng ML toàn diện với:

- ✅ **Kiến trúc modular** dễ mở rộng
- ✅ **15+ ML models** đa dạng
- ✅ **3 datasets** khác nhau
- ✅ **GPU acceleration** hiệu quả
- ✅ **Interactive web interface** thân thiện
- ✅ **Comprehensive evaluation** chuyên nghiệp
- ✅ **Caching system** tối ưu performance
- ✅ **Documentation** đầy đủ và chi tiết

Dự án thể hiện khả năng phát triển phần mềm ML chuyên nghiệp với best practices về architecture, performance optimization, và user experience.

---

*Tài liệu được tạo tự động từ phân tích cấu trúc dự án - Cập nhật: 2025-01-27*
