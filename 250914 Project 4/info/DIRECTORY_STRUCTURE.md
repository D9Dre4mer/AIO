# Cấu Trúc Thư Mục Project 4 - AIO2025

## Tổng Quan
Project 4 là một hệ thống Machine Learning với giao diện wizard để huấn luyện và đánh giá các mô hình phân loại trên nhiều dataset khác nhau.

## Cấu Trúc Thư Mục Chi Tiết

```
Project 4/
├── 📁 __pycache__/                    # Python cache files
├── 📄 app.py                          # Main Streamlit application
├── 📄 main.py                         # Entry point
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
│
├── 📁 auto_train_*.py                 # Automated training scripts
│   ├── 📄 auto_train_heart_dataset.py
│   ├── 📄 auto_train_large_dataset.py
│   └── 📄 auto_train_spam_ham.py
│
├── 📁 cache/                          # Cache system for models and results
│   ├── 📁 confusion_matrices/         # Confusion matrix visualizations
│   │   ├── 📄 naive_bayes_numeric_dataset_MinMaxScaler.png
│   │   └── 📄 voting_ensemble_hard_numeric_dataset_StandardScaler.png
│   ├── 📁 models/                     # Cached trained models
│   │   ├── 📁 adaboost/
│   │   ├── 📁 catboost/
│   │   ├── 📁 decision_tree/
│   │   ├── 📁 gradient_boosting/
│   │   ├── 📁 knn/
│   │   ├── 📁 lightgbm/
│   │   ├── 📁 logistic_regression/
│   │   ├── 📁 naive_bayes/
│   │   ├── 📁 random_forest/
│   │   ├── 📁 stacking_ensemble_logistic_regression/
│   │   ├── 📁 svm/
│   │   ├── 📁 voting_ensemble_hard/
│   │   └── 📁 xgboost/
│   ├── 📁 shap/                       # SHAP analysis cache
│   └── 📁 training_results/           # Training session results
│       ├── 📁 1561379dcdb2d5f9/
│       └── 📁 2a5d47b3cc208afe/
│
├── 📁 data/                           # Dataset storage
│   ├── 📄 20250822-004129_sample-300_000Samples.csv
│   ├── 📄 2cls_spam_text_cls.csv
│   ├── 📄 archive.zip
│   ├── 📄 arxiv_dataset_backup.csv
│   ├── 📄 cache_metadata.json
│   ├── 📄 Heart_disease_cleveland_new.csv
│   ├── 📄 heart.csv
│   └── 📄 heart+disease.zip
│
├── 📁 models/                         # Model implementations
│   ├── 📄 __init__.py
│   ├── 📄 README.md
│   ├── 📄 register_models.py
│   │
│   ├── 📁 base/                       # Base model classes
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_model.py
│   │   ├── 📄 interfaces.py
│   │   └── 📄 metrics.py
│   │
│   ├── 📁 classification/             # Classification models
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
│   ├── 📁 clustering/                 # Clustering models
│   │   ├── 📄 __init__.py
│   │   └── 📄 kmeans_model.py
│   │
│   ├── 📁 ensemble/                    # Ensemble methods
│   │   ├── 📄 __init__.py
│   │   ├── 📄 ensemble_manager.py
│   │   └── 📄 stacking_classifier.py
│   │
│   └── 📁 utils/                      # Model utilities
│       ├── 📄 __init__.py
│       ├── 📄 model_factory.py
│       ├── 📄 model_registry.py
│       └── 📄 validation_manager.py
│
├── 📁 utils/                          # General utilities
│   ├── 📄 progress_tracker.py
│   └── 📄 rapids_detector.py
│
├── 📁 wizard_ui/                      # Streamlit wizard interface
│   ├── 📄 __init__.py
│   ├── 📄 core.py
│   ├── 📄 main.py
│   ├── 📄 navigation.py
│   ├── 📄 session_backup.json
│   ├── 📄 session_manager.py
│   ├── 📄 validation.py
│   │
│   ├── 📁 components/                 # UI components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 dataset_preview.py
│   │   └── 📄 file_upload.py
│   │
│   ├── 📁 steps/                      # Wizard steps
│   │   ├── 📄 __init__.py
│   │   ├── 📄 step1_dataset.py
│   │   ├── 📄 step3_optuna_stacking.py
│   │   └── 📄 step5_shap_visualization.py
│   │
│   ├── 📁 responsive/                 # Responsive design
│   │   └── 📄 __init__.py
│   │
│   └── 📁 windows/                    # Window management
│       └── 📄 __init__.py
│
├── 📁 info/                           # Documentation and resources
│   ├── 📄 README.md
│   ├── 📄 log.txt                     # Application logs
│   ├── 📄 RECENT_CHANGES_SUMMARY.md
│   ├── 📄 RECENT_UI_UPDATES.md
│   ├── 📄 STEP4_README.md
│   ├── 📄 STEP4_FLOW_DIAGRAM.md
│   ├── 📄 STEP4_QUICK_FIX_REFERENCE.md
│   ├── 📄 STEP4_DEBUGGING_GUIDE.md
│   ├── 📄 STEP4_FIX_SUMMARY.md
│   ├── 📄 STEP5_GUIDE.md
│   ├── 📄 UI_IMPROVEMENTS_SUMMARY.md
│   │
│   ├── 📁 Blog/                       # LaTeX blog documentation
│   │   ├── 📄 251004 Project 4.pdf
│   │   ├── 📄 main.tex
│   │   ├── 📄 preamble.tex
│   │   ├── 📄 references.bib
│   │   │
│   │   ├── 📁 Content/                # Blog content files
│   │   │   ├── 📄 ADVANCED_FEATURES_GUIDE.md
│   │   │   ├── 📄 AUTOMATED_TRAINING_SCRIPTS_GUIDE.md
│   │   │   ├── 📄 CACHE_PROCESSING_GUIDE.md
│   │   │   ├── 📄 DATA_PROCESSING_AND_TRAINING_METHODOLOGY.md
│   │   │   ├── 📄 MODEL_PROCESSING_GUIDE.md
│   │   │   ├── 📄 OPTUNA_SCALER_VECTORIZATION_GUIDE.md
│   │   │   ├── 📄 PIPELINE_LOGIC_DOCUMENTATION.md
│   │   │   ├── 📄 PROJECT_STRUCTURE_DOCUMENTATION.md
│   │   │   ├── 📄 PROJECT_UPGRADES_COMPARISON.md
│   │   │   ├── 📄 STEP5_VISUALIZATION_METHODOLOGY.md
│   │   │   └── 📄 WIZARD_UI_ARCHITECTURE_GUIDE.md
│   │   │
│   │   ├── 📁 Logo/                   # Project logo
│   │   │   └── 📄 Logo.png
│   │   │
│   │   ├── 📁 projects/               # LaTeX project files
│   │   │   ├── 📁 Advanced-Features/
│   │   │   ├── 📁 Conclusion/
│   │   │   ├── 📁 Future-Development/
│   │   │   ├── 📁 Model-Improvements/
│   │   │   ├── 📁 Modular-Architecture/
│   │   │   ├── 📁 Project-Evolution/
│   │   │   ├── 📁 Project-Goals/
│   │   │   ├── 📁 Results-Analysis/
│   │   │   └── 📁 Wizard-Interface/
│   │   │
│   │   ├── 📁 Result/                 # Analysis results
│   │   │   ├── 📁 cleveland_dataset/
│   │   │   │   ├── 📄 Cleveland_Model_by_Model_SHAP_Analysis.md
│   │   │   │   ├── 📄 Cleveland_SHAP_Analysis_Report.md
│   │   │   │   ├── 📁 confusion_matrices/
│   │   │   │   ├── 📁 Catboost/SHAP/
│   │   │   │   ├── 📁 DT/SHAP/
│   │   │   │   ├── 📁 GB/SHAP/
│   │   │   │   ├── 📁 LightGBM/SHAP/
│   │   │   │   ├── 📁 RF/SHAP/
│   │   │   │   └── 📁 XGBoost/SHAP/
│   │   │   └── 📁 heart_dataset/
│   │   │       ├── 📄 detailed_shap_values_analysis.md
│   │   │       ├── 📄 HEART_DATASET_KAGGLE_SOURCE.md
│   │   │       ├── 📄 Individual_Model_SHAP_Analysis.md
│   │   │       ├── 📄 SHAP_Analysis_Summary_and_Insights.md
│   │   │       ├── 📁 confusion_matrices/
│   │   │       ├── 📁 Catboost/SHAP/
│   │   │       ├── 📁 DT/SHAP/
│   │   │       ├── 📁 GB/SHAP/
│   │   │       ├── 📁 LightGBM/SHAP/
│   │   │       ├── 📁 RF/SHAP/
│   │   │       └── 📁 XGBoost/SHAP/
│   │   │
│   │   └── 📁 UI/                     # UI screenshots
│   │       ├── 📄 description.md
│   │       ├── 📄 Step 1.jpg
│   │       ├── 📄 Step 3.jpg
│   │       ├── 📄 Step 4.jpg
│   │       ├── 📄 Step 4 -2.jpg
│   │       ├── 📄 Step2-1.jpg
│   │       ├── 📄 Step2-2.jpg
│   │       ├── 📄 Step5-1.jpg
│   │       ├── 📄 Step5-2.jpg
│   │       └── 📄 Step5-3.jpg
│   │
│   ├── 📁 Current UI/                 # Current UI state
│   │
│   ├── 📁 Other/                      # Additional documentation
│   │   ├── 📄 AUTO_TRAIN_SETUP_GUIDE.md
│   │   ├── 📄 CACHE_STRUCTURE_DOCUMENTATION.md
│   │   ├── 📄 CACHE_SYSTEM_GUIDE.md
│   │   ├── 📄 COMPLETE_GUIDE.md
│   │   ├── 📄 DUPLICATE_REMOVAL_TOGGLE_IMPLEMENTATION.md
│   │   ├── 📄 Heart Data Compare.png
│   │   ├── 📄 HEART_DATASET_DOWNLOAD_URLS.md
│   │   ├── 📄 HEART_DATASET_REFERENCES.md
│   │   └── 📄 HEART_DATASETS_ANALYSIS_REPORT.md
│   │
│   ├── 📁 Plan/                       # Project planning documents
│   │   ├── 📄 adding_new_models_guide.md
│   │   ├── 📄 requirement.md
│   │   ├── 📄 requirement.pdf
│   │   └── 📄 Upgrade.md
│   │
│   └── 📁 Presentation/               # Presentation materials
│       ├── 📄 Heart Dataset.jpg
│       ├── 📄 Project 4 - Slide - V1.0.pdf
│       ├── 📄 Project 4 - Slide - V1.0.pptx
│       ├── 📄 Project 4 - Slide - V2.0.pptx
│       ├── 📄 Project 4 - Slide.pptx
│       └── 📄 Screenshot 2025-10-04 213443.jpg
│
├── 📁 Root Code/                      # Legacy code versions
│   ├── 📁 Project 4.1/
│   │   └── 📄 *.ipynb
│   └── 📁 Project 4.2/
│       ├── 📁 *.png (51 files)
│       ├── 📁 *.joblib (15 files)
│       ├── 📁 *.md (11 files)
│       └── 📁 ... (other files)
│
└── 📄 Core Python Files               # Main application files
    ├── 📄 cache_manager.py
    ├── 📄 comprehensive_evaluation.py
    ├── 📄 config.py
    ├── 📄 confusion_matrix_cache.py
    ├── 📄 data_loader.py
    ├── 📄 detailed_shap_analyzer.py
    ├── 📄 estimate_training_time.py
    ├── 📄 gpu_config_manager.py
    ├── 📄 manage_embedding_cache.py
    ├── 📄 optuna_optimizer.py
    ├── 📄 shap_cache_manager.py
    ├── 📄 text_encoders.py
    ├── 📄 training_pipeline.py
    └── 📄 visualization.py
```

## Mô Tả Các Thành Phần Chính

### 🎯 Core Application Files
- **`app.py`**: Main Streamlit application với giao diện wizard
- **`main.py`**: Entry point của ứng dụng
- **`config.py`**: Cấu hình hệ thống

### 🤖 Models Directory
- **`base/`**: Các lớp cơ sở cho tất cả models
- **`classification/`**: Implementations của các thuật toán phân loại
- **`ensemble/`**: Các phương pháp ensemble (Stacking, Voting)
- **`clustering/`**: Thuật toán clustering (K-Means)

### 🧙‍♂️ Wizard UI
- **`components/`**: Các component UI tái sử dụng
- **`steps/`**: Các bước trong wizard workflow
- **`responsive/`**: Responsive design utilities

### 💾 Cache System
- **`models/`**: Cache các model đã train
- **`shap/`**: Cache SHAP analysis results
- **`training_results/`**: Kết quả training sessions

### 📊 Data & Results
- **`data/`**: Datasets (Heart Disease, Spam/Ham, Arxiv)
- **`info/Blog/Result/`**: Kết quả phân tích chi tiết
- **`info/Presentation/`**: Materials trình bày

### 📚 Documentation
- **`info/Blog/`**: LaTeX blog documentation
- **`info/Other/`**: Guides và documentation
- **`info/Plan/`**: Project planning documents

## Workflow Chính

1. **Data Loading**: Load dataset từ `data/` directory
2. **Model Training**: Sử dụng models từ `models/` directory
3. **Caching**: Lưu kết quả vào `cache/` system
4. **Visualization**: Tạo SHAP analysis và confusion matrices
5. **Documentation**: Export results vào `info/Blog/Result/`

## Tính Năng Nổi Bật

- ✅ **Modular Architecture**: Tách biệt rõ ràng giữa models, UI, và utilities
- ✅ **Comprehensive Caching**: Cache system cho models và analysis results
- ✅ **Wizard Interface**: Giao diện step-by-step thân thiện
- ✅ **Multiple Datasets**: Hỗ trợ Heart Disease, Spam/Ham, Arxiv datasets
- ✅ **Advanced Analysis**: SHAP analysis và confusion matrix visualization
- ✅ **Documentation**: LaTeX blog với kết quả chi tiết

---
*Cập nhật lần cuối: $(date)*
*Tác giả: AIO2025 Project 4 Team*
