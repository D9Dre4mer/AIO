# Machine Learning Pipeline - Hướng dẫn tổng quan

## 📋 Tổng quan hệ thống

Đây là một hệ thống machine learning pipeline hoàn chỉnh với giao diện Streamlit, hỗ trợ:

- **📊 Data Processing**: Upload, preview, preprocessing
- **🤖 Model Training**: Multiple algorithms với optimization
- **📈 Analysis**: SHAP, confusion matrix, model comparison
- **💾 Caching**: Intelligent caching system
- **🎨 UI/UX**: Modern, responsive interface

## 🚀 Quick Start Guide

### 1. Khởi động ứng dụng
```bash
# Kích hoạt môi trường conda
conda activate PJ3.1

# Chạy Streamlit app
streamlit run app.py
```

### 2. Workflow cơ bản
1. **Step 1**: Upload dataset
2. **Step 2**: Chọn columns và preprocessing
3. **Step 3**: Cấu hình models và optimization
4. **Step 4**: Training với data split tùy chỉnh
5. **Step 5**: Analysis và visualization

## 📊 Step 1: Dataset Upload

### Supported Formats
- **CSV**: Comma-separated values
- **Excel**: .xlsx, .xls files
- **JSON**: JSON format

### Features
- **Auto-detection**: Tự động detect data types
- **Preview**: Xem trước data
- **Validation**: Kiểm tra data quality
- **Info display**: Thống kê cơ bản

### Tips
- Đảm bảo data clean và có header
- Kiểm tra missing values
- Xem preview trước khi tiếp tục

## 🔧 Step 2: Data Configuration

### Tab Options

#### Text Data Tab
- **Abstract Column**: Chọn cột chứa text
- **Categories Column**: Chọn cột label
- **Vectorization Methods**: TF-IDF, BoW, Word2Vec
- **Preprocessing**: Text cleaning, stopwords removal

#### Multi Input Tab
- **Input Columns**: Chọn features
- **Label Column**: Chọn target variable
- **Numeric Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Remove Duplicates**: Tùy chọn xóa duplicates

### Default Settings
- **Label Column**: Cột cuối cùng
- **Input Columns**: Tất cả cột còn lại
- **Scaling**: StandardScaler
- **Duplicates**: Giữ nguyên (không xóa)

## 🤖 Step 3: Model Configuration

### Optimization Methods

#### Optuna Optimization
- **Models**: Chọn models để optimize
- **Trials**: Số lần thử (mặc định: 50)
- **Default**: Tất cả models được chọn

#### Ensemble Learning
- **Voting**: Hard/Soft voting
- **Stacking**: Meta-learner stacking
- **Default**: Tất cả models được chọn

### Available Models
- **Tree-based**: Random Forest, XGBoost, LightGBM, CatBoost
- **Linear**: Logistic Regression, SVM, Linear SVC
- **Ensemble**: AdaBoost, Gradient Boosting
- **Traditional**: KNN, Decision Tree, Naive Bayes

## 🚀 Step 4: Training Execution

### Data Split Configuration
- **Train Ratio**: 50-90% (mặc định: 80%)
- **Validation Ratio**: 5-30% (mặc định: 10%)
- **Test Ratio**: 5-30% (mặc định: 10%)
- **Auto-validation**: Cảnh báo nếu tổng ≠ 100%

### Training Process
1. **Data Splitting**: 3-way split (train/val/test)
2. **Scaling**: Apply selected scalers
3. **Model Training**: Train với Optuna optimization
4. **Evaluation**: Test trên test set
5. **Caching**: Lưu models và metrics

### Cache System
- **Intelligent Caching**: Tự động detect changes
- **Model Storage**: Lưu trained models
- **Metrics Storage**: Lưu performance metrics
- **SHAP Samples**: Lưu samples cho SHAP analysis

## 📈 Step 5: Analysis & Visualization

### SHAP Analysis
- **Purpose**: Giải thích model predictions
- **Models**: Tree-based models only
- **Plots**: Summary, Bar, Dependence, Waterfall
- **Sample Size**: 100-10000 samples

### Confusion Matrix
- **Purpose**: Đánh giá classification performance
- **Normalization**: None, True, Pred, All
- **Threshold**: Adjustable cho binary classification
- **Report**: Classification report chi tiết

### Model Comparison
- **Purpose**: So sánh tất cả models
- **Metrics**: Accuracy, F1, Precision, Recall
- **Ranking**: Sort theo performance
- **Export**: CSV download

## 💾 Cache System

### Cache Structure
```
cache/
├── models/
│   ├── model_name/
│   │   ├── dataset_id/
│   │   │   ├── config_hash/
│   │   │   │   ├── model.joblib
│   │   │   │   ├── metrics.json
│   │   │   │   ├── config.json
│   │   │   │   └── shap_sample.pkl
├── training_results/
└── embeddings/
```

### Cache Benefits
- **Speed**: Không cần retrain models
- **Consistency**: Kết quả reproducible
- **Storage**: Efficient storage format
- **Integration**: Seamless với Step 5

## 🎨 UI/UX Features

### Responsive Design
- **Mobile-friendly**: Responsive layout
- **Modern UI**: Clean, professional interface
- **Dark/Light**: Theme support
- **Navigation**: Step-by-step workflow

### User Experience
- **Progress Tracking**: Visual progress indicators
- **Error Handling**: Clear error messages
- **Help Text**: Tooltips và guidance
- **Session Management**: Auto-save progress

## ⚙️ Configuration Files

### config.py
```python
# SHAP Configuration
SHAP_ENABLE = True
SHAP_SAMPLE_SIZE = 1000
SHAP_OUTPUT_DIR = "info/Result/"

# Cache Configuration
CACHE_DIR = "cache"
CACHE_ENABLE = True
```

### Cache Manager
- **Automatic**: Tự động manage cache
- **Validation**: Kiểm tra cache integrity
- **Cleanup**: Remove old/broken cache
- **Integration**: Seamless với training

## 🔧 Troubleshooting

### Common Issues

#### "No cached models found"
- **Cause**: Chưa hoàn thành Step 4
- **Solution**: Complete Step 4 training

#### "Training failed: Unknown error"
- **Cause**: Model training error
- **Solution**: Check data quality, try different models

#### "SHAP analysis failed"
- **Cause**: Model không tương thích
- **Solution**: Use tree-based models (Random Forest, XGBoost)

#### "Memory error"
- **Cause**: Dataset quá lớn
- **Solution**: Reduce sample size, use smaller models

### Performance Tips
1. **Data Size**: Giới hạn dataset size cho testing
2. **Sample Size**: Giảm SHAP sample size nếu chậm
3. **Model Selection**: Chọn models phù hợp với data
4. **Cache Usage**: Sử dụng cache để tránh retrain

## 📚 Advanced Usage

### Custom Models
- Thêm models mới trong `models/` directory
- Implement `BaseModel` interface
- Register trong `register_models.py`

### Custom Metrics
- Thêm metrics mới trong `models/base/metrics.py`
- Implement metric calculation
- Integrate với training pipeline

### Custom Preprocessing
- Thêm preprocessing methods
- Implement trong `data_loader.py`
- Add UI controls trong Step 2

## 🎯 Best Practices

### Data Preparation
1. **Clean Data**: Remove missing values, outliers
2. **Feature Engineering**: Create meaningful features
3. **Data Validation**: Check data quality
4. **Sample Size**: Adequate sample size cho training

### Model Selection
1. **Start Simple**: Begin với simple models
2. **Compare Models**: Test multiple algorithms
3. **Cross-validation**: Use proper validation
4. **Ensemble**: Consider ensemble methods

### Performance Optimization
1. **Cache Usage**: Leverage caching system
2. **Sample Size**: Optimize SHAP sample size
3. **Model Complexity**: Balance accuracy vs speed
4. **Resource Management**: Monitor memory usage

## 📖 Documentation

### File Structure
```
├── app.py                 # Main Streamlit app
├── cache_manager.py       # Cache management
├── data_loader.py         # Data loading utilities
├── models/               # Model implementations
├── wizard_ui/            # UI components
├── info/                 # Documentation
└── cache/                # Cache storage
```

### Key Components
- **Session Manager**: Manages user session
- **Model Factory**: Creates model instances
- **Cache Manager**: Handles caching
- **Visualization**: SHAP và plotting utilities

## 🎉 Conclusion

Hệ thống này cung cấp:
- **Complete Pipeline**: End-to-end ML workflow
- **User-friendly**: Intuitive interface
- **Powerful Analysis**: SHAP, confusion matrix, comparison
- **Efficient Caching**: Smart caching system
- **Extensible**: Easy to add new features

Sử dụng hệ thống để build và analyze machine learning models một cách hiệu quả!
