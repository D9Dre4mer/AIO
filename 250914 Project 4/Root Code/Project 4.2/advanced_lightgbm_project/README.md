# 🚀 Advanced LightGBM Optimization Project

## 📋 Overview

This project provides a comprehensive framework for advanced LightGBM optimization using cutting-edge techniques to maximize model performance. It implements multiple optimization strategies, ensemble methods, and advanced evaluation techniques.

## ✨ Key Features

### 🔧 Advanced Optimization Techniques
- **Multi-objective Hyperparameter Optimization** with Optuna
- **Bayesian Optimization** with Gaussian Processes
- **Automated Feature Engineering** with polynomial, statistical, and interaction features
- **Target Encoding** for categorical variables
- **Feature Selection** with multiple algorithms

### 🎭 Ensemble Methods
- **Voting Classifier** (Hard & Soft)
- **Stacking Classifier** with meta-learner
- **Blending Ensemble** with holdout validation
- **Weighted Ensemble** based on individual performance

### 📊 Comprehensive Evaluation
- **Advanced Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, Matthews Correlation, Cohen's Kappa
- **Statistical Significance Testing**: McNemar's, Wilcoxon, t-test
- **Cross-Validation Analysis** with multiple strategies
- **Advanced Visualizations**: ROC curves, Precision-Recall curves, Radar charts

### 🔍 Model Interpretability
- **SHAP Analysis** with TreeExplainer
- **Feature Importance** analysis
- **Waterfall Plots** for individual predictions
- **Summary Plots** for global feature impact

### ⚡ Performance Optimization
- **GPU Support** with automatic fallback
- **Memory Optimization** with data type conversion
- **Speed Optimization** with advanced parameters
- **Parallel Processing** support

## 🏗️ Project Structure

```
advanced_lightgbm_project/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── feature_engineering.py   # Advanced feature engineering
│   ├── hyperparameter_optimizer.py  # Hyperparameter optimization
│   ├── ensemble_methods.py      # Ensemble methods
│   ├── model_evaluator.py       # Model evaluation
│   └── lightgbm_advanced.py     # Advanced LightGBM implementation
├── data/                        # Dataset storage
├── models/                      # Saved models
├── results/                     # Results and outputs
├── tests/                       # Test files
├── main.py                      # Main execution script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Installation

#### Installation
```bash
# Clone or download the project
cd advanced_lightgbm_project

# Install all dependencies
pip install -r requirements.txt
```

#### Fix Dependencies (if issues occur)
```bash
# Run dependency fixer
python fix_dependencies.py
```

### 2. Configuration

Edit `config/config.yaml` to customize:
- Dataset paths
- Optimization parameters
- Model settings
- Output directories

### 3. Run the Pipeline

```bash
# Quick demo (recommended for first run)
python main.py --quick

# Complete pipeline
python main.py
```

## 📊 Usage Examples

### Basic Usage

```python
from src import AdvancedLightGBMPipeline

# Initialize pipeline
pipeline = AdvancedLightGBMPipeline("config/config.yaml")

# Run complete optimization
results = pipeline.run_complete_pipeline('fe')  # Use feature engineering dataset

# Run quick demo
results = pipeline.run_quick_demo('fe')
```

### Advanced Usage

```python
from src import DataLoader, AdvancedFeatureEngineer, HyperparameterOptimizer
from src import AdvancedLightGBM, ModelEvaluator

# Load data
data_loader = DataLoader("config/config.yaml")
X_train, y_train, X_val, y_val, X_test, y_test = data_loader.load_dataset('fe')

# Feature engineering
feature_engineer = AdvancedFeatureEngineer(config)
X_train_processed, X_val_processed, X_test_processed = feature_engineer.create_comprehensive_features(
    X_train, y_train, X_val, X_test
)

# Hyperparameter optimization
optimizer = HyperparameterOptimizer(config)
study = optimizer.optimize_with_optuna(X_train_processed, y_train, X_val_processed, y_val)

# Train model
lgb_model = AdvancedLightGBM(config, use_gpu=True)
lgb_model.train_model(X_train_processed, y_train, X_val_processed, y_val)

# Evaluate
evaluator = ModelEvaluator(config)
y_pred = lgb_model.predict(X_test_processed)
y_pred_proba = lgb_model.predict(X_test_processed, return_proba=True)
metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
```

## ⚙️ Configuration

### Key Configuration Options

```yaml
# Hyperparameter optimization
optimization:
  n_trials: 200              # Number of optimization trials
  timeout: 3600              # Timeout in seconds
  cv_folds: 5                # Cross-validation folds
  direction: "maximize"      # Optimization direction

# Feature engineering
feature_engineering:
  polynomial_degree: 2       # Polynomial feature degree
  target_encoding: true      # Enable target encoding
  statistical_features: true # Enable statistical features
  feature_selection: true    # Enable feature selection
  max_features: 50           # Maximum number of features

# Performance optimization
performance:
  use_gpu: true              # Enable GPU acceleration
  n_jobs: -1                 # Number of parallel jobs
  memory_optimization: true  # Enable memory optimization
  speed_optimization: true   # Enable speed optimization
```

## 📈 Expected Performance Improvements

Based on the advanced techniques implemented:

- **Accuracy**: 85-90% (vs baseline 83.87%)
- **F1-Score**: 84-89% (vs baseline 82.76%)
- **AUC-ROC**: 93-96% (vs baseline 92.02%)
- **Training Speed**: 2-3x faster with GPU
- **Model Interpretability**: Comprehensive SHAP analysis

## 🔧 Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Install LightGBM with GPU support
   pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
   ```

2. **Memory Issues**
   ```yaml
   # Reduce in config.yaml
   optimization:
     n_trials: 50
   feature_engineering:
     max_features: 30
   ```

3. **Slow Training**
   ```yaml
   # Enable speed optimization
   performance:
     speed_optimization: true
     use_gpu: true
   ```

### Dependencies Issues

```bash
# Install specific versions
pip install lightgbm==4.0.0
pip install optuna==3.0.0
pip install shap==0.42.0
```

## 📊 Output Files

The pipeline generates several output files:

- `results/advanced_lightgbm_model.txt` - Trained LightGBM model
- `results/ensemble_models/` - Ensemble models
- `results/evaluation_report.txt` - Comprehensive evaluation report
- `results/plots/` - Visualization plots
- `results/results_summary.json` - Results summary

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Advanced Features

### Custom Feature Engineering

```python
# Add custom features
def create_custom_features(X):
    X_custom = X.copy()
    X_custom['custom_feature'] = X['feature1'] * X['feature2']
    return X_custom

# Use in pipeline
feature_engineer.add_custom_function(create_custom_features)
```

### Custom Metrics

```python
# Add custom evaluation metric
def custom_metric(y_true, y_pred):
    return custom_calculation(y_true, y_pred)

evaluator.add_custom_metric('custom', custom_metric)
```

### Model Persistence

```python
# Save model
lgb_model.save_model('my_model.txt')

# Load model
lgb_model = AdvancedLightGBM(config)
lgb_model.load_model('my_model.txt')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LightGBM team for the excellent gradient boosting framework
- Optuna team for hyperparameter optimization
- SHAP team for model interpretability
- Scikit-learn team for machine learning tools

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the configuration examples

---

**Happy Optimizing! 🚀**
