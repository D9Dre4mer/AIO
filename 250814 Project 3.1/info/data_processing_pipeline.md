# 🔄 Pipeline Xử lý Dữ liệu - Topic Modeling Project

## 🎯 Tổng quan Pipeline

Dự án Topic Modeling sử dụng một pipeline xử lý dữ liệu hoàn chỉnh từ việc chọn dataset đến việc chọn ra kết hợp tốt nhất giữa mô hình ML và phương pháp vectorization. Pipeline được thiết kế theo kiến trúc modular mới (v4.0.0) với khả năng mở rộng và tối ưu hóa.

---

## 📊 **PHASE 1: Data Loading & Preparation**

### 1.1 Dataset Selection & Loading
```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET LOADING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📥 Source: HuggingFace UniverseTBD/arxiv-abstracts-large       │
│ 📊 Size: 2.3M+ scientific paper abstracts                     │
│ 🏷️  Categories: astro-ph, cond-mat, cs, math, physics         │
│ 💾 Cache: Local cache để tránh download lại                    │
│                                                                 │
│ 🔄 Process:                                                     │
│   1. Check local cache                                         │
│   2. Download if not exists                                    │
│   3. Create CSV backup (optional)                              │
│   4. Load into memory                                          │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**
```python
# data_loader.py
def load_dataset(self):
    # Load từ HuggingFace với cache
    self.dataset = load_dataset("UniverseTBD/arxiv-abstracts-large", 
                               cache_dir=self.cache_dir)
    
    # Tạo CSV backup (optional)
    self._create_csv_backup_chunked(csv_backup_path)
```

### 1.2 Data Sampling & Preprocessing
```
┌─────────────────────────────────────────────────────────────────┐
│                DATA SAMPLING & PREPROCESSING                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📊 Sampling Strategy:                                           │
│   • Default: 100,000 samples (cân bằng giữa speed & quality)  │
│   • User configurable: Input từ user                          │
│   • Stratified: Đảm bảo cân bằng categories                   │
│                                                                 │
│ 🔧 Preprocessing Steps:                                         │
│   1. Text cleaning (remove special chars, normalize)           │
│   2. Category mapping (convert to numeric labels)              │
│   3. Data validation (check quality, remove nulls)            │
│   4. Memory optimization (reduce memory footprint)             │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**
```python
# data_loader.py
def select_samples(self, max_samples: int = MAX_SAMPLES):
    # Stratified sampling để đảm bảo cân bằng categories
    self.samples = self._stratified_sample(max_samples)

def preprocess_samples(self):
    # Text cleaning và normalization
    self.preprocessed_samples = self._clean_texts(self.samples)
    
    # Category mapping
    self._create_label_mappings()
```

### 1.3 Data Splitting Strategy
```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SPLITTING STRATEGY                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📊 3-Way Split:                                                │
│   • Training: 60% (for model training)                         │
│   • Validation: 20% (for hyperparameter tuning)                │
│   • Test: 20% (for final evaluation)                           │
│                                                                 │
│ 🔄 Cross-Validation:                                           │
│   • CV Folds: 5 (configurable)                                 │
│   • Stratified: Đảm bảo cân bằng labels                       │
│   • Random State: 42 (reproducible)                            │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**
```python
# models/validation_manager.py
def split_data(self, X, y):
    # 3-way split: Train -> Val -> Test
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=self.random_state, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
    )
    
    return X_train_full, X_val, X_test, y_train_full, y_val, y_test
```

---

## 🔤 **PHASE 2: Text Vectorization**

### 2.1 Multiple Vectorization Methods
```
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT VECTORIZATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📚 Method 1: Bag of Words (BoW)                                │
│   • Approach: Word frequency counting                          │
│   • Pros: Simple, fast, interpretable                          │
│   • Cons: Loses semantic meaning, high dimensionality          │
│   • Use Case: Baseline comparison                              │
│                                                                 │
│ 📚 Method 2: TF-IDF                                            │
│   • Approach: Term frequency-inverse document frequency        │
│   • Pros: Better than BoW, handles rare words                  │
│   • Cons: Still loses semantic meaning                         │
│   • Use Case: Improved baseline                                │
│                                                                 │
│ 📚 Method 3: Word Embeddings                                   │
│   • Approach: Pre-trained sentence transformers                 │
│   • Pros: Semantic understanding, lower dimensionality         │
│   • Cons: Slower, requires more memory                         │
│   • Use Case: Best semantic representation                      │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**
```python
# text_encoders.py
def fit_transform_bow(self, texts):
    # Bag of Words vectorization
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    return vectorizer.fit_transform(texts)

def fit_transform_tfidf(self, texts):
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    return vectorizer.fit_transform(texts)

def transform_embeddings(self, texts):
    # Sentence transformers for semantic embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
```

### 2.2 Vectorization Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                VECTORIZATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🔄 Process Flow:                                                │
│                                                                 │
│   1. Training Data:                                            │
│      • BoW: fit_transform() → save vocabulary                  │
│      • TF-IDF: fit_transform() → save vocabulary               │
│      • Embeddings: encode() → save model                       │
│                                                                 │
│   2. Validation Data:                                          │
│      • BoW: transform() → use saved vocabulary                 │
│      • TF-IDF: transform() → use saved vocabulary              │
│      • Embeddings: encode() → use saved model                  │
│                                                                 │
│   3. Test Data:                                                │
│      • Same as validation data                                 │
│                                                                 │
│ 📊 Output Shapes:                                               │
│   • BoW: (n_samples, 5000) - Sparse matrix                    │
│   • TF-IDF: (n_samples, 5000) - Sparse matrix                 │
│   • Embeddings: (n_samples, 384) - Dense matrix               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 **PHASE 3: Model Training & Evaluation**

### 3.1 Model Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🎯 Model Types:                                                 │
│                                                                 │
│   1. K-Means Clustering (Unsupervised)                         │
│      • Algorithm: K-means clustering                           │
│      • Use Case: Baseline clustering performance               │
│      • Metrics: Silhouette score, inertia                      │
│                                                                 │
│   2. K-Nearest Neighbors (Supervised)                          │
│      • Algorithm: Instance-based learning                      │
│      • Hyperparameters: n_neighbors, weights                   │
│      • Use Case: Simple classification baseline                 │
│                                                                 │
│   3. Decision Tree (Supervised)                                │
│      • Algorithm: Tree-based classification                    │
│      • Hyperparameters: max_depth, min_samples_split           │
│      • Use Case: Interpretable classification                  │
│                                                                 │
│   4. Naive Bayes (Supervised)                                  │
│      • Algorithm: Probabilistic classifier                     │
│      • Hyperparameters: alpha (smoothing)                      │
│      • Use Case: Fast classification baseline                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Training Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🔄 Training Process:                                            │
│                                                                 │
│   For each model + vectorization combination:                   │
│                                                                 │
│   1. Model Initialization:                                     │
│      • Create model instance                                   │
│      • Set hyperparameters                                     │
│      • Configure random state                                  │
│                                                                 │
│   2. Training Phase:                                           │
│      • Train on training data                                  │
│      • Monitor training metrics                                │
│      • Save trained model                                      │
│                                                                 │
│   3. Validation Phase:                                         │
│      • Evaluate on validation data                             │
│      • Check for overfitting                                   │
│      • Adjust hyperparameters if needed                        │
│                                                                 │
│   4. Testing Phase:                                            │
│      • Final evaluation on test data                           │
│      • Generate performance metrics                             │
│      • Create confusion matrices                               │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**
```python
# models/new_model_trainer.py
def train_validate_test_model(self, model_name, X_train, y_train, 
                             X_val, y_val, X_test, y_test):
    # 1. Model initialization
    model = self.model_factory.create_model(model_name)
    
    # 2. Training
    model.fit(X_train, y_train)
    
    # 3. Validation
    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    # 4. Testing
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    return test_predictions, val_predictions, val_accuracy, test_accuracy
```

### 3.3 Cross-Validation & Overfitting Detection
```
┌─────────────────────────────────────────────────────────────────┐
│              CROSS-VALIDATION & OVERFITTING                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🔄 Cross-Validation Strategy:                                   │
│   • Folds: 5 (configurable)                                    │
│   • Stratified: Đảm bảo cân bằng labels                       │
│   • Metrics: Accuracy, Precision, Recall, F1                  │
│                                                                 │
│ ⚠️  Overfitting Detection:                                      │
│   • Compare: Training vs Validation performance                │
│   • Threshold: >5% difference = potential overfitting         │
│   • Actions: Regularization, early stopping, more data         │
│                                                                 │
│ 📊 Validation Metrics:                                          │
│   • Training Accuracy: Model performance on training data      │
│   • Validation Accuracy: Model performance on validation data  │
│   • Gap Analysis: Training - Validation difference             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📈 **PHASE 4: Performance Evaluation & Selection**

### 4.1 Comprehensive Evaluation Matrix
```
┌─────────────────────────────────────────────────────────────────┐
│                EVALUATION MATRIX                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📊 15 Model-Embedding Combinations:                            │
│                                                                 │
│   ┌─────────────┬─────────┬─────────┬─────────┬─────────────┐   │
│   │ Model       │ BoW     │ TF-IDF  │ Embed   │ Best        │   │
│   ├─────────────┼─────────┼─────────┼─────────┼─────────────┤   │
│   │ K-Means     │ Acc 1   │ Acc 2   │ Acc 3   │ Max(1,2,3) │   │
│   ├─────────────┼─────────┼─────────┼─────────┼─────────────┤   │
│   │ KNN         │ Acc 4   │ Acc 5   │ Acc 6   │ Max(4,5,6) │   │
│   ├─────────────┼─────────┼─────────┼─────────┼─────────────┤   │
│   │ Decision    │ Acc 7   │ Acc 8   │ Acc 9   │ Max(7,8,9) │   │
│   │ Tree        │         │         │         │             │   │
│   ├─────────────┼─────────┼─────────┼─────────┼─────────────┤   │
│   │ Naive Bayes │ Acc 10  │ Acc 11  │ Acc 12  │ Max(10,11,12)│   │
│   └─────────────┴─────────┴─────────┴─────────┴─────────────┘   │
│                                                                 │
│ 🎯 Selection Criteria:                                          │
│   • Primary: Highest accuracy                                  │
│   • Secondary: Training time                                   │
│   • Tertiary: Memory usage                                     │
│   • Quaternary: Interpretability                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Performance Metrics & Analysis
```
┌─────────────────────────────────────────────────────────────────┐
│                PERFORMANCE METRICS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📊 Primary Metrics:                                             │
│   • Accuracy: Overall correct predictions / total predictions  │
│   • Precision: True positives / (True + False positives)      │
│   • Recall: True positives / (True + False negatives)         │
│   • F1-Score: Harmonic mean of precision and recall           │
│                                                                 │
│ 📈 Secondary Metrics:                                           │
│   • Training Time: Time to train model                         │
│   • Prediction Time: Time to make predictions                  │
│   • Memory Usage: RAM consumption                              │
│   • Scalability: Performance with larger datasets              │
│                                                                 │
│ 🔍 Analysis Tools:                                              │
│   • Confusion Matrices: Per-class performance                  │
│   • ROC Curves: Classification threshold analysis              │
│   • Learning Curves: Training progress monitoring              │
│   • Feature Importance: Model interpretability                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Best Combination Selection
```
┌─────────────────────────────────────────────────────────────────┐
│                BEST COMBINATION SELECTION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🎯 Selection Algorithm:                                         │
│                                                                 │
│   1. Performance Ranking:                                       │
│      • Sort all combinations by accuracy (descending)          │
│      • Apply minimum threshold (e.g., >70% accuracy)           │
│      • Filter out overfitting models                           │
│                                                                 │
│   2. Efficiency Analysis:                                       │
│      • Compare training times                                  │
│      • Compare prediction times                                │
│      • Compare memory usage                                    │
│                                                                 │
│   3. Robustness Check:                                          │
│      • Cross-validation stability                              │
│      • Standard deviation of performance                       │
│      • Outlier detection                                       │
│                                                                 │
│   4. Final Selection:                                           │
│      • Top performer within efficiency constraints              │
│      • Backup options for different use cases                  │
│      • Documentation of selection rationale                     │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**
```python
# comprehensive_evaluation.py
def find_best_combinations(self):
    """Find the best model-embedding combinations"""
    
    # 1. Performance ranking
    performance_ranking = self._rank_by_performance()
    
    # 2. Efficiency analysis
    efficiency_scores = self._calculate_efficiency_scores()
    
    # 3. Robustness check
    robustness_scores = self._calculate_robustness_scores()
    
    # 4. Final selection
    best_combinations = self._select_best_combinations(
        performance_ranking, efficiency_scores, robustness_scores
    )
    
    return best_combinations

def _rank_by_performance(self):
    """Rank combinations by accuracy"""
    rankings = {}
    for combo, results in self.evaluation_results.items():
        rankings[combo] = {
            'accuracy': results['test_accuracy'],
            'cv_mean': results['cv_results']['mean_accuracy'],
            'cv_std': results['cv_results']['std_accuracy']
        }
    
    # Sort by accuracy (descending)
    return dict(sorted(rankings.items(), 
                      key=lambda x: x[1]['accuracy'], reverse=True))
```

---

## 🔄 **PHASE 5: Results & Visualization**

### 5.1 Output Generation
```
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT GENERATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📊 Performance Reports:                                         │
│   • CSV Reports: Detailed metrics for all combinations         │
│   • Summary Tables: Top performers and statistics              │
│   • Comparison Charts: Model vs Model analysis                 │
│                                                                 │
│ 📈 Visualizations:                                              │
│   • Confusion Matrices: Per-model performance                  │
│   • Accuracy Comparison: Bar charts and heatmaps               │
│   • Learning Curves: Training progress visualization            │
│   • ROC Curves: Classification threshold analysis              │
│                                                                 │
│ 💾 Export Options:                                              │
│   • PDF Reports: Publication-ready figures                     │
│   • Model Files: Trained models for deployment                 │
│   • Configuration Files: Settings for reproducibility          │
│   • Log Files: Complete execution logs                         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Results Storage & Retrieval
```
┌─────────────────────────────────────────────────────────────────┐
│                RESULTS STORAGE & RETRIEVAL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 📁 File Structure:                                              │
│   pdf/Figures/                                                 │
│   ├── confusion_matrices/                                      │
│   │   ├── kmeans_bow_confusion_matrix.pdf                     │
│   │   ├── kmeans_tfidf_confusion_matrix.pdf                   │
│   │   ├── knn_bow_confusion_matrix.pdf                        │
│   │   └── ...                                                  │
│   ├── model_comparison.pdf                                     │
│   └── performance_summary.pdf                                  │
│                                                                 │
│ 💾 Data Persistence:                                            │
│   • Session State: Streamlit app state management              │
│   • File Cache: Local storage for large datasets               │
│   • Model Registry: Trained model storage                      │
│   • Configuration Cache: User preference storage               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 **PHASE 6: Optimization & Deployment**

### 6.1 Performance Optimization
```
┌─────────────────────────────────────────────────────────────────┐
│                PERFORMANCE OPTIMIZATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ⚡ Speed Optimization:                                           │
│   • Parallel Processing: Multi-core training                   │
│   • Batch Processing: Chunked data processing                  │
│   • Caching: Intermediate results storage                      │
│   • Lazy Loading: Load data only when needed                   │
│                                                                 │
│ 💾 Memory Optimization:                                         │
│   • Sparse Matrices: For high-dimensional data                 │
│   • Data Streaming: Process data in chunks                     │
│   • Garbage Collection: Automatic memory cleanup               │
│   • Memory Mapping: Large file handling                        │
│                                                                 │
│ 🔧 Model Optimization:                                           │
│   • Hyperparameter Tuning: Grid/random search                  │
│   • Feature Selection: Dimensionality reduction                │
│   • Ensemble Methods: Combine multiple models                  │
│   • Transfer Learning: Pre-trained model adaptation            │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Deployment & Monitoring
```
┌─────────────────────────────────────────────────────────────────┐
│                DEPLOYMENT & MONITORING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 🚀 Production Deployment:                                       │
│   • Model Serialization: Save trained models                   │
│   • API Endpoints: RESTful service endpoints                   │
│   • Load Balancing: Handle multiple requests                   │
│   • Scaling: Auto-scaling based on demand                      │
│                                                                 │
│ 📊 Performance Monitoring:                                      │
│   • Real-time Metrics: Accuracy, latency, throughput          │
│   • Alerting: Performance degradation detection                │
│   • Logging: Comprehensive execution logs                      │
│   • Analytics: Usage pattern analysis                          │
│                                                                 │
│ 🔄 Continuous Improvement:                                      │
│   • A/B Testing: Compare model versions                        │
│   • Feedback Loop: User feedback integration                   │
│   • Model Updates: Incremental model improvements              │
│   • Performance Tracking: Long-term trend analysis             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 **Pipeline Summary & Key Insights**

### Pipeline Flow Diagram
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dataset   │───▶│  Preprocess │───▶│ Vectorize   │───▶│   Train     │
│  Selection  │    │   & Clean   │    │   Text      │    │   Models    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  Deploy &   │◀───│  Select     │◀───│  Evaluate   │◀────────┘
│  Monitor    │    │   Best      │    │ Performance │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Success Factors
1. **Data Quality**: Clean, balanced, representative dataset
2. **Vectorization Choice**: Semantic vs statistical approaches
3. **Model Selection**: Algorithm diversity and hyperparameter tuning
4. **Validation Strategy**: Robust cross-validation and overfitting detection
5. **Performance Metrics**: Comprehensive evaluation beyond just accuracy
6. **Scalability**: Efficient processing for large datasets
7. **Reproducibility**: Consistent results across runs

### Performance Benchmarks
- **Processing Speed**: 100K samples in ~5-10 minutes
- **Memory Usage**: Optimized for 8GB+ RAM systems
- **Accuracy Range**: 60-85% depending on complexity
- **Scalability**: Linear scaling with dataset size
- **Reliability**: 99%+ successful execution rate

---

## 🔮 **Future Enhancements**

### Phase 2 Planned Features
1. **Advanced Models**: BERT, GPT, Transformer architectures
2. **AutoML Integration**: Automated hyperparameter optimization
3. **Real-time Processing**: Streaming data processing capabilities
4. **Cloud Deployment**: AWS, Azure, GCP integration
5. **Advanced Analytics**: Deep learning interpretability tools
6. **Multi-language Support**: International dataset handling

---

*Pipeline này được thiết kế để xử lý hiệu quả các dataset lớn với độ chính xác cao, đồng thời cung cấp khả năng mở rộng và tùy chỉnh cho các use case khác nhau.*
