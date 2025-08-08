# 🧠 SƠ ĐỒ TƯ DUY - SPAM EMAIL CLASSIFIER

## 📋 TỔNG QUAN DỰ ÁN

```
🎯 MỤC TIÊU: Phân loại email spam/ham sử dụng AI
🛠️ CÔNG NGHỆ: KNN + TF-IDF + Gmail API
📊 DỮ LIỆU: Email dataset + Real-time Gmail + User Corrections
🎨 GIAO DIỆN: Command Line + Streamlit Web
```

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

### 1. **ENTRY POINT** (`main.py`)
```
📥 INPUT: Command line arguments
├── --regenerate: Tái tạo embedding
├── --run-email-classifier: Chạy Gmail API
├── --merge-emails: Gộp email local
├── --evaluate: Đánh giá mô hình
└── --k-values: Tùy chỉnh k cho KNN

🔄 PROCESSING:
├── Khởi tạo config
├── Tạo pipeline
├── Cache priority logic (corrections > original)
├── Huấn luyện mô hình
├── Chạy chức năng được chọn
└── Logging và error handling

📤 OUTPUT: Kết quả phân loại + Logs
```

### 2. **CONFIGURATION** (`config.py`)
```
⚙️ CẤU HÌNH HỆ THỐNG:
├── Model Settings
│   ├── model_name: 'intfloat/multilingual-e5-base'
│   ├── max_length: 512
│   └── batch_size: 32
├── Performance Settings
│   ├── use_gpu: True
│   ├── num_workers: 4
│   └── pin_memory: True
├── Training Settings
│   ├── test_size: 0.1
│   └── random_state: 42
├── KNN Settings
│   ├── default_k: 3
│   └── k_values: [1, 3, 5]
└── Path Settings
    ├── dataset_path: './dataset/2cls_spam_text_cls.csv'
    ├── output_dir: './cache/output'
    └── credentials_path: './cache/input/credentials.json'
```

---

## 🔄 PIPELINE CHÍNH (`spam_classifier.py`)

### **SpamClassifierPipeline Class**
```
🎯 CHỨC NĂNG: Orchestrate toàn bộ quá trình

📦 COMPONENTS:
├── DataLoader: Tải và xử lý dữ liệu
├── EmbeddingGenerator: Tạo embedding
├── KNNClassifier: Phân loại KNN
└── TFIDFClassifier: Baseline TF-IDF

🔄 WORKFLOW:
1. Load Data → 2. Generate Embeddings → 3. Train Classifier → 4. Predict

🆕 CORRECTIONS HANDLING:
├── train_with_corrections(): Merge CSV + JSON corrections
├── _save_corrections_dataset(): Cache merged dataset
├── _load_corrections_dataset(): Load cached dataset
└── Cache priority: corrections > original
```

---

## 📊 DATA PROCESSING

### **DataLoader** (`data_loader.py`)
```
📥 INPUT SOURCES:
├── CSV Dataset: 2cls_spam_text_cls.csv
├── Local Folders: inbox/ (ham) + spam/ (spam)
├── Gmail API: Real-time emails
└── User Corrections: corrections.json

🛠️ PREPROCESSING:
├── Text Cleaning
│   ├── Remove URLs, emails, numbers
│   ├── Lowercase conversion
│   └── Remove punctuation
├── NLP Processing
│   ├── Remove stopwords
│   ├── Lemmatization
│   └── Tokenization
└── Label Encoding
    ├── ham → 0
    └── spam → 1

📤 OUTPUT:
├── Cleaned messages
├── Encoded labels
└── Metadata for evaluation
```

---

## 🤖 AI MODELS

### **1. Embedding Generator** (`embedding_generator.py`)
```
🧠 MODEL: intfloat/multilingual-e5-base
├── Multilingual transformer
├── 768-dimensional embeddings
└── Average pooling strategy

⚡ OPTIMIZATION:
├── GPU acceleration
├── Batch processing
├── Caching system (with suffixes)
└── Memory optimization

🔄 PROCESS:
1. Tokenize text
2. Generate embeddings
3. Average pooling
4. Normalize vectors
5. Cache results

🆕 CACHE SUFFIXES:
├── _original: Base dataset embeddings
├── _with_corrections: Merged dataset embeddings
└── Priority system for Gmail classification
```

### **2. KNN Classifier** (`knn_classifier.py`)
```
🎯 ALGORITHM: K-Nearest Neighbors
├── Distance metric: Inner Product (FAISS)
├── K values: [1, 3, 5] (configurable)
└── Voting: Majority vote

⚡ FAISS OPTIMIZATION:
├── IndexFlatIP: Fast similarity search
├── GPU acceleration (optional)
├── Efficient memory usage
└── Cache management (with suffixes)

🔄 PREDICTION:
1. Generate query embedding
2. Search k nearest neighbors
3. Collect neighbor labels
4. Majority voting
5. Return prediction + confidence

🆕 CACHE MANAGEMENT:
├── save_index(cache_suffix): Save FAISS index
├── load_index(cache_suffix): Load FAISS index
└── Priority: corrections > original
```

### **3. TF-IDF Classifier** (`tfidf_classifier.py`)
```
🎯 BASELINE: TF-IDF + Naive Bayes
├── Feature extraction: TF-IDF
├── Classifier: MultinomialNB
└── Purpose: Performance comparison

📊 FEATURES:
├── Term frequency
├── Inverse document frequency
└── N-gram features (1-2 grams)
```

---

## 📧 EMAIL INTEGRATION

### **Gmail Handler** (`email_handler.py`)
```
🔗 GMAIL API INTEGRATION:
├── OAuth 2.0 Authentication
├── Real-time email monitoring
└── Automatic labeling

🔄 EMAIL PROCESSING:
1. Fetch unread emails
2. Extract email content
3. Classify using trained model
4. Apply Gmail labels
5. Save locally (.txt files)
6. Mark as read

🏷️ LABELING STRATEGY:
├── Inbox_Custom: Ham emails
└── Spam_Custom: Spam emails

🆕 CACHE PRIORITY:
├── Use corrections model if available
├── Fallback to original model
└── Terminal logging for verification
```

---

## 📈 EVALUATION SYSTEM

### **Model Evaluator** (`evaluator.py`)
```
📊 METRICS:
├── Accuracy
├── Precision
├── Recall
├── F1-Score
└── Confusion Matrix

📈 VISUALIZATIONS:
├── Performance comparison (KNN vs TF-IDF)
├── K-value analysis
├── Error analysis
└── Confusion matrix heatmap

🔄 EVALUATION PROCESS:
1. Load test data
2. Generate embeddings (with _original suffix)
3. Run predictions
4. Calculate metrics
5. Create visualizations
6. Save results
```

---

## 💾 CACHING SYSTEM

### **Cache Structure**
```
📁 CACHE DIRECTORY:
├── cache/
│   ├── input/
│   │   ├── credentials.json
│   │   └── token.json
│   ├── output/
│   │   ├── evaluation_summary.png
│   │   └── error_analysis.json
│   ├── embeddings/
│   │   ├── embeddings_intfloat_multilingual-e5-base_original.npy
│   │   └── embeddings_intfloat_multilingual-e5-base_with_corrections.npy
│   ├── datasets/
│   │   └── with_corrections_dataset_intfloat_multilingual-e5-base.pkl
│   ├── faiss_index/
│   │   ├── faiss_index_intfloat_multilingual-e5-base_original.faiss
│   │   ├── faiss_index_intfloat_multilingual-e5-base_original.pkl
│   │   ├── faiss_index_intfloat_multilingual-e5-base_with_corrections.faiss
│   │   └── faiss_index_intfloat_multilingual-e5-base_with_corrections.pkl
│   └── models/
│       ├── model_intfloat_multilingual-e5-base.joblib
│       └── tokenizer_intfloat_multilingual-e5-base.joblib
```

### **Cache Priority System**
```
🔄 CACHE PRIORITY LOGIC:
├── Gmail Classification:
│   ├── Check _with_corrections cache first
│   ├── Fallback to _original cache
│   └── Terminal logging for verification
├── Training:
│   ├── train_with_corrections(): Use merged dataset
│   ├── train(): Use original dataset
│   └── Save separate caches for each
└── FAISS Index:
    ├── Load corresponding index for each cache
    ├── Save index with appropriate suffix
    └── Maintain consistency between embeddings and index
```

---

## 🚀 USAGE SCENARIOS

### **1. Initial Setup**
```bash
python main.py
# → Train model + Generate embeddings
```

### **2. Model Evaluation**
```bash
python main.py --evaluate --k-values "1,3,7"
# → Evaluate performance + Generate plots
```

### **3. Email Management**
```bash
python main.py --merge-emails --regenerate
# → Merge local emails + Update embeddings
```

### **4. Real-time Classification**
```bash
python main.py --run-email-classifier
# → Monitor Gmail + Auto-classify (with cache priority)
```

### **5. Retrain with Corrections**
```bash
# Via Streamlit interface
# → Merge CSV + corrections.json
# → Save merged dataset cache
# → Update embeddings and FAISS index
```

---

## 🔧 TECHNICAL STACK

### **Core Technologies**
```
🐍 Python 3.10+
├── numpy: Numerical computations
├── pandas: Data manipulation
├── scikit-learn: ML utilities
├── transformers: Hugging Face models
├── faiss-cpu: Similarity search
└── google-api-python-client: Gmail API
```

### **AI/ML Stack**
```
🤖 Machine Learning:
├── KNN (FAISS)
├── TF-IDF + Naive Bayes
└── Transformer embeddings

📊 Visualization:
├── matplotlib
├── seaborn
└── Streamlit (web interface)
```

---

## 📊 DATA FLOW

```
📥 INPUT DATA
    ↓
🛠️ PREPROCESSING (DataLoader)
    ↓
🤖 EMBEDDING GENERATION (EmbeddingGenerator)
    ↓
🎯 CLASSIFICATION (KNNClassifier/TFIDFClassifier)
    ↓
📤 OUTPUT RESULTS
    ↓
📧 GMAIL INTEGRATION (EmailHandler)
    ↓
📈 EVALUATION (ModelEvaluator)
```

---

## 🎯 KEY FEATURES

### **✅ Implemented**
- [x] Dual classifier system (KNN + TF-IDF)
- [x] Real-time Gmail integration
- [x] Local email management
- [x] Comprehensive evaluation
- [x] Advanced caching system
- [x] Multi-language support
- [x] Command-line interface
- [x] Error handling & logging
- [x] User corrections handling
- [x] Cache priority system
- [x] FAISS index management

### **🚀 Advanced Features**
- [x] GPU acceleration
- [x] Batch processing
- [x] Memory optimization
- [x] Configurable parameters
- [x] Performance metrics
- [x] Visualization tools
- [x] Terminal logging for cache verification
- [x] Stable merged dataset caching

---

## 🔍 DEBUGGING & MONITORING

### **Logging System**
```
📝 LOG FILES:
├── logs/spam_classifier.log
├── Console output
└── Error tracking

🔍 DEBUG INFO:
├── Model loading status
├── Embedding generation progress
├── Classification results
├── Performance metrics
├── Cache usage verification
└── FAISS index loading status

🆕 TERMINAL LOGGING:
├── Cache priority decisions
├── FAISS index loading
└── Gmail classification cache usage
```

---

## 📈 PERFORMANCE OPTIMIZATION

### **Speed Optimizations**
```
⚡ GPU ACCELERATION:
├── CUDA support for transformers
├── FAISS GPU index (optional)
└── Batch processing

💾 MEMORY OPTIMIZATION:
├── Embedding caching (with suffixes)
├── Model caching
├── FAISS index caching
├── Merged dataset caching
└── Efficient data structures
```

---

## 🎨 USER INTERFACES

### **1. Command Line Interface**
```bash
# Basic usage
python main.py

# Advanced options
python main.py --evaluate --k-values "1,3,7" --classifier knn
```

### **2. Streamlit Web Interface**
```bash
streamlit run app.py
# → Interactive web dashboard
# → Cache priority logic for email scanning
# → Terminal logging for verification
```

---

## 🔐 SECURITY & PRIVACY

### **Authentication**
```
🔑 GMAIL API:
├── OAuth 2.0 flow
├── Secure token storage
└── Automatic token refresh

📁 FILE SECURITY:
├── Credentials in cache/input/
├── Local email storage
└── Log file management
```

---

## 📚 LEARNING OUTCOMES

### **Technical Skills**
- [x] Transformer models (Hugging Face)
- [x] FAISS similarity search
- [x] Gmail API integration
- [x] Machine learning pipelines
- [x] Data preprocessing
- [x] Model evaluation
- [x] Cache management systems
- [x] User feedback integration

### **Software Engineering**
- [x] Modular architecture
- [x] Configuration management
- [x] Error handling
- [x] Logging systems
- [x] Advanced caching strategies
- [x] Performance optimization
- [x] Cache priority systems
- [x] Data consistency management

---

## 🎯 PROJECT HIGHLIGHTS

```
🏆 ACHIEVEMENTS:
├── Real-time email classification
├── Dual classifier comparison
├── Comprehensive evaluation system
├── Gmail API integration
├── Performance optimization
├── User-friendly interfaces
├── Advanced cache management
├── User corrections handling
└── FAISS index optimization

🚀 INNOVATION:
├── Multilingual support
├── GPU acceleration
├── Intelligent caching with priorities
├── Configurable parameters
├── Comprehensive documentation
├── Stable dataset caching
└── Terminal logging for verification
```

---

*📝 **Note**: This mind map provides a comprehensive overview of the Spam Email Classifier project architecture, components, and workflows. Use it as a reference for understanding and extending the project.*
