# 🚀 HƯỚNG DẪN NHANH - SPAM EMAIL CLASSIFIER

## 🎯 DỰ ÁN LÀ GÌ?

**Spam Email Classifier** là một hệ thống AI phân loại email spam/ham sử dụng:
- 🤖 **KNN + FAISS**: Thuật toán chính
- 📊 **TF-IDF**: Baseline so sánh
- 📧 **Gmail API**: Tích hợp thời gian thực
- 🎨 **CLI + Web**: Giao diện đa dạng
- 🔄 **User Corrections**: Học từ phản hồi người dùng

---

## 🏗️ KIẾN TRÚC ĐƠN GIẢN

```
📥 INPUT → 🛠️ PROCESS → 🤖 CLASSIFY → 📤 OUTPUT
   │           │           │           │
   ▼           ▼           ▼           ▼
📧 Email   🧹 Clean   🎯 KNN      🏷️ Spam/Ham
📁 Files   🔤 NLP     📊 TF-IDF   📈 Results
📊 Dataset 🤖 Embed    🗳️ Vote     📧 Labels
🔄 Corrections 🎯 Cache Priority 🏷️ Corrections
```

---

## 📁 CÁC FILE CHÍNH

| File | Chức năng |
|------|-----------|
| `main.py` | 🚪 Entry point, xử lý arguments |
| `config.py` | ⚙️ Cấu hình hệ thống |
| `spam_classifier.py` | 🔄 Pipeline chính |
| `data_loader.py` | 📊 Tải và xử lý dữ liệu |
| `embedding_generator.py` | 🤖 Tạo embedding |
| `knn_classifier.py` | 🎯 Thuật toán KNN |
| `email_handler.py` | 📧 Tích hợp Gmail API |
| `evaluator.py` | 📈 Đánh giá mô hình |
| `app.py` | 🌐 Streamlit web interface |

---

## 🚀 CÁCH SỬ DỤNG

### **1. Khởi tạo lần đầu**
```bash
python main.py
# → Train model + Generate embeddings
```

### **2. Đánh giá mô hình**
```bash
python main.py --evaluate
# → Tạo biểu đồ performance
```

### **3. Chạy phân loại Gmail**
```bash
python main.py --run-email-classifier
# → Tự động phân loại email mới (với cache priority)
```

### **4. Gộp email local**
```bash
python main.py --merge-emails --regenerate
# → Thêm email từ thư mục inbox/spam
```

### **5. Retrain với Corrections**
```bash
# Via Streamlit interface
# → Merge CSV + corrections.json
# → Save merged dataset cache
# → Update embeddings và FAISS index
```

---

## 🔄 LUỒNG XỬ LÝ

### **1. Data Loading**
```
📁 Dataset CSV → 🧹 Text Cleaning → 🔤 NLP Processing → 🏷️ Label Encoding
🔄 User Corrections → 📊 Merge Data → 💾 Cache Dataset
```

### **2. Model Training**
```
📝 Text → 🤖 Transformer → 📊 Embeddings → 🎯 KNN Index → 💾 Cache
🔄 Cache Priority → 📁 Separate Caches → 🔍 FAISS Index
```

### **3. Prediction**
```
📧 New Email → 🤖 Embedding → 🔍 FAISS Search → 🗳️ Majority Vote → 🏷️ Result
🔄 Cache Verification → 📊 Terminal Logging → 🎯 Cache Priority
```

### **4. Gmail Integration**
```
📧 Fetch Email → 🤖 Classify → 🏷️ Apply Label → 📁 Save Local → ✅ Mark Read
🔄 Cache Priority Logic → 📊 Terminal Logging → 🎯 Corrections First
```

---

## 🎯 CÁC THUẬT TOÁN

### **KNN Classifier**
- 🎯 **FAISS**: Tìm kiếm nhanh
- 🔍 **K-Nearest**: K neighbors gần nhất
- 🗳️ **Majority Vote**: Bỏ phiếu đa số
- ⚡ **GPU Support**: Tăng tốc độ
- 💾 **Cache Management**: Separate caches cho original/corrections

### **TF-IDF Classifier**
- 📊 **TF-IDF**: Feature extraction
- 🧠 **Naive Bayes**: Baseline classifier
- 📈 **N-gram**: 1-2 gram features
- 🏷️ **Comparison**: So sánh performance

---

## 📊 DỮ LIỆU

### **Input Sources**
- 📁 **CSV Dataset**: `2cls_spam_text_cls.csv`
- 📁 **Local Folders**: `inbox/` (ham) + `spam/` (spam)
- 📧 **Gmail API**: Real-time emails
- 🔄 **User Corrections**: `corrections.json`

### **Preprocessing**
- 🧹 **Text Cleaning**: Remove URLs, emails, numbers
- 🔤 **NLP**: Stopwords, lemmatization, tokenization
- 🏷️ **Encoding**: ham → 0, spam → 1
- 🔄 **Corrections Merge**: CSV + JSON corrections

---

## 💾 CACHE SYSTEM

```
cache/
├── input/          # credentials.json, token.json
├── output/         # plots, evaluation results
├── embeddings/     # cached embeddings (with suffixes)
│   ├── embeddings_intfloat_multilingual-e5-base_original.npy
│   └── embeddings_intfloat_multilingual-e5-base_with_corrections.npy
├── datasets/       # merged corrections dataset
│   └── with_corrections_dataset_intfloat_multilingual-e5-base.pkl
├── faiss_index/    # FAISS indices (with suffixes)
│   ├── faiss_index_intfloat_multilingual-e5-base_original.faiss
│   ├── faiss_index_intfloat_multilingual-e5-base_original.pkl
│   ├── faiss_index_intfloat_multilingual-e5-base_with_corrections.faiss
│   └── faiss_index_intfloat_multilingual-e5-base_with_corrections.pkl
└── models/         # cached models & tokenizers
```

### **Cache Priority System**
```
🔄 PRIORITY LOGIC:
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

## 🎨 GIAO DIỆN

### **Command Line**
```bash
# Basic
python main.py

# Advanced
python main.py --evaluate --k-values "1,3,7" --classifier knn
```

### **Web Interface**
```bash
streamlit run app.py
# → Interactive dashboard
# → Cache priority logic for email scanning
# → Terminal logging for verification
```

---

## 🔧 TECHNICAL STACK

### **Core Libraries**
- 🐍 **Python 3.10+**
- 📦 **numpy, pandas**: Data processing
- 🤖 **scikit-learn**: ML utilities
- 🤖 **transformers**: Hugging Face models
- 🤖 **faiss-cpu**: Similarity search
- 📧 **google-api-python-client**: Gmail API

### **AI/ML Stack**
- 🤖 **KNN (FAISS)**: Main classifier
- 🤖 **TF-IDF + Naive Bayes**: Baseline
- 🤖 **Transformer embeddings**: Text representation
- 📊 **matplotlib, seaborn**: Visualization

---

## 📈 PERFORMANCE

### **Optimizations**
- ⚡ **GPU Acceleration**: CUDA support
- 💾 **Advanced Caching**: Embeddings, models, FAISS indices
- 🔢 **Batch Processing**: Efficient computation
- 💾 **Memory Optimization**: Efficient data structures
- 🔄 **Cache Priority**: Corrections > Original
- 📊 **Terminal Logging**: Cache verification

### **Metrics**
- 📈 **Accuracy**: Overall performance
- 📈 **Precision**: Spam detection accuracy
- 📈 **Recall**: Spam coverage
- 📈 **F1-Score**: Balanced metric

---

## 🔐 SECURITY

### **Gmail API**
- 🔑 **OAuth 2.0**: Secure authentication
- 🔐 **Token Storage**: Local token management
- 🔄 **Auto Refresh**: Automatic token renewal

### **Data Privacy**
- 📁 **Local Storage**: Email files stored locally
- 🔒 **Secure Credentials**: Protected credential files
- 📝 **Logging**: Controlled log output

---

## 🎯 KEY FEATURES

### **✅ Core Features**
- [x] Dual classifier system
- [x] Real-time Gmail integration
- [x] Local email management
- [x] Comprehensive evaluation
- [x] Advanced caching system
- [x] Multi-language support
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

## 🔍 DEBUGGING

### **Log Files**
```
logs/spam_classifier.log
# → Detailed process logs
```

### **Terminal Logging**
```
# Cache priority decisions
EMAIL SCAN: Using cache _with_corrections for Gmail classification
FAISS INDEX: Loading from cache _with_corrections

# FAISS index loading
FAISS LOAD: _with_corrections index with 11327 vectors
```

### **Common Issues**
- 🔑 **Authentication**: Check credentials.json
- 💾 **Cache**: Use --regenerate for cache issues
- 📊 **Dataset**: Ensure dataset exists
- ⚡ **GPU**: Check CUDA availability
- 🔄 **Corrections**: Check corrections.json format
- 📊 **Cache Priority**: Verify terminal logging

---

## 📚 LEARNING OUTCOMES

### **Technical Skills**
- 🤖 Transformer models (Hugging Face)
- 🔍 FAISS similarity search
- 📧 Gmail API integration
- 🔄 Machine learning pipelines
- 🛠️ Data preprocessing
- 📊 Model evaluation
- 💾 Cache management systems
- 🔄 User feedback integration

### **Software Engineering**
- 🏗️ Modular architecture
- ⚙️ Configuration management
- 🛡️ Error handling
- 📝 Logging systems
- 💾 Advanced caching strategies
- ⚡ Performance optimization
- 🔄 Cache priority systems
- 📊 Data consistency management

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

*📝 **Quick Guide**: This guide provides a fast overview of the Spam Email Classifier project. For detailed information, see the full README.md and MIND_MAP.md files.*
