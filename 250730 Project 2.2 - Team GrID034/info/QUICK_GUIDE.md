# 🚀 HƯỚNG DẪN NHANH - SPAM EMAIL CLASSIFIER

## 🎯 DỰ ÁN LÀ GÌ?

**Spam Email Classifier** là một hệ thống AI phân loại email spam/ham sử dụng:
- 🤖 **KNN + FAISS**: Thuật toán chính
- 📊 **TF-IDF**: Baseline so sánh
- 📧 **Gmail API**: Tích hợp thời gian thực
- 🎨 **CLI + Web**: Giao diện đa dạng

---

## 🏗️ KIẾN TRÚC ĐƠN GIẢN

```
📥 INPUT → 🛠️ PROCESS → 🤖 CLASSIFY → 📤 OUTPUT
   │           │           │           │
   ▼           ▼           ▼           ▼
📧 Email   🧹 Clean   🎯 KNN      🏷️ Spam/Ham
📁 Files   🔤 NLP     📊 TF-IDF   📈 Results
📊 Dataset 🤖 Embed    🗳️ Vote     📧 Labels
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
# → Tự động phân loại email mới
```

### **4. Gộp email local**
```bash
python main.py --merge-emails --regenerate
# → Thêm email từ thư mục inbox/spam
```

---

## 🔄 LUỒNG XỬ LÝ

### **1. Data Loading**
```
📁 Dataset CSV → 🧹 Text Cleaning → 🔤 NLP Processing → 🏷️ Label Encoding
```

### **2. Model Training**
```
📝 Text → 🤖 Transformer → 📊 Embeddings → 🎯 KNN Index → 💾 Cache
```

### **3. Prediction**
```
📧 New Email → 🤖 Embedding → 🔍 FAISS Search → 🗳️ Majority Vote → 🏷️ Result
```

### **4. Gmail Integration**
```
📧 Fetch Email → 🤖 Classify → 🏷️ Apply Label → 📁 Save Local → ✅ Mark Read
```

---

## 🎯 CÁC THUẬT TOÁN

### **KNN Classifier**
- 🎯 **FAISS**: Tìm kiếm nhanh
- 🔍 **K-Nearest**: K neighbors gần nhất
- 🗳️ **Majority Vote**: Bỏ phiếu đa số
- ⚡ **GPU Support**: Tăng tốc độ

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

### **Preprocessing**
- 🧹 **Text Cleaning**: Remove URLs, emails, numbers
- 🔤 **NLP**: Stopwords, lemmatization, tokenization
- 🏷️ **Encoding**: ham → 0, spam → 1

---

## 💾 CACHE SYSTEM

```
cache/
├── input/          # credentials.json, token.json
├── output/         # plots, evaluation results
├── embeddings/     # cached embeddings
└── models/         # cached models & tokenizers
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
- 💾 **Caching**: Embeddings, models, results
- 🔢 **Batch Processing**: Efficient computation
- 💾 **Memory Optimization**: Efficient data structures

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
- [x] Caching system
- [x] Multi-language support

### **🚀 Advanced Features**
- [x] GPU acceleration
- [x] Batch processing
- [x] Memory optimization
- [x] Configurable parameters
- [x] Performance metrics
- [x] Visualization tools

---

## 🔍 DEBUGGING

### **Log Files**
```
logs/spam_classifier.log
# → Detailed process logs
```

### **Common Issues**
- 🔑 **Authentication**: Check credentials.json
- 💾 **Cache**: Use --regenerate for cache issues
- 📊 **Dataset**: Ensure dataset exists
- ⚡ **GPU**: Check CUDA availability

---

## 📚 LEARNING OUTCOMES

### **Technical Skills**
- 🤖 Transformer models (Hugging Face)
- 🔍 FAISS similarity search
- 📧 Gmail API integration
- 🔄 Machine learning pipelines
- 🛠️ Data preprocessing
- 📊 Model evaluation

### **Software Engineering**
- 🏗️ Modular architecture
- ⚙️ Configuration management
- 🛡️ Error handling
- 📝 Logging systems
- 💾 Caching strategies
- ⚡ Performance optimization

---

## 🎯 PROJECT HIGHLIGHTS

```
🏆 ACHIEVEMENTS:
├── Real-time email classification
├── Dual classifier comparison
├── Comprehensive evaluation system
├── Gmail API integration
├── Performance optimization
└── User-friendly interfaces

🚀 INNOVATION:
├── Multilingual support
├── GPU acceleration
├── Intelligent caching
├── Configurable parameters
└── Comprehensive documentation
```

---

*📝 **Quick Guide**: This guide provides a fast overview of the Spam Email Classifier project. For detailed information, see the full README.md and MIND_MAP.md files.*
