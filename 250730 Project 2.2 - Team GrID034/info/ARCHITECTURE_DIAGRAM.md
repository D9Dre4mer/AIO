# 🏗️ KIẾN TRÚC HỆ THỐNG - SPAM EMAIL CLASSIFIER

## 📊 SƠ ĐỒ TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SPAM EMAIL CLASSIFIER                            │
│                              (Main System)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINT                                  │
│                              (main.py)                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   --regenerate  │  │ --run-email-    │  │  --evaluate     │           │
│  │                 │  │  classifier     │  │                 │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CONFIGURATION                                  │
│                              (config.py)                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │  Model Settings │  │ Performance     │  │  Path Settings  │           │
│  │  - model_name   │  │  - use_gpu     │  │  - dataset_path │           │
│  │  - max_length   │  │  - num_workers │  │  - output_dir   │           │
│  │  - batch_size   │  │  - pin_memory  │  │  - credentials  │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN PIPELINE                                   │
│                        (spam_classifier.py)                               │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │   DataLoader    │  │EmbeddingGenerator│  │  KNNClassifier  │           │
│  │                 │  │                 │  │                 │           │
│  │  - Load data    │  │ - Generate      │  │ - FAISS index   │           │
│  │  - Preprocess   │  │   embeddings    │  │ - KNN search    │           │
│  │  - Split data   │  │ - Cache results │  │ - Majority vote │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │TFIDFClassifier  │  │  Model Cache    │  │  Error Handler  │           │
│  │                 │  │                 │  │                 │           │
│  │ - TF-IDF        │  │ - Embeddings   │  │ - Logging       │           │
│  │ - Naive Bayes   │  │ - Models       │  │ - Exception     │           │
│  │ - Baseline      │  │ - Tokenizers   │  │   handling      │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                       │
│                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   INPUT     │───▶│ PREPROCESS  │───▶│ EMBEDDING   │───▶│ CLASSIFY    │ │
│  │             │    │             │    │             │    │             │ │
│  │ - CSV data  │    │ - Clean     │    │ - Transform │    │ - KNN       │ │
│  │ - Local     │    │ - Tokenize  │    │ - Cache     │    │ - TF-IDF    │ │
│  │ - Gmail     │    │ - Lemmatize │    │ - Normalize │    │ - Vote      │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   OUTPUT    │◀───│  EVALUATE   │◀───│  PREDICT    │◀───│  TRAIN      │ │
│  │             │    │             │    │             │    │             │ │
│  │ - Results   │    │ - Metrics   │    │ - Labels    │    │ - Models    │ │
│  │ - Logs      │    │ - Plots     │    │ - Confidence│    │ - Cache     │ │
│  │ - Files     │    │ - Analysis  │    │ - Neighbors │    │ - Index     │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 CHI TIẾT LUỒNG XỬ LÝ

### **1. DATA LOADING & PREPROCESSING**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PROCESSING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   RAW DATA      │  │   TEXT CLEANING │  │   NLP PROCESS   │  │   LABEL ENCODE  │
│                 │  │                 │  │                 │  │                 │
│ 📁 CSV Dataset  │  │ 🧹 Remove URLs  │  │ 🚫 Stopwords    │  │ 🏷️ ham → 0      │
│ 📁 Local Files  │  │ 🧹 Remove emails│  │ 🔤 Lemmatize    │  │ 🏷️ spam → 1     │
│ 📧 Gmail API    │  │ 🧹 Lowercase    │  │ ✂️ Tokenize     │  │ 📊 Metadata     │
│                 │  │ 🧹 Punctuation  │  │ 🔄 Normalize    │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLEANED DATA                                    │
│  - Preprocessed text messages                                             │
│  - Encoded labels (0/1)                                                  │
│  - Metadata for evaluation                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **2. EMBEDDING GENERATION**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING GENERATION                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   TEXT INPUT    │  │   TOKENIZATION  │  │  TRANSFORMER    │  │   POOLING       │
│                 │  │                 │  │                 │  │                 │
│ 📝 Clean text   │  │ 🔤 Tokenize     │  │ 🤖 Model:       │  │ 📊 Average      │
│ 📏 Max length   │  │ 📏 Pad/truncate │  │ multilingual-   │  │ 📏 Normalize    │
│ 🔢 Batch size   │  │ 🔢 Batch        │  │ e5-base         │  │ 💾 Cache        │
│                 │  │                 │  │ 🧠 768-dim      │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDINGS                                      │
│  - 768-dimensional vectors                                               │
│  - Normalized and cached                                                 │
│  - Ready for similarity search                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **3. CLASSIFICATION SYSTEM**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DUAL CLASSIFIER SYSTEM                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ┌─────────────────┐                    ┌─────────────────┐               │
│  │   KNN CLASSIFIER│                    │ TF-IDF CLASSIFIER│               │
│  │                 │                    │                 │               │
│  │ 🎯 FAISS Index  │                    │ 📊 TF-IDF       │               │
│  │ 🔍 K-Nearest    │                    │ 🧠 Naive Bayes  │               │
│  │ 🗳️ Majority Vote│                    │ 📈 N-gram       │               │
│  │ ⚡ GPU Support  │                    │ 🏷️ Baseline     │               │
│  └─────────────────┘                    └─────────────────┘               │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PREDICTION RESULTS                              │   │
│  │                                                                   │   │
│  │  🏷️ Label: spam/ham                                              │   │
│  │  📊 Confidence: 0.0-1.0                                          │   │
│  │  👥 Neighbors: Top-k similar                                     │   │
│  │  📈 Performance: Accuracy metrics                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **4. EMAIL INTEGRATION**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GMAIL API INTEGRATION                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   AUTHENTICATE  │  │   FETCH EMAILS  │  │   CLASSIFY      │  │   APPLY LABELS  │
│                 │  │                 │  │                 │  │                 │
│ 🔑 OAuth 2.0    │  │ 📧 Unread only  │  │ 🤖 Use trained  │  │ 🏷️ Inbox_Custom │
│ 🔐 Token store  │  │ ⏰ Real-time    │  │   model         │  │ 🏷️ Spam_Custom  │
│ 🔄 Auto refresh │  │ 📊 Max 10 emails│  │ 📊 Confidence   │  │ 📁 Save local   │
│                 │  │                 │  │ 🗳️ Prediction    │  │ ✅ Mark read     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMAIL MANAGEMENT                                │
│  - Automatic classification                                               │
│  - Gmail label application                                               │
│  - Local file storage                                                    │
│  - Real-time monitoring                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **5. EVALUATION SYSTEM**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   TEST DATA     │  │   PREDICTIONS   │  │   METRICS       │  │   VISUALIZATION │
│                 │  │                 │  │                 │  │                 │
│ 📊 Split data   │  │ 🤖 Run models   │  │ 📈 Accuracy     │  │ 📊 Performance  │
│ 🏷️ True labels  │  │ 📊 Get results  │  │ 📈 Precision    │  │ 📊 Confusion    │
│ 🔢 Test set     │  │ 🗳️ Predictions  │  │ 📈 Recall       │  │ 📊 K-analysis   │
│                 │  │                 │  │ 📈 F1-Score     │  │ 📊 Error analysis│
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION RESULTS                              │
│  - Performance metrics (JSON)                                             │
│  - Visualization plots (PNG)                                              │
│  - Error analysis                                                         │
│  - Model comparison                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 💾 CACHE ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CACHE STRUCTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  cache/                                                                   │
│  ├── input/                                                              │
│  │   ├── credentials.json    # Gmail API credentials                     │
│  │   └── token.json         # OAuth token                               │
│  │                                                                       │
│  ├── output/                                                             │
│  │   ├── evaluation_summary.png  # Performance plots                     │
│  │   └── error_analysis.json    # Detailed metrics                      │
│  │                                                                       │
│  ├── embeddings/                                                         │
│  │   └── embeddings_multilingual-e5-base.npy  # Cached embeddings       │
│  │                                                                       │
│  └── models/                                                             │
│      ├── model_multilingual-e5-base.joblib    # Cached model            │
│      └── tokenizer_multilingual-e5-base.joblib # Cached tokenizer       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 TECHNICAL STACK DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TECHNICAL STACK                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CORE PYTHON   │  │   ML/ML LIBRARIES│  │   API INTEGRATION│  │   VISUALIZATION │
│                 │  │                 │  │                 │  │                 │
│ 🐍 Python 3.10+ │  │ 🤖 scikit-learn │  │ 📧 Gmail API    │  │ 📊 matplotlib   │
│ 📦 numpy        │  │ 🤖 transformers │  │ 🔑 OAuth 2.0    │  │ 📊 seaborn      │
│ 📦 pandas       │  │ 🤖 faiss-cpu    │  │ 🔄 google-api   │  │ 🌐 streamlit    │
│ 📦 logging      │  │ 🤖 nltk         │  │ 📁 google-auth  │  │ 📈 plotly       │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT OPTIONS                              │
│  - Command Line Interface (CLI)                                           │
│  - Streamlit Web Interface                                               │
│  - Docker Container (future)                                             │
│  - Cloud Deployment (future)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 PERFORMANCE OPTIMIZATION

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PERFORMANCE FEATURES                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   GPU SUPPORT   │  │   CACHING       │  │   BATCH PROCESS │  │   MEMORY OPT    │
│                 │  │                 │  │                 │  │                 │
│ ⚡ CUDA support  │  │ 💾 Embeddings   │  │ 🔢 Batch size   │  │ 💾 Efficient    │
│ ⚡ FAISS GPU     │  │ 💾 Models       │  │ ⚡ Parallel     │  │   data structs  │
│ ⚡ Transformers  │  │ 💾 Tokenizers   │  │ ⚡ Workers      │  │ 💾 Memory pool  │
│                 │  │ 💾 Results       │  │ ⚡ Pinning      │  │ 💾 Garbage coll │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 🔄 DATA FLOW SUMMARY

```
INPUT DATA
    │
    ▼
┌─────────────┐
│  DataLoader │ ← CSV, Local files, Gmail API
└─────────────┘
    │
    ▼
┌─────────────┐
│Preprocessing│ ← Text cleaning, NLP, Encoding
└─────────────┘
    │
    ▼
┌─────────────┐
│EmbeddingGen │ ← Transformer model, Caching
└─────────────┘
    │
    ▼
┌─────────────┐
│ Classifiers │ ← KNN (FAISS) + TF-IDF
└─────────────┘
    │
    ▼
┌─────────────┐
│  Prediction │ ← Labels, Confidence, Neighbors
└─────────────┘
    │
    ▼
┌─────────────┐
│  Evaluation │ ← Metrics, Plots, Analysis
└─────────────┘
    │
    ▼
┌─────────────┐
│   Output    │ ← Results, Logs, Files
└─────────────┘
```

---

*📝 **Note**: This architecture diagram provides a comprehensive visual overview of the Spam Email Classifier system design, data flow, and component interactions.*
