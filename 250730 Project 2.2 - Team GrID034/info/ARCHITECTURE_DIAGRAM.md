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
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │ Cache Priority  │  │ User Corrections│  │ Terminal Logging│           │
│  │ Logic           │  │ Handling        │  │ for Verification│           │
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
│  │  - Merge corr.  │  │ - Cache suffix  │  │ - Cache suffix  │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │TFIDFClassifier  │  │  Model Cache    │  │  Error Handler  │           │
│  │                 │  │                 │  │                 │           │
│  │ - TF-IDF        │  │ - Embeddings   │  │ - Logging       │           │
│  │ - Naive Bayes   │  │ - Models       │  │ - Exception     │           │
│  │ - Baseline      │  │ - Tokenizers   │  │   handling      │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │Corrections Cache│  │ FAISS Index     │  │ Cache Priority  │           │
│  │                 │  │ Management      │  │ Logic           │           │
│  │ - Save dataset  │  │ - Save index    │  │ - Corrections   │           │
│  │ - Load dataset  │  │ - Load index    │  │   first         │           │
│  │ - Merge data    │  │ - Cache suffix  │  │ - Fallback      │           │
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
│  │ - Corrections│   │ - Merge     │    │ - Suffix    │    │ - Priority  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   OUTPUT    │◀───│  EVALUATE   │◀───│  PREDICT    │◀───│  TRAIN      │ │
│  │             │    │             │    │             │    │             │ │
│  │ - Results   │    │ - Metrics   │    │ - Labels    │    │ - Models    │ │
│  │ - Logs      │    │ - Plots     │    │ - Confidence│    │ - Cache     │ │
│  │ - Files     │    │ - Analysis  │    │ - Neighbors │    │ - Index     │ │
│  │ - Terminal  │    │ - Terminal  │    │ - Terminal  │    │ - Priority  │ │
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
│ 🔄 Corrections  │  │ 🧹 Punctuation  │  │ 🔄 Normalize    │  │ 🔄 Merge Data   │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLEANED DATA                                    │
│  - Preprocessed text messages                                             │
│  - Encoded labels (0/1)                                                  │
│  - Metadata for evaluation                                                │
│  - Merged corrections dataset (cached)                                   │
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
│ 🔄 Cache suffix │  │ 🔄 Cache suffix │  │ 🧠 768-dim      │  │ 🔄 Priority     │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDINGS                                      │
│  - 768-dimensional vectors                                               │
│  - Normalized and cached (with suffixes)                                │
│  - Ready for similarity search                                           │
│  - Priority system: corrections > original                               │
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
│  │ 💾 Cache suffix │                    │                 │               │
│  │ 🔄 Priority     │                    │                 │               │
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
│  │  📊 Terminal Logging: Cache usage                                 │   │
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
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  CACHE PRIORITY │  │  TERMINAL LOG   │  │  CORRECTIONS    │  │  FALLBACK       │
│                 │  │                 │  │                 │  │                 │
│ 🔄 Check corr.  │  │ 📊 Log cache    │  │ 🎯 Use corr.    │  │ 🔄 Use original │
│ 🔄 Check orig.  │  │ 📊 Log FAISS    │  │ 📊 Log decision │  │ 📊 Log fallback │
│ 🔄 Priority     │  │ 📊 Log usage    │  │ 📊 Log success  │  │ 📊 Log usage    │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMAIL MANAGEMENT                                │
│  - Automatic classification                                               │
│  - Gmail label application                                               │
│  - Local file storage                                                    │
│  - Real-time monitoring                                                  │
│  - Cache priority system                                                 │
│  - Terminal logging for verification                                     │
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
│ 🔄 Original     │  │ 🔄 Original     │  │ 📈 F1-Score     │  │ 📊 Error analysis│
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION RESULTS                              │
│  - Performance metrics (JSON)                                             │
│  - Visualization plots (PNG)                                              │
│  - Error analysis                                                         │
│  - Model comparison                                                       │
│  - Original dataset evaluation                                            │
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
│  │   ├── embeddings_intfloat_multilingual-e5-base_original.npy          │
│  │   └── embeddings_intfloat_multilingual-e5-base_with_corrections.npy  │
│  │                                                                       │
│  ├── datasets/                                                           │
│  │   └── with_corrections_dataset_intfloat_multilingual-e5-base.pkl     │
│  │                                                                       │
│  ├── faiss_index/                                                        │
│  │   ├── faiss_index_intfloat_multilingual-e5-base_original.faiss       │
│  │   ├── faiss_index_intfloat_multilingual-e5-base_original.pkl         │
│  │   ├── faiss_index_intfloat_multilingual-e5-base_with_corrections.faiss│
│  │   └── faiss_index_intfloat_multilingual-e5-base_with_corrections.pkl │
│  │                                                                       │
│  └── models/                                                             │
│      ├── model_intfloat_multilingual-e5-base.joblib                     │
│      └── tokenizer_intfloat_multilingual-e5-base.joblib                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **Cache Priority System**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CACHE PRIORITY LOGIC                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  GMAIL SCAN     │  │  TRAINING       │  │  FAISS INDEX    │  │  TERMINAL LOG   │
│                 │  │                 │  │                 │  │                 │
│ 🔄 Check corr.  │  │ 🔄 train_with_  │  │ 🔄 Load corr.   │  │ 📊 Log decision │
│ 🔄 Check orig.  │  │   corrections() │  │ 🔄 Load orig.   │  │ 📊 Log FAISS    │
│ 🔄 Priority     │  │ 🔄 train()      │  │ 🔄 Save suffix  │  │ 📊 Log usage    │
│ 🔄 Fallback     │  │ 🔄 Save cache   │  │ 🔄 Consistency  │  │ 📊 Log success  │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
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
                                    │
                                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ CACHE PRIORITY  │  │ CORRECTIONS     │  │ FAISS INDEX     │  │ TERMINAL LOG    │
│                 │  │                 │  │                 │  │                 │
│ 🔄 Corrections  │  │ 🔄 Merge data   │  │ 🔄 Save index   │  │ 📊 Log cache    │
│ 🔄 Original     │  │ 🔄 Cache dataset│  │ 🔄 Load index   │  │ 📊 Log FAISS    │
│ 🔄 Fallback     │  │ 🔄 Preprocess   │  │ 🔄 Consistency  │  │ 📊 Log usage    │
│ 🔄 Verification │  │ 🔄 Stable cache │  │ 🔄 Priority     │  │ 📊 Log success  │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 🔄 DATA FLOW SUMMARY

```
INPUT DATA
    │
    ▼
┌─────────────┐
│  DataLoader │ ← CSV, Local files, Gmail API, Corrections
└─────────────┘
    │
    ▼
┌─────────────┐
│Preprocessing│ ← Text cleaning, NLP, Encoding, Merge corrections
└─────────────┘
    │
    ▼
┌─────────────┐
│EmbeddingGen │ ← Transformer model, Caching with suffixes
└─────────────┘
    │
    ▼
┌─────────────┐
│ Classifiers │ ← KNN (FAISS) + TF-IDF, Cache priority
└─────────────┘
    │
    ▼
┌─────────────┐
│  Prediction │ ← Labels, Confidence, Neighbors, Terminal logging
└─────────────┘
    │
    ▼
┌─────────────┐
│  Evaluation │ ← Metrics, Plots, Analysis
└─────────────┘
    │
    ▼
┌─────────────┐
│   Output    │ ← Results, Logs, Files, Terminal verification
└─────────────┘
```

---

*📝 **Note**: This architecture diagram provides a comprehensive visual overview of the Spam Email Classifier system design, data flow, and component interactions with the latest cache priority system and corrections handling.*
