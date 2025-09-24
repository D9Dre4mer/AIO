"""
Configuration file for Topic Modeling Project
Contains constants and settings used throughout the project
"""

# Cache directory for datasets
CACHE_DIR = "./cache"

# Dynamic category selection (will be set at runtime)
# CATEGORIES_TO_SELECT removed - now handled dynamically by DataLoader

# Number of samples to load (None = respect user choice from sampling config)
MAX_SAMPLES = None

# Test split ratio
TEST_SIZE = 0.2

# Random state for reproducibility
RANDOM_STATE = 42

# Model parameters
KMEANS_N_CLUSTERS = 5
KNN_N_NEIGHBORS = 5

# Memory optimization thresholds
KMEANS_SVD_THRESHOLD = 20000  # Use SVD if features > 20K (was 10K)
KMEANS_SVD_COMPONENTS = 2000  # Reduce to 2K dimensions (was 1K)
MAX_VOCABULARY_SIZE = 30000   # Maximum vocabulary for BoW/TF-IDF (reduced for 300k samples)

# BoW/TF-IDF SVD dimensionality reduction for speed optimization
BOW_TFIDF_SVD_COMPONENTS = 400  # Reduce to 400 dimensions (300-500 range)
BOW_TFIDF_SVD_THRESHOLD = 200   # Apply SVD if features > 200

# Embedding model configuration
EMBEDDING_MODEL_NAME = 'sentence-transformers/allenai-specter'  # 768d
EMBEDDING_NORMALIZE = True
EMBEDDING_DEVICE = 'auto'

# GPU Optimization Settings
ENABLE_GPU_OPTIMIZATION = False  # Use sparse matrices (memory efficient)
FORCE_DENSE_CONVERSION = False   # Force sparse->dense conversion for GPU

# RAPIDS cuML Settings
ENABLE_RAPIDS_CUML = True        # Enable RAPIDS cuML for GPU acceleration
RAPIDS_FALLBACK_TO_CPU = True    # Fallback to CPU if GPU not available
RAPIDS_AUTO_DETECT_DEVICE = True # Automatically detect best device (GPU/CPU)

# CPU Multithreading Settings
CPU_N_JOBS = -1  # Use all available CPU cores (-1 = all cores)
CPU_MAX_JOBS = 8  # Maximum number of parallel jobs (safety limit)
CPU_OPTIMIZATION = True  # Enable CPU multithreading optimization

# Output directories
FIGURES_DIR = "pdf/Figures"

# Enhanced ML Configuration
# Device Policy
DEVICE_POLICY = "gpu_first"  # "gpu_first" | "cpu_only"

# Optuna Configuration
OPTUNA_ENABLE = True
OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT = None  # seconds, None for no timeout
OPTUNA_DIRECTION = "maximize"

# SHAP Configuration
SHAP_ENABLE = True
SHAP_SAMPLE_SIZE = 5000
SHAP_OUTPUT_DIR = "info/Result/"

# Stacking Configuration
STACKING_ENABLE = False
STACKING_REQUIRE_MIN_BASE_MODELS = 4
STACKING_BASE_MODELS = ["lightgbm", "xgboost", "catboost", "random_forest"]
STACKING_META_LEARNER = "logistic_regression"  # "logistic_regression" | "lightgbm"
STACKING_USE_ORIGINAL_FEATURES = False
STACKING_CV_N_SPLITS = 5
STACKING_CV_STRATIFIED = True
STACKING_CACHE_OUTPUT_DIR = "cache/stacking/"
STACKING_CACHE_FORMAT = "parquet"  # "parquet" | "csv"

# Cache Configuration
CACHE_MODELS_ROOT_DIR = "cache/models/"
CACHE_STACKING_ROOT_DIR = "cache/stacking/"
CACHE_FORCE_RETRAIN = False
CACHE_USE_CACHE = True

# Data Processing Configuration
DATA_PROCESSING_AUTO_DETECT_TYPES = True
DATA_PROCESSING_NUMERIC_SCALER = "standard"  # "standard" | "minmax" | "robust"
DATA_PROCESSING_TEXT_ENCODING = "label"  # "label" | "onehot" | "target"
DATA_PROCESSING_HANDLE_MISSING_NUMERIC = "mean"  # "mean" | "median" | "mode" | "drop"
DATA_PROCESSING_HANDLE_MISSING_TEXT = "mode"  # "mean" | "median" | "mode" | "drop"
DATA_PROCESSING_OUTLIER_METHOD = "iqr"  # "iqr" | "zscore" | "isolation_forest" | "none"

# Evaluation Configuration
EVALUATION_CONFUSION_MATRIX_ENABLE = True
EVALUATION_CONFUSION_MATRIX_DATASET = "test"  # "test" | "val"
EVALUATION_CONFUSION_MATRIX_NORMALIZE = True  # True | False | "pred" | "all"
EVALUATION_CONFUSION_MATRIX_THRESHOLD = 0.5
EVALUATION_CONFUSION_MATRIX_LABELS_ORDER = []  # Empty list for auto-detection

# Global Settings
SEED = 42
N_JOBS = -1