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
MAX_VOCABULARY_SIZE = 50000   # Maximum vocabulary for BoW/TF-IDF

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

# Output directories
FIGURES_DIR = "pdf/Figures"
