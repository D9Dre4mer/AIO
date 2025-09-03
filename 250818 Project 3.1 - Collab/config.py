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

# Embedding model configuration
EMBEDDING_MODEL_NAME = 'sentence-transformers/allenai-specter'  # 768d
EMBEDDING_NORMALIZE = True
EMBEDDING_DEVICE = 'auto'

# Output directories
FIGURES_DIR = "pdf/Figures"
