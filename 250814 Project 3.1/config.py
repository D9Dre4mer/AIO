"""
Configuration file for Topic Modeling Project
Contains constants and settings used throughout the project
"""

# Cache directory for datasets
CACHE_DIR = "./cache"

# Categories to select for analysis
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']

# Number of samples to load
MAX_SAMPLES = 100000

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
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'
EMBEDDING_NORMALIZE = True
EMBEDDING_DEVICE = 'auto'

# Output directories
FIGURES_DIR = "pdf/Figures"
