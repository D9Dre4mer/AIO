"""
Configuration file for Topic Modeling Project
Contains constants and settings used throughout the project
"""

# Cache directory for datasets
CACHE_DIR = "./cache"

# Categories to select for analysis
CATEGORIES_TO_SELECT = ['astro-ph', 'cond-mat', 'cs', 'math', 'physics']

# Number of samples to load
MAX_SAMPLES = 1000

# Test split ratio
TEST_SIZE = 0.2

# Random state for reproducibility
RANDOM_STATE = 42

# Model parameters
KMEANS_N_CLUSTERS = 5
KNN_N_NEIGHBORS = 5

# Embedding model configuration
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'
EMBEDDING_NORMALIZE = True
EMBEDDING_DEVICE = 'auto'

# Output directories
FIGURES_DIR = "pdf/Figures"
