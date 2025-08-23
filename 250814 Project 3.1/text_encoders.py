"""
Text encoders module for Topic Modeling Project
Handles different text vectorization methods: BoW, TF-IDF, and Word Embeddings
"""

from typing import List, Literal
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_NORMALIZE, EMBEDDING_DEVICE,
    MAX_VOCABULARY_SIZE
)


class EmbeddingVectorizer:
    """Class for generating word embeddings using pre-trained models"""
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        normalize: bool = EMBEDDING_NORMALIZE,
        device: str = EMBEDDING_DEVICE
    ):
        # Auto-detect device if not specified
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model with GPU support
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name  # Store model name for later access
        self.normalize = normalize
        
        # Print device info for confirmation
        print(f"🚀 EmbeddingVectorizer initialized on device: {self.device}")

    def _format_inputs(
        self,
        texts: List[str],
        mode: Literal['query', 'passage']
    ) -> List[str]:
        """Format inputs based on mode"""
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {text.strip()}" for text in texts]

    def fit(self, texts: List[str]) -> 'EmbeddingVectorizer':
        """Fit the embedding model on training data (for consistency with sklearn API)"""
        # Note: SentenceTransformer doesn't need fitting, but we mark it as fitted
        # and store training data statistics for validation
        self.is_fitted = True
        self.training_stats = {
            'n_samples': len(texts),
            'n_features': self.model.get_sentence_embedding_dimension(),
            'model_name': self.model_name if hasattr(self, 'model_name') else 'unknown'
        }
        # Remove duplicate log - parent method already logs
        return self
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit on training data and transform to embeddings"""
        self.fit(texts)
        return self.transform(texts)
    
    def transform(
        self,
        texts: List[str],
        mode: Literal['query', 'passage'] = 'query'
    ) -> np.ndarray:
        """Transform texts to embeddings (must be fitted first)"""
        if not self.is_fitted:
            raise ValueError("EmbeddingVectorizer must be fitted before transform")
        
        if mode == 'raw':
            inputs = texts
        else:
            inputs = self._format_inputs(texts, mode)

        embeddings = self.model.encode(
            inputs, 
            normalize_embeddings=self.normalize,
            show_progress_bar=False  # Disable built-in progress bar
        )
        return embeddings

    def transform_numpy(
        self,
        texts: List[str],
        mode: Literal['query', 'passage'] = 'query'
    ) -> np.ndarray:
        """Transform texts to numpy array of embeddings with progress tracking"""
        return np.array(self.transform_with_progress(texts, mode=mode, stop_callback=None))
    
    def transform_with_progress(
        self,
        texts: List[str],
        mode: Literal['query', 'passage'] = 'query',
        batch_size: int = 100,
        stop_callback=None
    ) -> List[List[float]]:
        """Transform texts to embeddings with progress bar"""
        import time
        
        total_texts = len(texts)
        print(f"🔧 Processing {total_texts:,} texts for embeddings...")
        
        if mode == 'raw':
            inputs = texts
        else:
            inputs = self._format_inputs(texts, mode)
        
        all_embeddings = []
        start_time = time.time()
        
        # Process in batches to show progress
        for i in range(0, total_texts, batch_size):
            # Check if processing should stop
            if stop_callback and stop_callback():
                print(f"\n🛑 Embedding stopped by user request at {i:,}/{total_texts:,}")
                return all_embeddings  # Return partial results
                
            batch_end = min(i + batch_size, total_texts)
            batch_inputs = inputs[i:batch_end]
            
            # Generate embeddings for current batch
            batch_embeddings = self.model.encode(
                batch_inputs,
                normalize_embeddings=self.normalize,
                show_progress_bar=False  # Disable built-in progress bar
            )
            all_embeddings.extend(batch_embeddings.tolist())
            
            # Calculate time estimates
            elapsed_time = time.time() - start_time
            progress_percent = (batch_end / total_texts) * 100
            
            if progress_percent > 0:
                # Estimate total time based on current progress
                estimated_total_time = elapsed_time / (progress_percent / 100)
                remaining_time = estimated_total_time - elapsed_time
                eta_str = self._format_time(remaining_time)
            else:
                eta_str = "calculating..."
            
            # Show custom progress bar with ETA
            progress_bar = self._create_progress_bar(progress_percent, 40)
            progress_text = (f"\r🔄 Embedding Progress: {progress_bar} "
                           f"{progress_percent:5.1f}% "
                           f"({batch_end:,}/{total_texts:,}) "
                           f"⏱️ ETA: {eta_str}")
            print(progress_text, end="", flush=True)
        
        print()  # New line after progress bar
        total_time = time.time() - start_time
        print(f"✅ Embedding completed! "
              f"Generated {len(all_embeddings):,} embeddings "
              f"in {self._format_time(total_time)}")
        return all_embeddings
    
    def _create_progress_bar(self, percentage: float, width: int = 40) -> str:
        """Create a custom progress bar string"""
        filled_width = int(width * percentage / 100)
        # Use beautiful Unicode characters for progress bar
        bar = '█' * filled_width + '░' * (width - filled_width)
        return bar
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class TextVectorizer:
    """Class for handling different text vectorization methods"""
    
    def __init__(self):
        # Limit vocabulary size to prevent memory issues
        self.bow_vectorizer = CountVectorizer(
            max_features=MAX_VOCABULARY_SIZE,  # Configurable vocabulary limit
            min_df=2,           # Ignore words appearing in < 2 documents
            max_df=0.95,        # Ignore words appearing in > 95% documents
            stop_words='english'
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=MAX_VOCABULARY_SIZE,  # Configurable vocabulary limit
            min_df=2,           # Ignore words appearing in < 2 documents  
            max_df=0.95,        # Ignore words appearing in > 95% documents
            stop_words='english'
        )
        self.embedding_vectorizer = EmbeddingVectorizer()
        
    def fit_transform_bow(self, texts: List[str]):
        """Fit and transform texts using Bag of Words (returns sparse matrix)"""
        vectors = self.bow_vectorizer.fit_transform(texts)
        print(f"📊 BoW Features: {vectors.shape[1]:,} | Sparsity: {1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.3f}")
        return vectors  # Keep sparse for memory efficiency
        
    def transform_bow(self, texts: List[str]):
        """Transform texts using fitted Bag of Words vectorizer (returns sparse matrix)"""
        vectors = self.bow_vectorizer.transform(texts)
        return vectors  # Keep sparse for memory efficiency
        
    def fit_transform_tfidf(self, texts: List[str]):
        """Fit and transform texts using TF-IDF (returns sparse matrix)"""
        vectors = self.tfidf_vectorizer.fit_transform(texts)
        print(f"📊 TF-IDF Features: {vectors.shape[1]:,} | Sparsity: {1 - vectors.nnz / (vectors.shape[0] * vectors.shape[1]):.3f}")
        return vectors  # Keep sparse for memory efficiency
        
    def transform_tfidf(self, texts: List[str]):
        """Transform texts using fitted TF-IDF vectorizer (returns sparse matrix)"""
        vectors = self.tfidf_vectorizer.transform(texts)
        return vectors  # Keep sparse for memory efficiency
        
    def fit_transform_embeddings(self, texts: List[str], stop_callback=None) -> np.ndarray:
        """Fit embedding model on training data and transform to embeddings"""
        print(f"🔧 Fitting embedding model on {len(texts):,} training samples...")
        self.embedding_vectorizer.fit(texts)
        return np.array(self.embedding_vectorizer.transform_with_progress(texts, stop_callback=stop_callback))
        
    def fit_embeddings_only(self, texts: List[str]) -> 'EmbeddingVectorizer':
        """Fit embedding model on training data without transforming (for CV)"""
        print(f"🔧 Creating new embedding vectorizer for CV fold ({len(texts):,} samples)...")
        # Create a NEW instance to prevent data leakage
        new_embedding_vectorizer = EmbeddingVectorizer(
            model_name=self.embedding_vectorizer.model_name,
            normalize=self.embedding_vectorizer.normalize,
            device=self.embedding_vectorizer.device
        )
        return new_embedding_vectorizer.fit(texts)
        
    def transform_embeddings(self, texts: List[str], stop_callback=None) -> np.ndarray:
        """Transform texts using fitted embedding model"""
        return np.array(self.embedding_vectorizer.transform_with_progress(texts, stop_callback=stop_callback))
        
    def get_feature_names_bow(self) -> List[str]:
        """Get feature names from BoW vectorizer"""
        return self.bow_vectorizer.get_feature_names_out().tolist()
        
    def get_feature_names_tfidf(self) -> List[str]:
        """Get feature names from TF-IDF vectorizer"""
        return self.tfidf_vectorizer.get_feature_names_out().tolist()


def demonstrate_text_encoders():
    """Demonstrate different text encoding methods"""
    docs = [
        "I am going to school to study for the final exam.",
        "The weather is nice today and I feel happy.",
        "I love programming in Python and exploring new libraries.",
        "Data science is an exciting field with many opportunities.",
    ]
    
    # Bag of Words demonstration
    print("=== Bag of Words (BoW) ===")
    print("BoW is a simple and commonly used text representation technique.")
    print("It converts text into a fixed-length vector by counting the")
    print("occurrences of each word in the text. This method ignores")
    print("grammar and word order but retains the frequency of words.\n")
    
    bow = CountVectorizer()
    vectors = bow.fit_transform(docs)
    
    for i, vec in enumerate(vectors):
        print(f"Document {i+1}: {vec.toarray()}")
    
    print("\n" + "="*50 + "\n")
    
    # TF-IDF demonstration
    print("=== TF-IDF ===")
    print("Tf-idf (Term Frequency-Inverse Document Frequency) is another")
    print("popular text representation technique. It not only considers the")
    print("frequency of words in a document but also how common or rare a")
    print("word is across all documents. This helps to reduce the weight")
    print("of common words and highlight more informative ones.\n")
    
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(docs)
    
    for i, vec in enumerate(tfidf_vectors):
        print(f"TF-IDF for Document {i+1}:")
        print(vec.toarray())
    
    print("\n" + "="*50 + "\n")
    
    # Word Embeddings demonstration
    print("=== Word Embeddings ===")
    print("Word embeddings are dense vector representations of words that")
    print("capture semantic relationships between them. They are typically")
    print("pre-trained on large corpora and can be used to represent words")
    print("in a continuous vector space, allowing for better generalization")
    print("and understanding of word meanings.\n")
    
    vectorizer = EmbeddingVectorizer()
    embeddings = vectorizer.transform(docs)
    
    for i, emb in enumerate(embeddings):
        print(f"Embedding for Document {i+1}:")
        print(emb[:10])  # Print first 10 dimensions for brevity
        print("#" * 20)
