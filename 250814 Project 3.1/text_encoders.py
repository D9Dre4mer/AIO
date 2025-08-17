"""
Text encoders module for Topic Modeling Project
Handles different text vectorization methods: BoW, TF-IDF, and Word Embeddings
"""

from typing import List, Literal
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDING_NORMALIZE, EMBEDDING_DEVICE


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

    def transform(
        self,
        texts: List[str],
        mode: Literal['query', 'passage'] = 'query'
    ) -> List[List[float]]:
        """Transform texts to embeddings"""
        if mode == 'raw':
            inputs = texts
        else:
            inputs = self._format_inputs(texts, mode)

        embeddings = self.model.encode(
            inputs, 
            normalize_embeddings=self.normalize
        )
        return embeddings.tolist()

    def transform_numpy(
        self,
        texts: List[str],
        mode: Literal['query', 'passage'] = 'query'
    ) -> np.ndarray:
        """Transform texts to numpy array of embeddings"""
        return np.array(self.transform(texts, mode=mode))


class TextVectorizer:
    """Class for handling different text vectorization methods"""
    
    def __init__(self):
        self.bow_vectorizer = CountVectorizer()
        self.tfidf_vectorizer = TfidfVectorizer()
        self.embedding_vectorizer = EmbeddingVectorizer()
        
    def fit_transform_bow(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts using Bag of Words"""
        vectors = self.bow_vectorizer.fit_transform(texts)
        return np.array(vectors.toarray())
        
    def transform_bow(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted Bag of Words vectorizer"""
        vectors = self.bow_vectorizer.transform(texts)
        return np.array(vectors.toarray())
        
    def fit_transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts using TF-IDF"""
        vectors = self.tfidf_vectorizer.fit_transform(texts)
        return np.array(vectors.toarray())
        
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted TF-IDF vectorizer"""
        vectors = self.tfidf_vectorizer.transform(texts)
        return np.array(vectors.toarray())
        
    def transform_embeddings(self, texts: List[str]) -> np.ndarray:
        """Transform texts using word embeddings"""
        return self.embedding_vectorizer.transform_numpy(texts)
        
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
