# =========================================
# Text Processor for Academic Papers
# Preprocessing utilities for improved classification
# =========================================

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception:
    pass


class TextProcessor:
    """
    Advanced text preprocessing for academic papers.
    Implements multiple preprocessing strategies for optimal classification.
    """
    
    def __init__(self):
        """Initialize text processor with NLTK components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Academic-specific stop words
        self.academic_stop_words = {
            'abstract', 'introduction', 'conclusion', 'methodology',
            'results', 'discussion', 'references', 'appendix',
            'figure', 'table', 'equation', 'section', 'chapter',
            'paper', 'study', 'research', 'analysis', 'data',
            'model', 'method', 'approach', 'framework', 'system'
        }
        
        # Update stop words with academic terms
        self.stop_words.update(self.academic_stop_words)
    
    def preprocess(self, text: str, strategy: str = 'comprehensive') -> str:
        """
        Preprocess text using specified strategy.
        
        Args:
            text: Raw text to preprocess
            strategy: Preprocessing strategy ('basic', 'advanced', 'comprehensive')
            
        Returns:
            Preprocessed text
        """
        
        if not text or len(text.strip()) == 0:
            return ""
        
        if strategy == 'basic':
            return self._basic_preprocessing(text)
        elif strategy == 'advanced':
            return self._advanced_preprocessing(text)
        elif strategy == 'comprehensive':
            return self._comprehensive_preprocessing(text)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _basic_preprocessing(self, text: str) -> str:
        """Basic text preprocessing."""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove stop words
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def _advanced_preprocessing(self, text: str) -> str:
        """Advanced text preprocessing with lemmatization."""
        
        # Apply basic preprocessing
        text = self._basic_preprocessing(text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        # Remove short words (length < 3)
        words = [word for word in words if len(word) >= 3]
        
        return ' '.join(words)
    
    def _comprehensive_preprocessing(self, text: str) -> str:
        """Comprehensive preprocessing for academic papers."""
        
        # Extract main content (remove headers, footers, references)
        text = self._extract_main_content(text)
        
        # Apply advanced preprocessing
        text = self._advanced_preprocessing(text)
        
        # Extract key phrases and technical terms
        text = self._extract_key_phrases(text)
        
        # Normalize academic terminology
        text = self._normalize_academic_terms(text)
        
        return text
    
    def _extract_main_content(self, text: str) -> str:
        """Extract main content from academic paper."""
        
        # Remove common academic paper sections
        sections_to_remove = [
            r'abstract.*?introduction',
            r'references.*',
            r'appendix.*',
            r'acknowledgments.*',
            r'conflicts of interest.*',
            r'funding.*',
            r'data availability.*'
        ]
        
        for pattern in sections_to_remove:
            text = re.sub(pattern, '', text, 
                         flags=re.IGNORECASE | re.DOTALL)
        
        # Remove figure and table captions
        text = re.sub(r'figure \d+.*?\.', '', text, 
                     flags=re.IGNORECASE)
        text = re.sub(r'table \d+.*?\.', '', text, 
                     flags=re.IGNORECASE)
        
        # Remove equations
        text = re.sub(r'\$.*?\$', '', text)
        
        return text
    
    def _extract_key_phrases(self, text: str) -> str:
        """Extract key phrases and technical terms."""
        
        # Extract noun phrases (simplified)
        words = text.split()
        key_phrases = []
        
        for i, word in enumerate(words):
            # Keep technical terms (words with capital letters)
            if any(c.isupper() for c in word):
                key_phrases.append(word)
            
            # Keep compound terms (words with hyphens or underscores)
            elif '-' in word or '_' in word:
                key_phrases.append(word)
            
            # Keep words that appear to be technical terms
            elif len(word) > 5 and word.isalpha():
                key_phrases.append(word)
        
        return ' '.join(key_phrases)
    
    def _normalize_academic_terms(self, text: str) -> str:
        """Normalize academic terminology."""
        
        # Common academic term mappings
        term_mappings = {
            'machine learning': 'ml',
            'artificial intelligence': 'ai',
            'deep learning': 'dl',
            'natural language processing': 'nlp',
            'computer vision': 'cv',
            'data science': 'ds',
            'big data': 'bigdata',
            'internet of things': 'iot',
            'blockchain': 'blockchain',
            'cloud computing': 'cloud',
            'cybersecurity': 'security',
            'quantum computing': 'quantum'
        }
        
        # Apply mappings
        for full_term, short_term in term_mappings.items():
            text = text.replace(full_term, short_term)
        
        return text
    
    def extract_features(self, text: str) -> dict:
        """
        Extract various text features for analysis.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary containing text features
        """
        
        if not text:
            return {}
        
        # Basic statistics
        words = text.split()
        sentences = sent_tokenize(text)
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_words': len(set(words)),
            'vocabulary_diversity': len(set(words)) / max(len(words), 1)
        }
        
        # Technical term analysis
        technical_terms = self._count_technical_terms(text)
        features.update(technical_terms)
        
        # Academic style indicators
        style_indicators = self._analyze_academic_style(text)
        features.update(style_indicators)
        
        return features
    
    def _count_technical_terms(self, text: str) -> dict:
        """Count technical terms and jargon."""
        
        words = text.split()
        
        # Count technical indicators
        technical_count = sum(1 for word in words if len(word) > 8)
        acronym_count = sum(1 for word in words 
                           if word.isupper() and len(word) > 1)
        compound_count = sum(1 for word in words 
                            if '-' in word or '_' in word)
        
        return {
            'technical_terms': technical_count,
            'acronyms': acronym_count,
            'compound_terms': compound_count
        }
    
    def _analyze_academic_style(self, text: str) -> dict:
        """Analyze academic writing style."""
        
        # Count formal language indicators
        formal_indicators = [
            'furthermore', 'moreover', 'additionally', 'consequently',
            'therefore', 'thus', 'hence', 'nevertheless', 'however',
            'although', 'despite', 'regarding', 'concerning'
        ]
        
        formal_count = sum(1 for indicator in formal_indicators 
                          if indicator in text.lower())
        
        # Count citation patterns
        citation_patterns = [
            r'\(\w+ et al\.', r'\(\w+, \d{4}\)', r'\[\d+\]',
            r'\(\w+ and \w+', r'\(\w+ & \w+'
        ]
        
        citation_count = sum(len(re.findall(pattern, text)) 
                            for pattern in citation_patterns)
        
        return {
            'formal_language': formal_count,
            'citations': citation_count
        }
    
    def clean_references(self, text: str) -> str:
        """Remove reference sections and citations."""
        
        # Remove reference patterns
        patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\w+ et al\.',  # (Author et al.
            r'\(\w+, \d{4}\)',  # (Author, 2024)
            r'\(\w+ and \w+',  # (Author and Author
            r'\(\w+ & \w+'     # (Author & Author
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    def extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract section from academic paper."""
        
        # Common abstract patterns
        abstract_patterns = [
            r'abstract\s*[:.]?\s*(.*?)(?=introduction|introduction|$)',
            r'abstract\s*[:.]?\s*(.*?)(?=keywords|keywords|$)',
            r'abstract\s*[:.]?\s*(.*?)(?=1\.|1\.|$)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract
        
        return None
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from academic paper."""
        
        # Look for keyword sections
        keyword_patterns = [
            r'keywords\s*[:.]?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'key words\s*[:.]?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'index terms\s*[:.]?\s*(.*?)(?=\n\n|\n[A-Z]|$)'
        ]
        
        keywords = []
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                keyword_text = match.group(1).strip()
                # Split by common separators
                keywords = re.split(r'[,;]|\band\b', keyword_text)
                keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
                break
        
        return keywords
    
    def get_preprocessing_summary(self, original_text: str, 
                                processed_text: str) -> dict:
        """
        Get summary of preprocessing changes.
        
        Args:
            original_text: Original text
            processed_text: Processed text
            
        Returns:
            Dictionary with preprocessing statistics
        """
        
        original_words = len(original_text.split())
        processed_words = len(processed_text.split())
        
        reduction = ((original_words - processed_words) / 
                    max(original_words, 1)) * 100
        
        return {
            'original_word_count': original_words,
            'processed_word_count': processed_words,
            'reduction_percentage': reduction,
            'compression_ratio': processed_words / max(original_words, 1)
        }
