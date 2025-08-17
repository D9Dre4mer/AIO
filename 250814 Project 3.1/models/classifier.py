# =========================================
# Academic Paper Classifier
# Core classification engine with multiple algorithms
# =========================================

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class AcademicPaperClassifier:
    """
    High-performance academic paper classifier using ensemble methods
    and state-of-the-art ML techniques.
    """
    
    def __init__(self, model_type: str = "Ensemble", 
                 confidence_threshold: float = 0.8):
        """
        Initialize the classifier with specified model type.
        
        Args:
            model_type: Type of classification model to use
            confidence_threshold: Minimum confidence for classification
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.models = {}
        self.vectorizer = None
        self.is_trained = False
        
        # Classification categories
        self.research_domains = [
            'Computer Science', 'Medicine', 'Physics', 'Chemistry', 
            'Biology', 'Economics', 'Psychology', 'Engineering',
            'Mathematics', 'Social Sciences', 'Arts & Humanities'
        ]
        
        self.publication_types = [
            'Research Article', 'Review', 'Case Study', 'Methodology',
            'Short Communication', 'Editorial', 'Letter to Editor'
        ]
        
        self.quality_levels = ['High Impact', 'Medium Impact', 'Low Impact']
        
        self.methodologies = [
            'Quantitative', 'Qualitative', 'Mixed Methods', 
            'Experimental', 'Observational', 'Theoretical'
        ]
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available classification models."""
        
        # Traditional ML models
        self.models['SVM'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, 
                                      ngram_range=(1, 3))),
            ('classifier', SVC(probability=True, random_state=42))
        ])
        
        self.models['Random Forest'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, 
                                      ngram_range=(1, 3))),
            ('classifier', RandomForestClassifier(n_estimators=200, 
                                                random_state=42))
        ])
        
        self.models['Naive Bayes'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, 
                                      ngram_range=(1, 3))),
            ('classifier', MultinomialNB())
        ])
        
        self.models['Neural Network'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, 
                                      ngram_range=(1, 3))),
            ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), 
                                        random_state=42, max_iter=500))
        ])
        
        # Ensemble model (combination of best models)
        self.models['Ensemble'] = self._create_ensemble_model()
        
        # BERT-based model (if transformers available)
        try:
            self.models['BERT'] = self._create_bert_model()
        except ImportError:
            print("Warning: Transformers not available. BERT model disabled.")
    
    def _create_ensemble_model(self) -> Pipeline:
        """Create an ensemble model combining multiple classifiers."""
        
        # Create base models for ensemble
        base_models = [
            ('svm', SVC(probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
            ('nb', MultinomialNB()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), 
                                 random_state=42, max_iter=500))
        ]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=-1
        )
        
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, 
                                      ngram_range=(1, 3))),
            ('ensemble', ensemble)
        ])
    
    def _create_bert_model(self):
        """Create a BERT-based classification model."""
        try:
            # Use a pre-trained BERT model for text classification
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(self.research_domains)
            )
            
            # Create a custom BERT classifier
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True
            )
        except Exception as e:
            print(f"Error creating BERT model: {e}")
            return None
    
    def train(self, texts: List[str], labels: List[str], 
              category: str = 'research_domain') -> Dict[str, float]:
        """
        Train the classifier with provided data.
        
        Args:
            texts: List of academic paper texts
            labels: List of corresponding labels
            category: Category to classify (research_domain, publication_type, etc.)
            
        Returns:
            Dictionary containing training metrics
        """
        
        if not texts or not labels:
            raise ValueError("Texts and labels cannot be empty")
        
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
        
        # Select model based on model_type
        if self.model_type not in self.models:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model = self.models[self.model_type]
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train the model
        if self.model_type == 'BERT':
            # BERT training requires special handling
            self._train_bert_model(model, X_train, y_train, X_test, y_test)
        else:
            # Traditional ML model training
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            confidence_scores = np.max(y_pred_proba, axis=1)
            avg_confidence = np.mean(confidence_scores)
            
            # Store the trained model
            self.current_model = model
            self.is_trained = True
            
            # Return metrics
            return {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'training_samples': len(X_train),
                'validation_samples': len(X_test),
                'model_type': self.model_type
            }
    
    def _train_bert_model(self, model, X_train, y_train, X_test, y_test):
        """Train BERT model with custom training loop."""
        # This is a simplified BERT training implementation
        # In practice, you would use a more sophisticated training loop
        print("BERT model training not fully implemented in this version")
        print("Using pre-trained BERT for inference only")
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a single academic paper text.
        
        Args:
            text: Academic paper text to classify
            
        Returns:
            Dictionary containing classification results and confidence scores
        """
        
        if not self.is_trained and self.model_type != 'BERT':
            raise RuntimeError("Model must be trained before classification")
        
        if not text or len(text.strip()) == 0:
            return {
                'error': 'Empty or invalid text provided',
                'confidence': 0.0
            }
        
        try:
            if self.model_type == 'BERT':
                return self._classify_with_bert(text)
            else:
                return self._classify_with_ml(text)
                
        except Exception as e:
            return {
                'error': f'Classification failed: {str(e)}',
                'confidence': 0.0
            }
    
    def _classify_with_ml(self, text: str) -> Dict[str, Any]:
        """Classify using traditional ML models."""
        
        if not hasattr(self, 'current_model'):
            raise RuntimeError("No trained model available")
        
        # Make prediction
        prediction = self.current_model.predict([text])[0]
        probabilities = self.current_model.predict_proba([text])[0]
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            prediction = 'Uncertain'
            confidence = 0.0
        
        return {
            'research_domain': prediction,
            'publication_type': 'Research Article',  # Default
            'quality_level': 'Medium Impact',  # Default
            'methodology': 'Mixed Methods',  # Default
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'model_type': self.model_type
        }
    
    def _classify_with_bert(self, text: str) -> Dict[str, Any]:
        """Classify using BERT model."""
        
        if (self.model_type not in self.models or 
                self.models[self.model_type] is None):
            return {
                'error': 'BERT model not available',
                'confidence': 0.0
            }
        
        try:
            # Use BERT pipeline for classification
            results = self.models[self.model_type](text)
            
            # Extract predictions and confidence scores
            predictions = []
            confidences = []
            
            for result in results[0]:
                predictions.append(result['label'])
                confidences.append(result['score'])
            
            # Get the most confident prediction
            max_confidence_idx = np.argmax(confidences)
            prediction = predictions[max_confidence_idx]
            confidence = confidences[max_confidence_idx]
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                prediction = 'Uncertain'
                confidence = 0.0
            
            return {
                'research_domain': prediction,
                'publication_type': 'Research Article',
                'quality_level': 'Medium Impact',
                'methodology': 'Mixed Methods',
                'confidence': confidence,
                'probabilities': confidences,
                'model_type': 'BERT'
            }
            
        except Exception as e:
            return {
                'error': f'BERT classification failed: {str(e)}',
                'confidence': 0.0
            }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple academic paper texts.
        
        Args:
            texts: List of academic paper texts
            
        Returns:
            List of classification results
        """
        
        if not texts:
            return []
        
        results = []
        for text in texts:
            result = self.classify(text)
            results.append(result)
        
        return results
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate the classifier performance.
        
        Args:
            texts: List of test texts
            true_labels: List of true labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Make predictions
        predictions = []
        for text in texts:
            result = self.classify(text)
            predictions.append(result.get('research_domain', 'Unknown'))
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Generate detailed report
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'detailed_report': report
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        
        if not self.is_trained:
            raise RuntimeError("No trained model to save")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        joblib.dump(self.current_model, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the model
        self.current_model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from: {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'confidence_threshold': self.confidence_threshold,
            'available_categories': {
                'research_domains': self.research_domains,
                'publication_types': self.publication_types,
                'quality_levels': self.quality_levels,
                'methodologies': self.methodologies
            },
            'model_parameters': {
                'vectorizer_features': 10000,
                'ngram_range': (1, 3),
                'ensemble_voting': 'soft'
            }
        }
