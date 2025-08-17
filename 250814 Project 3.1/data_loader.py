"""
Data loader module for Topic Modeling Project
Handles dataset loading, preprocessing, and text cleaning
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import CACHE_DIR, CATEGORIES_TO_SELECT, MAX_SAMPLES, TEST_SIZE, RANDOM_STATE


class DataLoader:
    """Class for loading and preprocessing the ArXiv abstracts dataset"""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.dataset = None
        self.samples = []
        self.preprocessed_samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        
    def load_dataset(self) -> None:
        """Load the ArXiv abstracts dataset from HuggingFace"""
        dataset_cache_path = Path(self.cache_dir) / "UniverseTBD___arxiv-abstracts-large"
        
        if dataset_cache_path.exists():
            print(f"✅ Dataset found in cache: {dataset_cache_path}")
            print("📁 Loading from cache (no download needed)...")
        else:
            print(f"📥 Dataset not found in cache: {dataset_cache_path}")
            print("🌐 Will download dataset to cache...")
        
        # Load the dataset from the specified cache directory
        self.dataset = load_dataset("UniverseTBD/arxiv-abstracts-large", cache_dir=self.cache_dir)
        
        print(f"🎉 Dataset loaded successfully!")
        print(f"📊 Dataset info: {self.dataset}")
        
    def print_sample_examples(self, num_examples: int = 3) -> None:
        """Print sample examples from the dataset"""
        for i in range(num_examples):
            print(f"Example {i+1}:")
            print(self.dataset['train'][i]['abstract'])
            print(self.dataset['train'][i]['categories'])
            print("---" * 20)
            
    def get_all_categories(self, sample_size: int = 1000) -> set:
        """Get all unique categories from a sample of the dataset"""
        all_categories = self.dataset['train']['categories'][:sample_size]
        return set(all_categories)
        
    def get_primary_categories(self) -> List[str]:
        """Get all unique primary categories from the entire dataset"""
        all_categories = self.dataset['train']['categories']
        category_set = set()
        
        # Collect unique labels
        for category in all_categories:
            parts = category.split(' ')
            for part in parts:
                topic = part.split('.')[0]
                category_set.add(topic)
        
        # Sort the labels and return them
        sorted_categories = sorted(list(category_set), key=lambda x: x.lower())
        print(f'There are {len(sorted_categories)} unique primary categories in the dataset:')
        for category in sorted_categories:
            print(category)
            
        return sorted_categories
        
    def select_samples(self) -> None:
        """Select samples with single labels from specific categories"""
        self.samples = []
        
        for s in self.dataset['train']:
            if len(s['categories'].split(' ')) != 1:
                continue
                
            cur_category = s['categories'].strip().split('.')[0]
            if cur_category not in CATEGORIES_TO_SELECT:
                continue
                
            self.samples.append(s)
            
            if len(self.samples) >= MAX_SAMPLES:
                break
                
        print(f"Number of samples: {len(self.samples)}")
        
        # Print first 3 samples
        for sample in self.samples[:3]:
            print(f"Category: {sample['categories']}")
            print("Abstract:", sample['abstract'])
            print("#" * 20 + "\n")
            
    def preprocess_samples(self) -> None:
        """Preprocess the selected samples"""
        self.preprocessed_samples = []
        
        for s in self.samples:
            abstract = s['abstract']
            
            # Remove \n characters in the middle and leading/trailing spaces
            abstract = abstract.strip().replace("\n", " ")
            
            # Remove special characters
            abstract = re.sub(r'[^\w\s]', '', abstract)
            
            # Remove digits
            abstract = re.sub(r'\d+', '', abstract)
            
            # Remove extra spaces
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            
            # Convert to lower case
            abstract = abstract.lower()
            
            # For the label, we only keep the first part
            parts = s['categories'].split(' ')
            category = parts[0].split('.')[0]
            
            self.preprocessed_samples.append({
                "text": abstract,
                "label": category
            })
            
        # Print first 3 preprocessed samples
        for sample in self.preprocessed_samples[:3]:
            print(f"Label: {sample['label']}")
            print("Text:", sample['text'])
            print("#" * 20 + "\n")
            
    def create_label_mappings(self) -> None:
        """Create label to ID and ID to label mappings"""
        labels = set([s['label'] for s in self.preprocessed_samples])
        sorted_labels = sorted(labels)
        
        for label in sorted_labels:
            print(label)
            
        self.label_to_id = {label: i for i, label in enumerate(sorted_labels)}
        self.id_to_label = {i: label for i, label in enumerate(sorted_labels)}
        
        # Print label to ID mapping
        print("Label to ID mapping:")
        for label, id_ in self.label_to_id.items():
            print(f"{label} --> {id_}")
            
    def prepare_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and testing data"""
        X_full = [sample['text'] for sample in self.preprocessed_samples]
        y_full = [self.label_to_id[sample['label']] for sample in self.preprocessed_samples]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y_full
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
        
    def get_sorted_labels(self) -> List[str]:
        """Get sorted list of labels"""
        return sorted(list(set([s['label'] for s in self.preprocessed_samples])))
