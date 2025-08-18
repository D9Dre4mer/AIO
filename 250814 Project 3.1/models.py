"""
Models module for Topic Modeling Project
Handles different machine learning models and their training/testing
"""

from collections import Counter
from typing import Tuple, Dict, Any, Union

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from config import (
    KMEANS_N_CLUSTERS, KNN_N_NEIGHBORS, 
    KMEANS_SVD_THRESHOLD, KMEANS_SVD_COMPONENTS
)


class ModelTrainer:
    """Class for training and testing different machine learning models"""
    
    def __init__(self):
        pass
        
    def train_and_test_kmeans(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray, 
        n_clusters: int = KMEANS_N_CLUSTERS
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test K-Means clustering model"""
        # Convert sparse to dense for KMeans if needed (memory efficient)
        if sparse.issparse(X_train):
            print(f"💾 K-Means preprocessing: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")
            
            if X_train.shape[1] > KMEANS_SVD_THRESHOLD:
                print(f"📊 High-dimensional data detected ({X_train.shape[1]:,} > {KMEANS_SVD_THRESHOLD:,} features)")
                print(f"🔧 Applying Truncated SVD dimensionality reduction...")
                
                from sklearn.decomposition import TruncatedSVD
                n_components = min(KMEANS_SVD_COMPONENTS, X_train.shape[1] - 1, X_train.shape[0] - 1)
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                
                from tqdm import tqdm
                print(f"⚡ Reducing {X_train.shape[1]:,} → {n_components:,} dimensions...")

                # Show progress bar for SVD fit_transform on X_train
                print("🔄 Performing SVD fit_transform on training data...")
                X_train = svd.fit_transform(
                    tqdm(X_train, desc="SVD fit_transform (train)", unit="samples", total=X_train.shape[0])
                )

                # Show progress bar for SVD transform on X_test
                print("🔄 Performing SVD transform on test data...")
                X_test = svd.transform(
                    tqdm(X_test, desc="SVD transform (test)", unit="samples", total=X_test.shape[0])
                )
                
                explained_variance = svd.explained_variance_ratio_.sum()
                print(f"✅ SVD completed: {X_train.shape[1]:,} dimensions | "
                      f"Variance preserved: {explained_variance:.1%}")
            else:
                print(f"✅ Converting to dense matrix ({X_train.shape[1]:,} features ≤ {KMEANS_SVD_THRESHOLD:,} threshold)")
                X_train = X_train.toarray()
                X_test = X_test.toarray()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_ids = kmeans.fit_predict(X_train)

        # Assign label to clusters
        cluster_to_label = {}
        for cluster_id in set(cluster_ids):
            # Get all labels in this cluster
            labels_in_cluster = [
                y_train[i] for i in range(len(y_train)) 
                if cluster_ids[i] == cluster_id
            ]
            most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
            cluster_to_label[cluster_id] = most_common_label

        # Predict labels for test set
        test_cluster_ids = kmeans.predict(X_test)
        y_pred = [cluster_to_label[cluster_id] for cluster_id in test_cluster_ids]
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get unique labels for classification report
        unique_labels = sorted(list(set(y_train) | set(y_test)))
        report = classification_report(
            y_test, y_pred, 
            labels=unique_labels,
            output_dict=True
        )

        return y_pred, accuracy, report
        
    def train_and_test_knn(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray, 
        n_neighbors: int = KNN_N_NEIGHBORS
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test K-Nearest Neighbors classifier"""
        # KNN works with sparse matrices
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='brute' if sparse.issparse(X_train) else 'auto')
        knn.fit(X_train, y_train)

        # Predict on the test set
        y_pred = knn.predict(X_test)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        unique_labels = sorted(list(set(y_train) | set(y_test)))
        report = classification_report(
            y_test, y_pred, 
            labels=unique_labels,
            output_dict=True
        )

        return y_pred, accuracy, report
        
    def train_and_test_decision_tree(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Decision Tree classifier"""
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)

        # Predict on the test set
        y_pred = dt.predict(X_test)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        unique_labels = sorted(list(set(y_train) | set(y_test)))
        report = classification_report(
            y_test, y_pred, 
            labels=unique_labels,
            output_dict=True
        )

        return y_pred, accuracy, report
        
    def train_and_test_naive_bayes(
        self, 
        X_train: Union[np.ndarray, sparse.csr_matrix], 
        y_train: np.ndarray, 
        X_test: Union[np.ndarray, sparse.csr_matrix], 
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Train and test Naive Bayes classifier"""
        # Use MultinomialNB for sparse matrices (better for text data)
        if sparse.issparse(X_train):
            print("📊 Using MultinomialNB for sparse text features")
            nb = MultinomialNB()
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
        else:
            print("📊 Using GaussianNB for dense features")
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        unique_labels = sorted(list(set(y_train) | set(y_test)))
        report = classification_report(
            y_test, y_pred, 
            labels=unique_labels,
            output_dict=True
        )

        return y_pred, accuracy, report


def get_model_descriptions():
    """Get descriptions of different models"""
    descriptions = {
        'kmeans': {
            'name': 'K-Means Clustering',
            'description': 'K-Means is an unsupervised learning algorithm that groups similar data points into clusters. It works by iteratively assigning data points to the nearest cluster center and updating the cluster centers based on the mean of assigned points.',
            'use_case': 'Often used for exploratory data analysis and as a baseline for clustering tasks.'
        },
        'knn': {
            'name': 'K-Nearest Neighbors (KNN)',
            'description': 'KNN is a simple and effective classification algorithm that classifies a data point based on the majority class of its k nearest neighbors in the feature space.',
            'use_case': 'Often used for text classification tasks and when interpretability is important.'
        },
        'decision_tree': {
            'name': 'Decision Tree',
            'description': 'Decision Tree is a tree-like model that makes decisions based on asking a series of questions. Each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label.',
            'use_case': 'Good for interpretable models and handling both numerical and categorical data.'
        },
        'naive_bayes': {
            'name': 'Naive Bayes',
            'description': 'Naive Bayes is a probabilistic classifier based on Bayes theorem with an assumption of conditional independence between every pair of features given the value of the class variable.',
            'use_case': 'Excellent for text classification tasks and when features are conditionally independent.'
        }
    }
    
    return descriptions
