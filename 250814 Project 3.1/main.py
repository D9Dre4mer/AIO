"""
Main execution script for Topic Modeling Project
Orchestrates the entire pipeline from data loading to model evaluation
"""

import warnings
import numpy as np

from data_loader import DataLoader
from text_encoders import TextVectorizer
from models.new_model_trainer import NewModelTrainer
from models.utils.validation_manager import validation_manager
from visualization import (
    plot_confusion_matrix, 
    create_output_directories,
    plot_model_comparison,
    print_model_results
)

warnings.filterwarnings("ignore")


def main():
    """Main execution function"""
    print("🚀 Starting Topic Modeling Project")
    print("=" * 50)
    
    # Initialize components
    data_loader = DataLoader()
    text_vectorizer = TextVectorizer()
    model_trainer = NewModelTrainer(
        cv_folds=5, 
        validation_size=0.2,
        test_size=0.2
    )
    
    # Step 1: Load and explore dataset
    print("\n📊 Step 1: Loading Dataset")
    print("-" * 30)
    data_loader.load_dataset()
    data_loader.print_sample_examples(3)
    
    # Step 2: Explore categories
    print("\n🏷️  Step 2: Exploring Categories")
    print("-" * 30)
    all_categories = data_loader.get_all_categories(1000)
    print(f"Sample categories: {all_categories}")
    
    # Get primary categories for reference (not used in current pipeline)
    _ = data_loader.get_primary_categories()
    
    # Step 3: Select and preprocess samples
    print("\n🔍 Step 3: Selecting and Preprocessing Samples")
    print("-" * 30)
    
    # Get user input for number of samples
    try:
        user_samples = input("📊 Enter number of samples to process (default: 100000): ").strip()
        if user_samples:
            num_samples = int(user_samples)
            print(f"✅ Using {num_samples:,} samples as requested")
        else:
            num_samples = 100000
            print(f"✅ Using default: {num_samples:,} samples")
    except ValueError:
        num_samples = 100000
        print(f"⚠️ Invalid input, using default: {num_samples:,} samples")
    
    data_loader.select_samples(max_samples=num_samples)
    data_loader.preprocess_samples()
    
    # Step 4: Create label mappings
    print("\n🔄 Step 4: Creating Label Mappings")
    print("-" * 30)
    data_loader.create_label_mappings()
    
    # Step 5: Prepare training and testing data
    print("\n✂️  Step 5: Preparing Training and Testing Data")
    print("-" * 30)
    X_train, X_test, y_train, y_test = data_loader.prepare_train_test_data()
    sorted_labels = data_loader.get_sorted_labels()
    
    # Create 3-way split using validation manager
    print("\n🔄 Step 5.1: Creating Train/Validation/Test Split")
    print("-" * 30)
    X_train_full, X_val, X_test, y_train_full, y_val, y_test = \
        validation_manager.split_data(
            np.concatenate([X_train, X_test]), 
            np.concatenate([y_train, y_test])
        )
    
    print("✅ Data split created:")
    print(f"   • Training: {len(X_train_full)} samples")
    print(f"   • Validation: {len(X_val)} samples")
    print(f"   • Test: {len(X_test)} samples")
    
    # Store original test size for visualization
    original_test_size = len(X_test)
    
    # Step 6: Text vectorization
    print("\n🔤 Step 6: Text Vectorization")
    print("-" * 30)
    
    # Bag of Words
    print("Processing Bag of Words...")
    X_train_bow = text_vectorizer.fit_transform_bow(X_train_full)
    X_val_bow = text_vectorizer.transform_bow(X_val)
    X_test_bow = text_vectorizer.transform_bow(X_test)
    
    # TF-IDF
    print("Processing TF-IDF...")
    X_train_tfidf = text_vectorizer.fit_transform_tfidf(X_train_full)
    X_val_tfidf = text_vectorizer.transform_tfidf(X_val)
    X_test_tfidf = text_vectorizer.transform_tfidf(X_test)
    
    # Word Embeddings
    print("Processing Word Embeddings...")
    print("📊 Using sentence-transformers for semantic embeddings")
    
    print("🔄 Processing training set embeddings...")
    X_train_embeddings = text_vectorizer.transform_embeddings(X_train_full)
    
    print("🔄 Processing validation set embeddings...")
    X_val_embeddings = text_vectorizer.transform_embeddings(X_val)
    
    print("🔄 Processing test set embeddings...")
    X_test_embeddings = text_vectorizer.transform_embeddings(X_test)
    
    # Print shapes
    print(f"Shape of X_train_bow: {X_train_bow.shape}")
    print(f"Shape of X_test_bow: {X_test_bow.shape}")
    print(f"Shape of X_train_tfidf: {X_train_tfidf.shape}")
    print(f"Shape of X_test_tfidf: {X_test_tfidf.shape}")
    print(f"Shape of X_train_embeddings: {X_train_embeddings.shape}")
    print(f"Shape of X_test_embeddings: {X_test_embeddings.shape}")
    
    # Step 7: Train and test models
    print("\n🤖 Step 7: Training and Testing Models")
    print("-" * 30)
    
    # Create output directories
    create_output_directories()
    
    # Store results
    results = {}
    
    # K-Means - Using train_validate_test_model for consistent data splits
    print("\n--- K-Means Clustering ---")
    km_bow_labels, _, _, _, km_bow_accuracy, km_bow_report = model_trainer.train_validate_test_model(
        'kmeans', X_train_bow, y_train_full, X_val_bow, y_val, X_test_bow, y_test
    )
    km_tfidf_labels, _, _, _, km_tfidf_accuracy, km_tfidf_report = model_trainer.train_validate_test_model(
        'kmeans', X_train_tfidf, y_train_full, X_val_tfidf, y_val, X_test_tfidf, y_test
    )
    km_embeddings_labels, _, _, _, km_embeddings_accuracy, km_embeddings_report = model_trainer.train_validate_test_model(
        'kmeans', X_train_embeddings, y_train_full, X_val_embeddings, y_val, X_test_embeddings, y_test
    )
    
    results.update({
        'kmeans_bow_accuracy': km_bow_accuracy,
        'kmeans_tfidf_accuracy': km_tfidf_accuracy,
        'kmeans_embeddings_accuracy': km_embeddings_accuracy
    })
    
    print("Accuracies for K-Means:")
    print(f"Bag of Words: {km_bow_accuracy:.4f}")
    print(f"TF-IDF: {km_tfidf_accuracy:.4f}")
    print(f"Embeddings: {km_embeddings_accuracy:.4f}")
    
    # KNN
    print("\n--- K-Nearest Neighbors ---")
    knn_bow_labels, _, _, _, knn_bow_accuracy, knn_bow_report = model_trainer.train_validate_test_model(
        'knn', X_train_bow, y_train_full, X_val_bow, y_val, X_test_bow, y_test
    )
    knn_tfidf_labels, _, _, _, knn_tfidf_accuracy, knn_tfidf_report = model_trainer.train_validate_test_model(
        'knn', X_train_tfidf, y_train_full, X_val_tfidf, y_val, X_test_tfidf, y_test
    )
    knn_embeddings_labels, _, _, _, knn_embeddings_accuracy, knn_embeddings_report = model_trainer.train_validate_test_model(
        'knn', X_train_embeddings, y_train_full, X_val_embeddings, y_val, X_test_embeddings, y_test
    )
    
    results.update({
        'knn_bow_accuracy': knn_bow_accuracy,
        'knn_tfidf_accuracy': knn_tfidf_accuracy,
        'knn_embeddings_accuracy': knn_embeddings_accuracy
    })
    
    print("Accuracies for KNN:")
    print(f"Bag of Words: {knn_bow_accuracy:.4f}")
    print(f"TF-IDF: {knn_tfidf_accuracy:.4f}")
    print(f"Embeddings: {knn_embeddings_accuracy:.4f}")
    
    # Decision Tree
    print("\n--- Decision Tree ---")
    dt_bow_labels, _, _, _, dt_bow_accuracy, dt_bow_report = model_trainer.train_validate_test_model(
        'decision_tree', X_train_bow, y_train_full, X_val_bow, y_val, X_test_bow, y_test
    )
    dt_tfidf_labels, _, _, _, dt_tfidf_accuracy, dt_tfidf_report = model_trainer.train_validate_test_model(
        'decision_tree', X_train_tfidf, y_train_full, X_val_tfidf, y_val, X_test_tfidf, y_test
    )
    dt_embeddings_labels, _, _, _, dt_embeddings_accuracy, dt_embeddings_report = model_trainer.train_validate_test_model(
        'decision_tree', X_train_embeddings, y_train_full, X_val_embeddings, y_val, X_test_embeddings, y_test
    )
    
    results.update({
        'decision_tree_bow_accuracy': dt_bow_accuracy,
        'decision_tree_tfidf_accuracy': dt_tfidf_accuracy,
        'decision_tree_embeddings_accuracy': dt_embeddings_accuracy
    })
    
    print("Accuracies for Decision Tree:")
    print(f"Bag of Words: {dt_bow_accuracy:.4f}")
    print(f"TF-IDF: {dt_tfidf_accuracy:.4f}")
    print(f"Embeddings: {dt_embeddings_accuracy:.4f}")
    
    # Naive Bayes
    print("\n--- Naive Bayes ---")
    nb_bow_labels, _, _, _, nb_bow_accuracy, nb_bow_report = model_trainer.train_validate_test_model(
        'naive_bayes', X_train_bow, y_train_full, X_val_bow, y_val, X_test_bow, y_test
    )
    nb_tfidf_labels, _, _, _, nb_tfidf_accuracy, nb_tfidf_report = model_trainer.train_validate_test_model(
        'naive_bayes', X_train_tfidf, y_train_full, X_val_tfidf, y_val, X_test_tfidf, y_test
    )
    nb_embeddings_labels, _, _, _, nb_embeddings_accuracy, nb_embeddings_report = model_trainer.train_validate_test_model(
        'naive_bayes', X_train_embeddings, y_train_full, X_val_embeddings, y_val, X_test_embeddings, y_test
    )
    
    results.update({
        'naive_bayes_bow_accuracy': nb_bow_accuracy,
        'naive_bayes_tfidf_accuracy': nb_tfidf_accuracy,
        'naive_bayes_embeddings_accuracy': nb_embeddings_accuracy
    })
    
    print("Accuracies for Naive Bayes:")
    print(f"Bag of Words: {nb_bow_accuracy:.4f}")
    print(f"TF-IDF: {nb_tfidf_accuracy:.4f}")
    print(f"Embeddings: {nb_embeddings_accuracy:.4f}")
    
    # Step 8: Generate visualizations
    print("\n📊 Step 8: Generating Visualizations")
    print("-" * 30)
    
    # Plot confusion matrices - use y_test from validation manager (200 samples)
    plot_confusion_matrix(
        y_test, km_bow_labels, sorted_labels, 
        "KMeans Confusion Matrix (Bag of Words)", 
        "pdf/Figures/kmeans_bow_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, km_tfidf_labels, sorted_labels, 
        "KMeans Confusion Matrix (TF-IDF)", 
        "pdf/Figures/kmeans_tfidf_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, km_embeddings_labels, sorted_labels, 
        "KMeans Confusion Matrix (Embeddings)", 
        "pdf/Figures/kmeans_embeddings_confusion_matrix.pdf"
    )
    
    plot_confusion_matrix(
        y_test, knn_bow_labels, sorted_labels, 
        "KNN Confusion Matrix (Bag of Words)", 
        "pdf/Figures/knn_bow_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, knn_tfidf_labels, sorted_labels, 
        "KNN Confusion Matrix (TF-IDF)", 
        "pdf/Figures/knn_tfidf_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, knn_embeddings_labels, sorted_labels, 
        "KNN Confusion Matrix (Embeddings)", 
        "pdf/Figures/knn_embeddings_confusion_matrix.pdf"
    )
    
    plot_confusion_matrix(
        y_test, dt_bow_labels, sorted_labels, 
        "Decision Tree Confusion Matrix (Bag of Words)", 
        "pdf/Figures/dt_bow_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, dt_tfidf_labels, sorted_labels, 
        "Decision Tree Confusion Matrix (TF-IDF)", 
        "pdf/Figures/dt_tfidf_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, dt_embeddings_labels, sorted_labels, 
        "Decision Tree Confusion Matrix (Embeddings)", 
        "pdf/Figures/dt_embeddings_confusion_matrix.pdf"
    )
    
    plot_confusion_matrix(
        y_test, nb_bow_labels, sorted_labels, 
        "Naive Bayes Confusion Matrix (Bag of Words)", 
        "pdf/Figures/nb_bow_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, nb_tfidf_labels, sorted_labels, 
        "Naive Bayes Confusion Matrix (TF-IDF)", 
        "pdf/Figures/nb_tfidf_confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        y_test, nb_embeddings_labels, sorted_labels, 
        "Naive Bayes Confusion Matrix (Embeddings)", 
        "pdf/Figures/nb_embeddings_confusion_matrix.pdf"
    )
    
    # Step 9: Results summary
    print("\n📋 Step 9: Results Summary")
    print("-" * 30)
    print_model_results(results)
    
    # Plot model comparison
    plot_model_comparison(
        results, 
        "pdf/Figures/model_comparison.pdf"
    )
    
    print("\n🎉 Topic Modeling Project completed successfully!")
    print("📁 Results and visualizations saved to 'pdf/Figures/' directory")


if __name__ == "__main__":
    main()
