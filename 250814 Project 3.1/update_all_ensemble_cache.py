#!/usr/bin/env python3
"""
Script to update ALL ensemble CV accuracy in cache
"""

import pickle
import json
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import time

def update_all_ensemble_cache():
    """Update ALL ensemble CV accuracy in cache"""
    
    cache_dir = "cache/training_results"
    metadata_file = os.path.join(cache_dir, "cache_metadata.json")
    
    if not os.path.exists(metadata_file):
        print("❌ No cache metadata found")
        return
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("🔍 Updating ALL Ensemble CV Accuracy in Cache:")
    print("=" * 60)
    
    # Find 300000 samples cache
    target_cache = None
    for cache_key, cache_info in metadata.items():
        if "300000samples" in cache_key:
            target_cache = (cache_key, cache_info)
            break
    
    if not target_cache:
        print("❌ No 300000 samples cache found")
        return
    
    cache_key, cache_info = target_cache
    cache_file = cache_info['file_path']
    
    if not os.path.exists(cache_file):
        print(f"❌ Cache file not found: {cache_file}")
        return
    
    print(f"📁 Cache: {cache_key[:50]}...")
    print(f"📄 File: {cache_file}")
    
    try:
        # Load cache data
        print("📥 Loading cache data...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        # Find ALL ensemble results
        all_results = data['results']['all_results']
        ensemble_results = []
        
        for i, result in enumerate(all_results):
            if (isinstance(result, dict) and 
                'model_name' in result and 
                'Ensemble Learning' in result['model_name']):
                ensemble_results.append((i, result))
                print(f"✅ Found ensemble result at index {i}")
        
        print(f"📊 Found {len(ensemble_results)} ensemble results")
        
        if not ensemble_results:
            print("❌ No ensemble results found")
            return
        
        # Get training data
        if 'embedding_info' not in data:
            print("❌ No embedding_info found in cache")
            return
        
        embedding_info = data['embedding_info']
        print(f"🔤 Available embeddings: {list(embedding_info.keys())}")
        
        # Get labels
        if 'labels' not in data:
            print("❌ No labels found in cache")
            return
        
        labels_data = data['labels']
        if isinstance(labels_data, dict):
            # This is class names, get actual labels from dataframe
            if 'step1_data' in data and 'dataframe' in data['step1_data']:
                df = data['step1_data']['dataframe']
                if 'label' in df.columns:
                    y = df['label'].values
                    print(f"✅ Found labels in dataframe")
                else:
                    print(f"❌ No 'label' column in dataframe")
                    return
            else:
                print(f"❌ No dataframe found to extract labels")
                return
        else:
            y = labels_data
        
        y = np.array(y)
        print(f"🏷️  y shape: {y.shape}")
        print(f"🏷️  y unique: {np.unique(y)}")
        
        # Update each ensemble result
        updated_count = 0
        for idx, ensemble_result in ensemble_results:
            print(f"\n🔧 Updating ensemble result {idx + 1}/{len(ensemble_results)}")
            
            # Check if already has CV accuracy
            current_cv = ensemble_result.get('cv_mean_accuracy', 0.0)
            if current_cv > 0.0:
                print(f"   ✅ Already has CV accuracy: {current_cv:.4f}")
                continue
            
            # Get embedding name
            embedding_name = ensemble_result.get('embedding_name', 'Unknown')
            print(f"   🔤 Embedding: {embedding_name}")
            
            # Get embedding data
            if embedding_name in embedding_info:
                embedding_data = embedding_info[embedding_name]
                if isinstance(embedding_data, dict) and 'X_train' in embedding_data:
                    X = embedding_data['X_train']
                else:
                    X = embedding_data
            else:
                print(f"   ❌ Embedding '{embedding_name}' not found")
                continue
            
            print(f"   📐 X shape: {X.shape}")
            
            # Ensure X and y have same size
            if X.shape[0] != len(y):
                min_size = min(X.shape[0], len(y))
                X = X[:min_size]
                y_subset = y[:min_size]
                print(f"   ⚠️  Using {min_size} samples")
            else:
                y_subset = y
            
            # Get ensemble info
            ensemble_info = ensemble_result.get('ensemble_info', {})
            if not ensemble_info:
                print(f"   ❌ No ensemble_info found")
                continue
            
            # Get base models
            base_models = ensemble_info.get('base_models', [])
            if not base_models:
                print(f"   ❌ No base_models found")
                continue
            
            print(f"   📋 Base models: {base_models}")
            
            # Map display names to internal names
            name_mapping = {
                'K-Nearest Neighbors': 'knn',
                'Naive Bayes': 'naive_bayes', 
                'Decision Tree': 'decision_tree'
            }
            
            # Create VotingClassifier
            estimators = []
            for model_name in base_models:
                internal_name = name_mapping.get(model_name, model_name)
                
                # Find trained model
                trained_model = None
                for result in all_results:
                    if (isinstance(result, dict) and 
                        result.get('model_name') == internal_name and
                        'trained_model' in result):
                        trained_model = result['trained_model']
                        break
                
                if trained_model and hasattr(trained_model, 'predict'):
                    if hasattr(trained_model, 'model') and trained_model.model is not None:
                        sklearn_model = trained_model.model
                        estimators.append((model_name, sklearn_model))
                    else:
                        estimators.append((model_name, trained_model))
            
            if len(estimators) < 2:
                print(f"   ❌ Not enough base models: {len(estimators)}")
                continue
            
            # Create ensemble model
            ensemble_model = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=1  # Use single job for stability
            )
            
            # Perform cross-validation
            print(f"   🔄 Performing cross-validation...")
            start_time = time.time()
            
            try:
                cv_scores = cross_val_score(
                    ensemble_model,
                    X,
                    y_subset,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=1
                )
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                cv_time = time.time() - start_time
                
                print(f"   ✅ CV completed in {cv_time:.2f}s")
                print(f"   🎯 CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
                
                # Update ensemble result
                ensemble_result['cv_mean_accuracy'] = cv_mean
                ensemble_result['cv_std_accuracy'] = cv_std
                ensemble_result['cv_stability_score'] = 1.0 - cv_std
                
                updated_count += 1
                print(f"   ✅ Updated ensemble result")
                
            except Exception as e:
                print(f"   ❌ Error during CV: {e}")
                continue
        
        if updated_count > 0:
            # Save updated cache
            print(f"\n💾 Saving updated cache...")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✅ Cache updated successfully!")
            print(f"   Updated {updated_count}/{len(ensemble_results)} ensemble results")
        else:
            print(f"❌ No ensemble results were updated")
            
    except Exception as e:
        print(f"❌ Error updating cache: {e}")

if __name__ == "__main__":
    update_all_ensemble_cache()
