#!/usr/bin/env python3
"""
Script to check embeddings structure in cache
"""

import pickle
import json
import os

def check_embeddings():
    """Check embeddings structure in cache"""
    
    cache_dir = "cache/training_results"
    metadata_file = os.path.join(cache_dir, "cache_metadata.json")
    
    if not os.path.exists(metadata_file):
        print("❌ No cache metadata found")
        return
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("🔍 Checking Embeddings Structure:")
    print("=" * 50)
    
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
    
    try:
        # Load cache data
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📊 Cache Structure:")
        print(f"   Keys: {list(data.keys())}")
        
        # Check embedding_info
        if 'embedding_info' in data:
            embedding_info = data['embedding_info']
            print(f"\n🔤 Embedding Info:")
            print(f"   Keys: {list(embedding_info.keys())}")
            
            for key, value in embedding_info.items():
                print(f"   {key}: {type(value)}")
                if isinstance(value, dict):
                    print(f"      Keys: {list(value.keys())}")
                elif hasattr(value, 'shape'):
                    print(f"      Shape: {value.shape}")
        
        # Check comprehensive_results for embeddings
        if 'comprehensive_results' in data:
            comp_results = data['comprehensive_results']
            print(f"\n📈 Comprehensive Results:")
            print(f"   Type: {type(comp_results)}")
            
            if isinstance(comp_results, list):
                for i, result in enumerate(comp_results):
                    if isinstance(result, dict) and 'embeddings' in result:
                        print(f"   [{i}] Found embeddings:")
                        embeddings = result['embeddings']
                        print(f"      Keys: {list(embeddings.keys()) if isinstance(embeddings, dict) else 'Not a dict'}")
        
        # Check all_results for ensemble results
        if 'results' in data and 'all_results' in data['results']:
            all_results = data['results']['all_results']
            print(f"\n📊 All Results:")
            print(f"   Total: {len(all_results)}")
            
            for i, result in enumerate(all_results):
                if (isinstance(result, dict) and 
                    'model_name' in result and 
                    'Ensemble Learning' in result['model_name']):
                    embedding_name = result.get('embedding_name', 'Unknown')
                    cv_accuracy = result.get('cv_mean_accuracy', 0.0)
                    print(f"   [{i}] Ensemble - {embedding_name}: CV={cv_accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Error loading cache: {e}")

if __name__ == "__main__":
    check_embeddings()
