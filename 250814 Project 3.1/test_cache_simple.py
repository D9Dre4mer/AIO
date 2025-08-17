#!/usr/bin/env python3
"""
Simple test script to verify cache reading mechanism
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.cache_manager import get_cache_manager, is_dataset_cached, load_cached_dataset


def test_cache_simple():
    """Simple test of cache functionality."""
    
    print("🔍 Testing Cache Functionality...")
    print("=" * 50)
    
    # Get cache manager
    cache_manager = get_cache_manager()
    
    # Test dataset name
    dataset_name = "UniverseTBD/arxiv-abstracts-large"
    
    print(f"📊 Testing dataset: {dataset_name}")
    
    # Check if dataset is cached
    is_cached = is_dataset_cached(dataset_name)
    print(f"✅ Is dataset cached: {is_cached}")
    
    if is_cached:
        print("📁 Dataset found in cache, attempting to load...")
        
        try:
            # Try to load the dataset
            dataset = load_cached_dataset(dataset_name)
            
            if dataset is not None:
                print(f"🎉 Successfully loaded dataset!")
                print(f"   - Dataset type: {type(dataset)}")
                
                # Handle dataset with splits
                if hasattr(dataset, 'keys') and 'train' in dataset:
                    # Dataset has splits, access the train split
                    train_dataset = dataset['train']
                    print(f"   - Train split length: {len(train_dataset)}")
                    
                    if len(train_dataset) > 0:
                        sample = train_dataset[0]
                        print(f"   - Sample keys: {list(sample.keys())}")
                        
                        # Check authors_parsed structure
                        if 'authors_parsed' in sample:
                            authors = sample['authors_parsed']
                            print(f"   - Authors type: {type(authors)}")
                            print(f"   - Authors value: {authors}")
                            
                            if isinstance(authors, list) and len(authors) > 0:
                                print(f"   - First author type: {type(authors[0])}")
                                print(f"   - First author value: {authors[0]}")
                        
                        # Check title
                        if 'title' in sample:
                            title = sample['title']
                            print(f"   - Title: {title[:100]}...")
                            
                        # Check abstract
                        if 'abstract' in sample:
                            abstract = sample['abstract']
                            print(f"   - Abstract length: {len(abstract)}")
                            print(f"   - Abstract preview: {abstract[:150]}...")
                            
                        # Check categories
                        if 'categories' in sample:
                            categories = sample['categories']
                            print(f"   - Categories: {categories}")
                else:
                    # Single dataset without splits
                    print(f"   - Dataset length: {len(dataset)}")
                    
                    if len(dataset) > 0:
                        sample = dataset[0]
                        print(f"   - Sample keys: {list(sample.keys())}")
                    
                    # Check authors_parsed structure
                    if 'authors_parsed' in sample:
                        authors = sample['authors_parsed']
                        print(f"   - Authors type: {type(authors)}")
                        print(f"   - Authors value: {authors}")
                        
                        if isinstance(authors, list) and len(authors) > 0:
                            print(f"   - First author type: {type(authors[0])}")
                            print(f"   - First author value: {authors[0]}")
                    
                    # Check title
                    if 'title' in sample:
                        title = sample['title']
                        print(f"   - Title: {title[:100]}...")
                    
            else:
                print("❌ Failed to load dataset from cache")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ Dataset not found in cache")
    
    # Show cache info
    print("\n📋 Cache Information:")
    print("-" * 30)
    cache_info = cache_manager.get_cache_info()
    print(f"Cache directory: {cache_info['cache_directory']}")
    print(f"Total datasets: {cache_info['total_datasets']}")
    print(f"Cache size: {cache_info['cache_size_mb']:.2f} MB")
    print(f"Cached datasets: {cache_info['cached_datasets']}")


if __name__ == "__main__":
    test_cache_simple()
