#!/usr/bin/env python3
"""
Test script to verify sample data loading functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.cache_manager import get_cache_manager, is_dataset_cached, load_cached_dataset


def test_sample_data_loading():
    """Test sample data loading functionality."""
    
    print("🔍 Testing Sample Data Loading...")
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
            # Load dataset (similar to main.py)
            dataset = load_cached_dataset(dataset_name, split="train")
            
            # Handle dataset with splits
            if hasattr(dataset, 'keys') and 'train' in dataset:
                # Dataset has splits, access the train split
                dataset = dataset['train']
                print(f"✅ Accessed train split, length: {len(dataset)}")
            
            if dataset is not None:
                print(f"🎉 Successfully loaded dataset!")
                print(f"   - Dataset type: {type(dataset)}")
                print(f"   - Dataset length: {len(dataset)}")
                
                # Take a small sample for demo (similar to main.py)
                sample_size = min(100, len(dataset))
                print(f"   - Sample size: {sample_size}")
                
                sample_data = dataset.select(range(sample_size))
                print(f"   - Sample data length: {len(sample_data)}")
                
                # Test the first item
                if len(sample_data) > 0:
                    first_item = sample_data[0]
                    print(f"\n📖 First paper sample:")
                    print(f"   - Title: {first_item.get('title', 'N/A')[:100]}...")
                    print(f"   - Abstract length: {len(first_item.get('abstract', ''))}")
                    print(f"   - Categories: {first_item.get('categories', 'N/A')}")
                    
                    # Test authors formatting
                    authors_parsed = first_item.get('authors_parsed', [])
                    print(f"   - Authors parsed: {authors_parsed}")
                    
                    # Format authors (similar to _format_authors function)
                    if authors_parsed:
                        formatted_authors = []
                        for author in authors_parsed:
                            if isinstance(author, list) and len(author) >= 2:
                                formatted_authors.append(f"{author[0]}, {author[1]}")
                            elif isinstance(author, list) and len(author) == 1:
                                formatted_authors.append(author[0])
                            else:
                                formatted_authors.append(str(author))
                        
                        authors_str = '; '.join(formatted_authors)
                        print(f"   - Formatted authors: {authors_str}")
                    
                    print(f"✅ Sample data loading successful!")
                    
            else:
                print("❌ Failed to load dataset from cache")
                
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ Dataset not found in cache")
    
    print("\n📋 Cache Information:")
    print("-" * 30)
    cache_info = cache_manager.get_cache_info()
    print(f"Cache directory: {cache_info['cache_directory']}")
    print(f"Total datasets: {cache_info['total_datasets']}")
    print(f"Cache size: {cache_info['cache_size_mb']:.2f} MB")
    print(f"Cached datasets: {cache_info['cached_datasets']}")


if __name__ == "__main__":
    test_sample_data_loading()
