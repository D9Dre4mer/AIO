#!/usr/bin/env python3
"""
Embedding Cache Management Script
Allows users to view, clear, and manage embedding cache files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_evaluation import ComprehensiveEvaluator

def main():
    """Main function for embedding cache management"""
    print("🔧 Embedding Cache Management")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        cv_folds=5,
        validation_size=0.0,
        test_size=0.2,
        random_state=42
    )
    
    while True:
        print("\n📋 Available Operations:")
        print("1. 📊 Show cache status")
        print("2. 🗑️  Clear all cache")
        print("3. 🗑️  Clear specific cache")
        print("4. 📁 Show cache directory")
        print("5. ❌ Exit")
        
        try:
            choice = input("\nSelect operation (1-5): ").strip()
            
            if choice == "1":
                print("\n" + "="*50)
                evaluator.show_embedding_cache_status()
                
            elif choice == "2":
                confirm = input("\n⚠️  Are you sure you want to clear ALL embedding cache? (y/N): ").strip().lower()
                if confirm == 'y':
                    evaluator.clear_embedding_cache()
                    print("✅ All embedding cache cleared!")
                else:
                    print("❌ Operation cancelled")
                    
            elif choice == "3":
                print("\n📊 Current cache files:")
                evaluator.show_embedding_cache_status()
                
                cache_key = input("\nEnter cache key to clear (or press Enter to cancel): ").strip()
                if cache_key:
                    evaluator.clear_embedding_cache(cache_key)
                else:
                    print("❌ Operation cancelled")
                    
            elif choice == "4":
                from config import CACHE_DIR
                embeddings_cache_dir = os.path.join(CACHE_DIR, "embeddings")
                print(f"\n📁 Cache directory: {embeddings_cache_dir}")
                
                if os.path.exists(embeddings_cache_dir):
                    files = os.listdir(embeddings_cache_dir)
                    print(f"📄 Files in directory: {len(files)}")
                    for file in files:
                        print(f"   - {file}")
                else:
                    print("📭 Directory does not exist")
                    
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
