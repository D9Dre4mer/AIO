"""
Data loader module for Topic Modeling Project
Handles dataset loading, preprocessing, and text cleaning
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import (
    CACHE_DIR, MAX_SAMPLES, 
    TEST_SIZE, RANDOM_STATE
)


class DataLoader:
    """Class for loading and preprocessing any dataset with dynamic category detection"""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.dataset = None
        self.samples = []
        self.preprocessed_samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        # Dynamic category management
        self.available_categories = []
        self.selected_categories = []
        self.category_stats = {}
        
    def load_dataset(self, skip_csv_prompt: bool = False) -> None:
        """Load any dataset and automatically detect categories"""
        dataset_cache_path = (Path(self.cache_dir) / 
                             "UniverseTBD___arxiv-abstracts-large")
        csv_backup_path = Path(self.cache_dir) / "arxiv_dataset_backup.csv"
        
        if dataset_cache_path.exists():
                    print(f"✅ Dataset found in cache: {dataset_cache_path}")
        else:
            print(f"📥 Dataset not found in cache: {dataset_cache_path}")
            print("🌐 Will download dataset to cache...")
        
        # Load the dataset from the specified cache directory
        self.dataset = load_dataset("UniverseTBD/arxiv-abstracts-large", 
                                   cache_dir=self.cache_dir)
        
        print(f"🎉 Dataset loaded successfully!")
        
        # Skip CSV backup prompt if requested (for Streamlit usage)
        if skip_csv_prompt:
            print("🚀 Streamlit mode: Skipping CSV backup prompt...")
            choice = "3"  # Skip CSV backup for Streamlit
        else:
            # CSV Backup Options
            print("\n💾 CSV Backup Options:")
            print(f"📊 Dataset size: {len(self.dataset['train']):,} samples")
            print("1. 🚀 Quick Sample (1,000 samples) - 30 seconds")
            print("2. 📋 Full Export (2.3M+ samples) - ~45 minutes, ~1.7GB")
            print("3. ⏭️  Skip CSV backup (fastest)")
            
            # Check if we're running in interactive mode
            try:
                choice = input("Choose option (1/2/3) [default: 1]: ").strip()
            except (EOFError, NameError):
                # Running in non-interactive mode, use default
                print("🚀 Non-interactive mode detected, using default: Quick Sample (1,000 samples)")
                choice = "1"
        
        if choice == "2":
            print("\n📋 Creating Full CSV Backup (this will take ~45 minutes):")
            print("💡 You can stop anytime with Ctrl+C and use option 1 next time")
            self._create_csv_backup_chunked(csv_backup_path)
        elif choice == "3":
            print("⏭️  Skipping CSV backup")
        else:
            print("\n🚀 Creating Quick Sample Backup:")
            self._create_sample_csv_backup(csv_backup_path)
        
    def _create_sample_csv_backup(self, csv_path: Path) -> None:
        """Create a quick CSV backup with only sample data (much faster)"""
        try:
            import pandas as pd
            
            print(f"💾 Creating sample CSV backup at: {csv_path}")
            
            # Only take first 1000 samples for speed
            sample_size = min(1000, len(self.dataset['train']))
            print(f"📝 Exporting {sample_size:,} sample records (for speed)...")
            
            # Create list of sample data
            sample_data = []
            for i in range(sample_size):
                sample = self.dataset['train'][i]
                sample_info = {
                    "id": i,
                    "abstract": sample['abstract'][:200] + "..." if len(sample['abstract']) > 200 else sample['abstract'],
                    "categories": sample['categories'],
                    "title": sample.get('title', '')[:100] + "..." if sample.get('title') and len(sample.get('title', '')) > 100 else sample.get('title', ''),
                    "authors": sample.get('authors', ''),
                    "doi": sample.get('doi', ''),
                    "date": sample.get('date', ''),
                    "journal_ref": sample.get('journal_ref', ''),
                    "report_no": sample.get('report_no', ''),
                    "license": sample.get('license', ''),
                    "update_date": sample.get('update_date', '')
                }
                sample_data.append(sample_info)
            
            # Convert to DataFrame and save as CSV
            print("💾 Saving sample data to CSV file...")
            df = pd.DataFrame(sample_data)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"✅ Sample CSV backup created successfully!")
            print(f"📁 File size: {csv_path.stat().st_size / 1024:.2f} KB")
            print(f"📊 Sample rows: {len(df):,}")
            print(f"🔗 Path: {csv_path}")
            print("💡 This is a sample backup. Use _create_csv_backup() for full dataset.")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create sample CSV backup: {e}")
            print("Dataset will continue to work normally")
        
    def _create_csv_backup_chunked(self, csv_path: Path, chunk_size: int = 10000) -> None:
        """Create CSV backup with chunked processing to avoid memory issues"""
        try:
            import pandas as pd
            
            print(f"💾 Creating chunked CSV backup at: {csv_path}")
            
            # Get total samples for progress tracking
            total_samples = len(self.dataset['train'])
            print(f"📝 Processing {total_samples:,} samples in chunks of {chunk_size:,}")
            
            # Process in chunks to avoid memory issues
            first_chunk = True
            
            for i in range(0, total_samples, chunk_size):
                chunk_end = min(i + chunk_size, total_samples)
                chunk_data = []
                
                # Process current chunk
                for j in range(i, chunk_end):
                    sample = self.dataset['train'][j]
                    sample_info = {
                        "id": j,
                        "abstract": sample['abstract'],
                        "categories": sample['categories'],
                        "title": sample.get('title', ''),
                        "authors": sample.get('authors', ''),
                        "doi": sample.get('doi', ''),
                        "date": sample.get('date', ''),
                        "journal_ref": sample.get('journal_ref', ''),
                        "report_no": sample.get('report_no', ''),
                        "license": sample.get('license', ''),
                        "update_date": sample.get('update_date', '')
                    }
                    chunk_data.append(sample_info)
                
                # Write chunk to CSV
                df_chunk = pd.DataFrame(chunk_data)
                df_chunk.to_csv(csv_path, 
                               mode='w' if first_chunk else 'a',
                               header=first_chunk, 
                               index=False, 
                               encoding='utf-8')
                first_chunk = False
                
                # Progress update
                progress_percent = (chunk_end / total_samples) * 100
                progress_bar = self._create_progress_bar(progress_percent, 50)
                elapsed_time = (chunk_end / total_samples) * 45  # Estimate based on 45 min total
                remaining_time = 45 - elapsed_time
                
                progress_text = (f"\r🔄 Progress: {progress_bar} "
                               f"{progress_percent:.1f}% "
                               f"({chunk_end:,}/{total_samples:,}) "
                               f"ETA: {remaining_time:.1f}m")
                print(progress_text, end="", flush=True)
            
            print()  # New line after progress bar
            
            print(f"✅ Chunked CSV backup created successfully!")
            print(f"📁 File size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"📊 Total rows: {total_samples:,}")
            print(f"🔗 Path: {csv_path}")
            
            # Also create a statistics summary file
            self._create_statistics_summary(csv_path.parent)
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create chunked CSV backup: {e}")
            print("Dataset will continue to work normally")
    
    def _create_csv_backup(self, csv_path: Path) -> None:
        """Create a CSV backup file containing all dataset information"""
        try:
            import pandas as pd
            
            print(f"💾 Creating CSV backup at: {csv_path}")
            
            # Get total samples for progress tracking
            total_samples = len(self.dataset['train'])
            print(f"📝 Exporting ALL {total_samples:,} samples to CSV...")
            
            # Create list of all samples with progress tracking
            all_samples = []
            progress_step = max(1, total_samples // 100)  # Update every 1% or every sample if < 100
            
            for i, sample in enumerate(self.dataset['train']):
                sample_info = {
                    "id": i,
                    "abstract": sample['abstract'],
                    "categories": sample['categories'],
                    "title": sample.get('title', ''),
                    "authors": sample.get('authors', ''),
                    "doi": sample.get('doi', ''),
                    "date": sample.get('date', ''),
                    "journal_ref": sample.get('journal_ref', ''),
                    "report_no": sample.get('report_no', ''),
                    "license": sample.get('license', ''),
                    "update_date": sample.get('update_date', '')
                }
                all_samples.append(sample_info)
                
                # Show progress every 1% or every sample for small datasets
                if (i + 1) % progress_step == 0 or i == total_samples - 1:
                    progress_percent = ((i + 1) / total_samples) * 100
                    progress_bar = self._create_progress_bar(progress_percent, 50)
                    progress_text = (f"\r🔄 Progress: {progress_bar} "
                                   f"{progress_percent:.1f}% "
                                   f"({i + 1:,}/{total_samples:,})")
                    print(progress_text, end="", flush=True)
            
            print()  # New line after progress bar
            
            # Convert to DataFrame and save as CSV
            print("💾 Saving to CSV file...")
            df = pd.DataFrame(all_samples)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"✅ CSV backup created successfully!")
            print(f"📁 File size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"📊 Total rows: {len(df):,}")
            print(f"🔗 Path: {csv_path}")
            
            # Also create a statistics summary file
            self._create_statistics_summary(csv_path.parent)
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create CSV backup: {e}")
            print("Dataset will continue to work normally")
    
    def _create_progress_bar(self, percentage: float, width: int = 50) -> str:
        """Create a visual progress bar"""
        filled_width = int(width * percentage / 100)
        bar = "█" * filled_width + "░" * (width - filled_width)
        return bar
    
    def _create_statistics_summary(self, output_dir: Path) -> None:
        """Create a separate statistics summary file"""
        try:
            import pandas as pd
            
            stats_path = output_dir / "arxiv_dataset_statistics.csv"
            print(f"📊 Creating statistics summary at: {stats_path}")
            
            # Calculate statistics
            stats_data = {
                "metric": [
                    "total_samples",
                    "total_abstracts",
                    "unique_categories",
                    "avg_abstract_length",
                    "min_abstract_length",
                    "max_abstract_length",
                    "samples_with_title",
                    "samples_with_authors",
                    "samples_with_doi",
                    "samples_with_date"
                ],
                "value": [
                    len(self.dataset['train']),
                    len(self.dataset['train']),
                    len(set([s['categories'].split('.')[0] for s in self.dataset['train']])),
                    int(np.mean([len(s['abstract']) for s in self.dataset['train']])),
                    int(np.min([len(s['abstract']) for s in self.dataset['train']])),
                    int(np.max([len(s['abstract']) for s in self.dataset['train']])),
                    sum(1 for s in self.dataset['train'] if s.get('title')),
                    sum(1 for s in self.dataset['train'] if s.get('authors')),
                    sum(1 for s in self.dataset['train'] if s.get('doi')),
                    sum(1 for s in self.dataset['train'] if s.get('date'))
                ]
            }
            
            # Create categories distribution
            categories = [s['categories'] for s in self.dataset['train']]
            category_counts = Counter()
            
            for cat_list in categories:
                for cat in cat_list.split(' '):
                    primary_cat = cat.split('.')[0]
                    category_counts[primary_cat] += 1
            
            # Save main statistics
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_csv(stats_path, index=False, encoding='utf-8')
            
            # Save categories distribution
            cat_dist_path = output_dir / "arxiv_categories_distribution.csv"
            cat_dist_df = pd.DataFrame([
                {"category": cat, "count": count} 
                for cat, count in category_counts.most_common()
            ])
            cat_dist_df.to_csv(cat_dist_path, index=False, encoding='utf-8')
            
            print(f"✅ Statistics files created:")
            print(f"  📊 Main stats: {stats_path}")
            print(f"  🏷️ Categories: {cat_dist_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create statistics summary: {e}")
    
    def _get_categories_distribution(self) -> Dict[str, int]:
        """Get distribution of categories in the dataset"""
        try:
            categories = [s['categories'] for s in self.dataset['train']]
            category_counts = Counter()
            
            for cat_list in categories:
                for cat in cat_list.split(' '):
                    primary_cat = cat.split('.')[0]
                    category_counts[primary_cat] += 1
            
            # Return top 20 categories
            return dict(category_counts.most_common(20))
        except Exception:
            return {"error": "Could not calculate categories distribution"}
    
    def _calculate_avg_abstract_length(self) -> Dict[str, float]:
        """Calculate average abstract length statistics"""
        try:
            abstract_lengths = [len(s['abstract']) for s in self.dataset['train']]
            
            return {
                "mean": float(np.mean(abstract_lengths)),
                "median": float(np.median(abstract_lengths)),
                "min": int(np.min(abstract_lengths)),
                "max": int(np.max(abstract_lengths)),
                "std": float(np.std(abstract_lengths))
            }
        except Exception:
            return {"error": "Could not calculate abstract length statistics"}
    
    def _get_date_range(self) -> Dict[str, str]:
        """Get the date range of the dataset"""
        try:
            dates = [s.get('date', '') for s in self.dataset['train'] if s.get('date')]
            if dates:
                return {
                    "earliest": min(dates),
                    "latest": max(dates),
                    "total_with_dates": len(dates)
                }
            else:
                return {"error": "No date information available"}
        except Exception:
            return {"error": "Could not calculate date range"}
        
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
        
    def select_samples(self, max_samples: int = None, categories: List[str] = None) -> None:
        """
        Select samples with dynamic category filtering
        
        Args:
            max_samples: Maximum number of samples to select
            categories: List of categories to filter by (uses self.selected_categories if None)
        """
        self.samples = []
        
        # Use provided categories or fall back to selected categories
        if categories is None:
            if not self.selected_categories:
                print("❌ No categories selected. Call set_selected_categories() first.")
                return
            categories = self.selected_categories
        
        # Use provided max_samples or fall back to config default
        if max_samples is None:
            if MAX_SAMPLES is None:
                # If no max_samples specified, use entire dataset
                max_samples = len(self.dataset['train'])
                print(f"📊 No sample limit specified, using entire dataset: {max_samples:,} samples")
            else:
                max_samples = MAX_SAMPLES
                print(f"📊 Using config default: {max_samples:,} samples")
        else:
            print(f"📊 Using specified samples: {max_samples:,} samples")
        
        print(f"🎯 Filtering for categories: {categories}")
        
        for s in self.dataset['train']:
            if 'categories' not in s or not s['categories']:
                continue
            
            # Split categories and check if any match selected categories
            sample_categories = [cat.strip() for cat in str(s['categories']).split()]
            
            # Check if any category matches (more flexible than single label only)
            if not any(cat in categories for cat in sample_categories):
                continue
                
            self.samples.append(s)
            
            if len(self.samples) >= max_samples:
                break
                
        print(f"✅ Selected samples: {len(self.samples):,}")
        
        # Print first 3 samples
        for sample in self.samples[:3]:
            print(f"Category: {sample['categories']}")
            print("Abstract:", sample['abstract'])
            print("#" * 20 + "\n")
    
    def discover_categories(self) -> List[str]:
        """
        Automatically discover all available categories from the loaded dataset
        
        Returns:
            List[str]: List of available category codes
        """
        if self.dataset is None:
            print("❌ No dataset loaded. Call load_dataset() first.")
            return []
        
        print("🔍 Discovering available categories...")
        
        # Extract all unique categories
        all_categories = set()
        category_counts = Counter()
        
        for sample in self.dataset['train']:
            if 'categories' in sample and sample['categories']:
                # Split multiple categories and clean
                categories = [cat.strip() for cat in str(sample['categories']).split()]
                all_categories.update(categories)
                
                # Count occurrences
                for cat in categories:
                    category_counts[cat] += 1
        
        # Convert to sorted list
        self.available_categories = sorted(all_categories)
        
        # Store category statistics
        self.category_stats = {
            'total_categories': len(self.available_categories),
            'total_samples': len(self.dataset['train']),
            'category_counts': dict(category_counts),
            'top_categories': sorted(
                category_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
        
        print(f"✅ Discovered {len(self.available_categories)} categories")
        print("📊 Top 5 categories by sample count:")
        for cat, count in self.category_stats['top_categories'][:5]:
            print(f"   {cat}: {count:,} samples")
        
        return self.available_categories
    
    def set_selected_categories(self, categories: List[str]) -> bool:
        """
        Set the categories to use for sample selection
        
        Args:
            categories: List of category codes to select
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not categories:
            print("❌ No categories provided")
            return False
        
        # Validate categories
        invalid_categories = [cat for cat in categories if cat not in self.available_categories]
        if invalid_categories:
            print(f"❌ Invalid categories: {invalid_categories}")
            return False
        
        self.selected_categories = categories
        print(f"✅ Set selected categories: {categories}")
        return True
    
    def get_category_recommendations(self, max_categories: int = 5) -> List[str]:
        """
        Get recommended categories based on sample count
        
        Args:
            max_categories: Maximum number of categories to recommend
            
        Returns:
            List[str]: Recommended category codes
        """
        if not self.category_stats:
            print("❌ No category statistics available. Call discover_categories() first.")
            return []
        
        # Get top categories by sample count
        recommended = [cat for cat, _ in self.category_stats['top_categories'][:max_categories]]
        print(f"💡 Recommended categories: {recommended}")
        return recommended
            
    def preprocess_samples(self, preprocessing_config: Dict = None) -> None:
        """Preprocess the selected samples with configurable options"""
        self.preprocessed_samples = []
        
        # Use default preprocessing config if none provided
        if preprocessing_config is None:
            preprocessing_config = {
                'text_cleaning': True,
                'data_validation': True,
                'category_mapping': True,
                'memory_optimization': True,
                # Advanced preprocessing options with defaults
                'rare_words_removal': False,
                'rare_words_threshold': 2,
                'lemmatization': False,
                'context_aware_stopwords': False,
                'stopwords_aggressiveness': 'Moderate',
                'phrase_detection': False,
                'min_phrase_freq': 3
            }
        
        print(f"🔧 [DATALOADER] Applying preprocessing with config: "
              f"{preprocessing_config}")
        
        for s in self.samples:
            abstract = s['abstract']
            
            # Text cleaning (configurable)
            if preprocessing_config.get('text_cleaning', True):
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
            
            # Apply advanced preprocessing options
            if preprocessing_config.get('rare_words_removal', False):
                # Professional rare words removal with global frequency analysis (ML standard)
                try:
                    import nltk
                    try:
                        # Download required NLTK data
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True)
                    
                    from nltk.tokenize import word_tokenize
                    
                    # Get global word frequencies from all samples (ML standard approach)
                    if not hasattr(self, '_global_word_freq'):
                        print("🔧 [DATALOADER] Computing global word frequencies for rare words removal...")
                        self._global_word_freq = {}
                        
                        # Count word frequencies across all samples (including stopwords)
                        total_samples = len(self.samples)
                        print(f"🔄 Computing global word frequencies from {total_samples:,} samples...")
                        
                        for i, sample in enumerate(self.samples):
                            sample_tokens = word_tokenize(sample['abstract'].lower())
                            for word in sample_tokens:
                                if len(word) > 1:  # Only exclude single characters
                                    self._global_word_freq[word] = self._global_word_freq.get(word, 0) + 1
                            
                            # Show progress every 2% or every 100 samples for better visibility
                            if (i + 1) % max(1, min(total_samples // 50, 100)) == 0 or i == total_samples - 1:
                                progress_percent = ((i + 1) / total_samples) * 100
                                progress_bar = self._create_progress_bar(progress_percent, 40)
                                print(f"\r🔄 Word Frequencies: {progress_bar} {progress_percent:.1f}% ({i + 1:,}/{total_samples:,})", end="", flush=True)
                        
                        print()  # New line after progress bar
                        print(f"🔧 [DATALOADER] Global word frequencies computed: {len(self._global_word_freq)} unique words")
                    
                    # Apply rare words removal using global frequencies
                    tokens = word_tokenize(abstract.lower())
                    
                    # Filter tokens based on global frequency threshold
                    threshold = preprocessing_config.get('rare_words_threshold', 2)
                    filtered_tokens = []
                    
                    for word in tokens:
                        # Keep only words above frequency threshold (including stopwords if they meet threshold)
                        if self._global_word_freq.get(word.lower(), 0) >= threshold:
                            filtered_tokens.append(word)
                    
                    abstract = ' '.join(filtered_tokens)
                    
                    removed_count = len(tokens) - len(filtered_tokens)
                    # Rare words removal applied silently
                        
                except ImportError:
                    # Fallback - no external libraries available for rare words removal
                    print("⚠️ [DATALOADER] NLTK not available, skipping rare words removal")
                    pass
            
            if preprocessing_config.get('lemmatization', False):
                # Professional lemmatization using NLTK WordNetLemmatizer
                try:
                    import nltk
                    try:
                        # Download WordNet if not available
                        nltk.data.find('corpora/wordnet')
                    except LookupError:
                        print("⚠️ [DATALOADER] WordNet not found, downloading...")
                        nltk.download('wordnet', quiet=True)
                    
                    try:
                        # Download POS tagger if not available
                        nltk.data.find('taggers/averaged_perceptron_tagger')
                    except LookupError:
                        print("⚠️ [DATALOADER] POS tagger not found, downloading...")
                        nltk.download('averaged_perceptron_tagger', quiet=True)
                    
                    # Verify all required data is available
                    try:
                        nltk.data.find('taggers/averaged_perceptron_tagger')
                        nltk.data.find('corpora/wordnet')
                        # NLTK data verified silently
                    except LookupError as e:
                        print(f"❌ [DATALOADER] NLTK data verification failed: {e}")
                        raise ImportError("Required NLTK data not available")
                    
                    from nltk.stem import WordNetLemmatizer
                    from nltk.corpus import wordnet
                    
                    def get_wordnet_pos(word):
                        """Map POS tag to first character lemmatize() accepts"""
                        try:
                            tag = nltk.pos_tag([word])[0][1][0].upper()
                            tag_dict = {"J": wordnet.ADJ,
                                      "N": wordnet.NOUN,
                                      "V": wordnet.VERB,
                                      "R": wordnet.ADV}
                            return tag_dict.get(tag, wordnet.NOUN)
                        except Exception as pos_error:
                            print(f"⚠️ [DATALOADER] POS tagging failed for word '{word}': {pos_error}")
                            # Fallback to noun
                            return wordnet.NOUN
                    
                    lemmatizer = WordNetLemmatizer()
                    words = abstract.split()
                    lemmatized_words = []
                    
                    for word in words:
                        # Get POS tag and lemmatize accordingly
                        pos = get_wordnet_pos(word)
                        lemmatized_word = lemmatizer.lemmatize(word, pos)
                        lemmatized_words.append(lemmatized_word)
                    
                    abstract = ' '.join(lemmatized_words)
                    # Lemmatization applied silently
                    
                except ImportError:
                    # Fallback - no external libraries available for lemmatization
                    print("⚠️ [DATALOADER] NLTK not available, skipping lemmatization")
                    pass
            
            if preprocessing_config.get('context_aware_stopwords', False):
                # Professional stopwords removal using NLTK or stop-words library only
                try:
                    # Try to use NLTK first (more comprehensive)
                    import nltk
                    try:
                        # Download stopwords if not available
                        nltk.data.find('corpora/stopwords')
                    except LookupError:
                        nltk.download('stopwords', quiet=True)
                    
                    from nltk.corpus import stopwords
                    english_stopwords = set(stopwords.words('english'))
                    
                    # Apply aggressiveness level using only library stopwords
                    aggressiveness = preprocessing_config.get('stopwords_aggressiveness', 'Moderate')
                    if aggressiveness == 'Conservative':
                        # Take first 20 most common stopwords from NLTK
                        all_stopwords = set(list(english_stopwords)[:20])
                    elif aggressiveness == 'Aggressive':
                        # Use all NLTK stopwords
                        all_stopwords = english_stopwords
                    else:  # 'Moderate' - use default NLTK stopwords
                        all_stopwords = english_stopwords
                    
                    words = abstract.split()
                    filtered_words = [word for word in words if word.lower() not in all_stopwords]
                    abstract = ' '.join(filtered_words)
                    
                    # Stopwords removal applied silently
                    
                except ImportError:
                    # Fallback to stop-words library if NLTK not available
                    try:
                        from stop_words import get_stop_words
                        english_stopwords = set(get_stop_words('en'))
                        
                        # Apply aggressiveness level for stop-words library
                        aggressiveness = preprocessing_config.get('stopwords_aggressiveness', 'Moderate')
                        if aggressiveness == 'Conservative':
                            # Take first 20 most common stopwords from the library
                            all_stopwords = set(list(english_stopwords)[:20])
                        elif aggressiveness == 'Aggressive':
                            all_stopwords = english_stopwords
                        else:  # 'Moderate'
                            all_stopwords = english_stopwords
                        
                        words = abstract.split()
                        filtered_words = [word for word in words if word.lower() not in all_stopwords]
                        abstract = ' '.join(filtered_words)
                        
                        # Stopwords removal applied silently
                        
                    except ImportError:
                        # Final fallback - no external libraries available
                        print("⚠️ [DATALOADER] NLTK and stop-words not available, skipping stopwords removal")
                        pass
            
            if preprocessing_config.get('phrase_detection', False):
                # Professional phrase detection using NLTK or basic n-gram analysis
                try:
                    import nltk
                    try:
                        # Download required NLTK data
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt', quiet=True)
                    
                    from nltk.util import ngrams
                    from nltk.tokenize import word_tokenize
                    
                    # Use NLTK's built-in collocation detection for phrases
                    min_freq = preprocessing_config.get('min_phrase_freq', 3)
                    
                    # Tokenize and create bigrams/trigrams
                    tokens = word_tokenize(abstract.lower())
                    bigrams = list(ngrams(tokens, 2))
                    trigrams = list(ngrams(tokens, 3))
                    
                    # Count all bigram and trigram frequencies (no predefined phrases)
                    phrase_counts = {}
                    
                    # Count bigrams
                    for bigram in bigrams:
                        phrase = ' '.join(bigram)
                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                    
                    # Count trigrams
                    for trigram in trigrams:
                        phrase = ' '.join(trigram)
                        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                    
                    # Replace frequent phrases with underscore version
                    replaced_count = 0
                    for phrase, count in phrase_counts.items():
                        if count >= min_freq:
                            # Replace with underscore version
                            underscore_phrase = phrase.replace(' ', '_')
                            abstract = abstract.replace(phrase, underscore_phrase)
                            replaced_count += 1
                    
                    # Phrase detection applied silently
                    
                except ImportError:
                    # Fallback - no external libraries available for phrase detection
                    print("⚠️ [DATALOADER] NLTK not available, skipping phrase detection")
                    pass
            
            # For the label, we only keep the first part (always applied)
            parts = s['categories'].split(' ')
            category = parts[0].split('.')[0]
            
            self.preprocessed_samples.append({
                "text": abstract,
                "label": category
            })
            
        # Apply additional preprocessing options
        if preprocessing_config.get('data_validation', True):
            # Clean samples but preserve count - don't remove any samples
            original_count = len(self.preprocessed_samples)
            cleaned_samples = []
            
            print(f"🧹 [DATALOADER] Starting data validation for {original_count:,} samples...")
            
            for i, sample in enumerate(self.preprocessed_samples):
                # Clean text and label
                text = sample['text'].strip() if sample['text'] else ""
                label = sample['label'].strip() if sample['label'] else ""
                
                # Always keep the sample, fill empty fields with defaults
                if not text:
                    text = "No text available"
                if not label:
                    label = "Unknown"
                
                cleaned_samples.append({
                    'text': text,
                    'label': label
                })
                
                # Show progress every 2% or every 1000 samples for better visibility
                if (i + 1) % max(1, min(original_count // 50, 1000)) == 0 or i == original_count - 1:
                    progress_percent = ((i + 1) / original_count) * 100
                    progress_bar = self._create_progress_bar(progress_percent, 40)
                    print(f"\r🧹 Data Validation: {progress_bar} {progress_percent:.1f}% ({i + 1:,}/{original_count:,})", end="", flush=True)
            
            print()  # New line after progress bar
            self.preprocessed_samples = cleaned_samples
            print(f"🧹 [DATALOADER] Data validation completed: {len(self.preprocessed_samples)} samples cleaned (preserved count)")
        
        if preprocessing_config.get('memory_optimization', True):
            # Convert text to category for memory efficiency
            print(f"💾 [DATALOADER] Applying memory optimization to {len(self.preprocessed_samples):,} samples...")
            
            for i, sample in enumerate(self.preprocessed_samples):
                sample['text'] = str(sample['text'])
                
                # Show progress every 5% or every 2000 samples for memory optimization
                if (i + 1) % max(1, min(len(self.preprocessed_samples) // 20, 2000)) == 0 or i == len(self.preprocessed_samples) - 1:
                    progress_percent = ((i + 1) / len(self.preprocessed_samples)) * 100
                    progress_bar = self._create_progress_bar(progress_percent, 40)
                    print(f"\r💾 Memory Optimization: {progress_bar} {progress_percent:.1f}% ({i + 1:,}/{len(self.preprocessed_samples):,})", end="", flush=True)
            
            print()  # New line after progress bar
            print(f"💾 [DATALOADER] Memory optimization completed")
        
        print(f"✅ [DATALOADER] Preprocessing completed: "
              f"{len(self.preprocessed_samples)} samples")
        # Preprocessing options applied silently
        
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
            
    def prepare_train_test_data(self, requested_samples: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and testing data with dynamic split based on requested samples"""
        X_full = [sample['text'] for sample in self.preprocessed_samples]
        y_full = [self.label_to_id[sample['label']] for sample in self.preprocessed_samples]
        
        total_samples = len(X_full)
        
        # Use dynamic split based on requested samples from Step 1
        if requested_samples and requested_samples > 0:
            # Calculate test size to get 20% of requested samples for test
            target_test_samples = int(requested_samples * 0.2)  # 20% for test
            target_cv_samples = requested_samples - target_test_samples  # 80% for CV
            
            print(f"📊 Total samples: {total_samples}")
            print(f"📊 Requested samples: {requested_samples}")
            print(f"📊 Target: {target_test_samples} test samples, {target_cv_samples} CV samples")
            
            if total_samples >= requested_samples:
                # Use requested samples with 20/80 split
                test_size = target_test_samples / total_samples
                print(f"📊 Using dynamic split: test_size={test_size:.3f} ({test_size*100:.1f}%)")
            else:
                # Not enough samples, use what we have with 20/80 split
                test_size = TEST_SIZE
                print(f"⚠️ Not enough samples ({total_samples} < {requested_samples}), using default test_size: {test_size}")
        else:
            # Fallback to original TEST_SIZE if no requested samples
            test_size = TEST_SIZE
            print(f"📊 No requested samples, using default test_size: {test_size}")
        
        # Check if we have enough samples for stratified split
        unique_classes = len(np.unique(y_full))
        test_samples = int(len(y_full) * test_size)
        
        if test_samples >= unique_classes:
            # Use stratified split if we have enough test samples
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, 
                test_size=test_size, 
                random_state=RANDOM_STATE, 
                stratify=y_full
            )
        else:
            # Use non-stratified split for small datasets
            print(f"⚠️ Small dataset detected: {len(y_full)} samples, {unique_classes} classes")
            print(f"   Test samples: {test_samples} < {unique_classes} classes")
            print(f"   Using non-stratified split to avoid error")
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, 
                test_size=test_size, 
                random_state=RANDOM_STATE
            )
        
        print(f"✅ CV samples: {len(X_train)} (for 5-fold CV)")
        print(f"✅ Test samples: {len(X_test)}")
        print(f"✅ Validation samples per fold: {len(X_train) // 5}")
        
        return X_train, X_test, y_train, y_test
        
    def get_sorted_labels(self) -> List[str]:
        """Get sorted list of labels"""
        return sorted(list(set([s['label'] for s in self.preprocessed_samples])))
