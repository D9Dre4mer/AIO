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
        
    def load_dataset(self, skip_csv_prompt: bool = False) -> None:
        """Load the ArXiv abstracts dataset from HuggingFace and create CSV backup"""
        dataset_cache_path = (Path(self.cache_dir) / 
                             "UniverseTBD___arxiv-abstracts-large")
        csv_backup_path = Path(self.cache_dir) / "arxiv_dataset_backup.csv"
        
        if dataset_cache_path.exists():
            print(f"✅ Dataset found in cache: {dataset_cache_path}")
            print("📁 Loading from cache (no download needed)...")
        else:
            print(f"📥 Dataset not found in cache: {dataset_cache_path}")
            print("🌐 Will download dataset to cache...")
        
        # Load the dataset from the specified cache directory
        self.dataset = load_dataset("UniverseTBD/arxiv-abstracts-large", 
                                   cache_dir=self.cache_dir)
        
        print(f"🎉 Dataset loaded successfully!")
        print(f"📊 Dataset info: {self.dataset}")
        
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
        
    def select_samples(self, max_samples: int = None) -> None:
        """Select samples with single labels from specific categories"""
        self.samples = []
        
        # Use provided max_samples or fall back to config default
        max_samples = max_samples or MAX_SAMPLES
        
        for s in self.dataset['train']:
            if len(s['categories'].split(' ')) != 1:
                continue
                
            cur_category = s['categories'].strip().split('.')[0]
            if cur_category not in CATEGORIES_TO_SELECT:
                continue
                
            self.samples.append(s)
            
            if len(self.samples) >= max_samples:
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
