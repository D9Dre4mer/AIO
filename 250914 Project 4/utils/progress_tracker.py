#!/usr/bin/env python3
"""
Progress Tracking Utility for Model Training
Provides progress bars and time estimation for training processes
"""

import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import queue

class ProgressTracker:
    """Progress tracker with time estimation and progress bars"""
    
    def __init__(self, total_tasks: int, task_name: str = "Training"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.task_name = task_name
        self.start_time = time.time()
        self.task_times = []
        self.current_task = ""
        self.current_task_start = None
        
        # Threading for real-time updates
        self.update_queue = queue.Queue()
        self.stop_updates = False
        self.update_thread = None
        
        # Print initial progress bar
        if total_tasks > 0:
            print(f"\n🚀 Starting {task_name} - {total_tasks} tasks to complete")
            self._print_progress()
        
    def start_task(self, task_description: str):
        """Start a new task"""
        if self.current_task_start:
            self.end_task()
        
        self.current_task = task_description
        self.current_task_start = time.time()
        self._print_progress()
        
    def end_task(self):
        """End current task and update progress"""
        if self.current_task_start:
            task_time = time.time() - self.current_task_start
            self.task_times.append(task_time)
            self.completed_tasks += 1
            self.current_task_start = None
            self._print_progress()
            
    def update_progress(self, additional_completed: int = 0):
        """Update progress manually"""
        self.completed_tasks += additional_completed
        self._print_progress()
        
    def _print_progress(self):
        """Print progress bar and time estimation"""
        if self.total_tasks == 0:
            return
            
        # Calculate progress percentage
        progress = self.completed_tasks / self.total_tasks
        percentage = progress * 100
        
        # Create progress bar
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        
        if self.completed_tasks > 0:
            avg_time_per_task = elapsed_time / self.completed_tasks
            remaining_tasks = self.total_tasks - self.completed_tasks
            estimated_remaining = avg_time_per_task * remaining_tasks
            
            # Format time
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining)
            
            # Calculate completion time
            completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
            completion_str = completion_time.strftime("%H:%M:%S")
        else:
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = "Calculating..."
            completion_str = "Calculating..."
        
        # Print progress bar on a new line to avoid conflicts with other prints
        print(f"\n📊 {self.task_name}: [{bar}] {percentage:.1f}% "
              f"({self.completed_tasks}/{self.total_tasks}) "
              f"⏱️ {elapsed_str} | ⏳ {remaining_str} | 🎯 {completion_str}")
        
        # Print current task if available
        if self.current_task:
            print(f"   🔄 {self.current_task}")
            
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        
    def finish(self):
        """Finish progress tracking"""
        self.end_task()
        total_time = time.time() - self.start_time
        
        # Show final completion message
        print(f"\n{'='*80}")
        print(f"🎉 {self.task_name.upper()} COMPLETED!")
        print(f"{'='*80}")
        print(f"📊 Total time: {self._format_time(total_time)}")
        if self.total_tasks > 0:
            print(f"📈 Average time per task: {self._format_time(total_time/self.total_tasks)}")
        print(f"🎯 Completed: {self.completed_tasks}/{self.total_tasks} tasks")
        
        if self.task_times:
            print(f"⚡ Fastest task: {self._format_time(min(self.task_times))}")
            print(f"🐌 Slowest task: {self._format_time(max(self.task_times))}")
        print(f"{'='*80}")

class ModelTrainingProgress:
    """Specialized progress tracker for model training"""
    
    def __init__(self, models: List[str], vectorizations: List[str]):
        self.models = models
        self.vectorizations = vectorizations
        self.total_combinations = len(models) * len(vectorizations)
        
        # Create main progress tracker
        self.main_tracker = ProgressTracker(self.total_combinations, "Model Training")
        
        # Track individual model progress
        self.model_trackers = {}
        self.current_model = None
        self.current_vectorization = None
        
    def start_combination(self, model: str, vectorization: str):
        """Start training a model-vectorization combination"""
        self.current_model = model
        self.current_vectorization = vectorization
        
        combination_name = f"{model} + {vectorization}"
        
        # Print a clear separator and progress update
        print(f"\n{'='*80}")
        print(f"🔄 TRAINING: {combination_name.upper()}")
        print(f"{'='*80}")
        
        # Update main progress tracker
        self.main_tracker.start_task(f"Training {combination_name}")
        
        # Create model-specific tracker if needed
        if model not in self.model_trackers:
            self.model_trackers[model] = ProgressTracker(1, f"{model} Training")
            
    def update_model_progress(self, progress: float, stage: str = ""):
        """Update progress for current model"""
        if self.current_model and self.current_model in self.model_trackers:
            tracker = self.model_trackers[self.current_model]
            if stage:
                tracker.start_task(f"{stage} - {progress:.1%}")
            else:
                tracker.update_progress()
                
    def end_combination(self):
        """End current model-vectorization combination"""
        self.main_tracker.end_task()
        
        # Print completion message with clear formatting
        if self.current_model and self.current_vectorization:
            print(f"\n{'='*80}")
            print(f"✅ COMPLETED: {self.current_model.upper()} + {self.current_vectorization.upper()}")
            print(f"{'='*80}")
        
        self.current_model = None
        self.current_vectorization = None
        
    def finish(self):
        """Finish all training progress tracking"""
        self.main_tracker.finish()
        
        print("\n📊 Model Performance Summary:")
        print("-" * 40)
        for model, tracker in self.model_trackers.items():
            if tracker.completed_tasks > 0:
                print(f"   {model}: {tracker.completed_tasks} combinations completed")

class EmbeddingProgress:
    """Progress tracker for embedding generation"""
    
    def __init__(self, total_samples: int, embedding_type: str = "Text Embeddings"):
        self.total_samples = total_samples
        self.embedding_type = embedding_type
        self.processed_samples = 0
        self.start_time = time.time()
        self.batch_size = 1000  # Default batch size
        
    def update_batch(self, batch_size: int):
        """Update progress after processing a batch"""
        self.processed_samples += batch_size
        self._print_progress()
        
    def _print_progress(self):
        """Print embedding progress"""
        if self.total_samples == 0:
            return
            
        progress = self.processed_samples / self.total_samples
        percentage = progress * 100
        
        # Create progress bar
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        
        if self.processed_samples > 0:
            avg_time_per_sample = elapsed_time / self.processed_samples
            remaining_samples = self.total_samples - self.processed_samples
            estimated_remaining = avg_time_per_sample * remaining_samples
            
            # Format time
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining)
            
            # Calculate completion time
            completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
            completion_str = completion_time.strftime("%H:%M:%S")
        else:
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = "Calculating..."
            completion_str = "Calculating..."
        
        # Print progress
        print(f"\r{self.embedding_type}: [{bar}] {percentage:.1f}% "
              f"({self.processed_samples:,}/{self.total_samples:,}) "
              f"⏱️ {elapsed_str} | ⏳ {remaining_str} | 🎯 {completion_str}", end="", flush=True)
        
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m {seconds%60:.0f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
            
    def finish(self):
        """Finish embedding progress"""
        total_time = time.time() - self.start_time
        
        print(f"\n✅ {self.embedding_type} completed!")
        print(f"   📊 Total time: {self._format_time(total_time)}")
        print(f"   📈 Samples processed: {self.processed_samples:,}")
        print(f"   ⚡ Average speed: {self.processed_samples/total_time:.0f} samples/sec")

def create_training_progress(models: List[str], vectorizations: List[str]) -> ModelTrainingProgress:
    """Create a training progress tracker"""
    return ModelTrainingProgress(models, vectorizations)

def create_embedding_progress(total_samples: int, embedding_type: str = "Text Embeddings") -> EmbeddingProgress:
    """Create an embedding progress tracker"""
    return EmbeddingProgress(total_samples, embedding_type)

# Example usage
if __name__ == "__main__":
    # Test progress tracker
    print("Testing Progress Tracker...")
    
    # Test model training progress
    models = ["Logistic Regression", "SVM", "KNN"]
    vectorizations = ["BoW", "TF-IDF", "Embeddings"]
    
    progress = create_training_progress(models, vectorizations)
    
    for model in models:
        for vec in vectorizations:
            progress.start_combination(model, vec)
            time.sleep(1)  # Simulate training
            progress.end_combination()
    
    progress.finish()
    
    print("\n" + "="*50)
    
    # Test embedding progress
    embedding_progress = create_embedding_progress(10000, "Sentence Transformers")
    
    for i in range(0, 10000, 1000):
        embedding_progress.update_batch(1000)
        time.sleep(0.1)  # Simulate processing
    
    embedding_progress.finish()
