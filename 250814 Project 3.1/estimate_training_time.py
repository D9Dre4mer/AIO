#!/usr/bin/env python3
"""
Training Time Estimation Script
Estimates training time for large datasets (300,000+ samples)
"""

import time
import psutil
import numpy as np
from datetime import datetime, timedelta

def estimate_training_time():
    """Estimate training time for 300,000 samples"""
    
    print("🕐 TRAINING TIME ESTIMATION FOR 300,000 SAMPLES")
    print("=" * 60)
    
    # Dataset parameters
    total_samples = 300000
    train_samples = int(total_samples * 0.8)  # 240,000
    test_samples = int(total_samples * 0.2)   # 60,000
    cv_folds = 5
    
    print(f"📊 Dataset Configuration:")
    print(f"   • Total samples: {total_samples:,}")
    print(f"   • Training samples: {train_samples:,}")
    print(f"   • Test samples: {test_samples:,}")
    print(f"   • CV folds: {cv_folds}")
    print()
    
    # Model configurations
    models = [
        ("Logistic Regression", "Fast", 0.1),
        ("SVM", "Medium", 0.3),
        ("Random Forest", "Medium", 0.4),
        ("KNN", "Slow", 0.8),
        ("Naive Bayes", "Fast", 0.05),
        ("Decision Tree", "Fast", 0.2)
    ]
    
    vectorization_methods = [
        ("BoW", "Medium", 0.2),
        ("TF-IDF", "Medium", 0.25),
        ("Embeddings", "Slow", 0.6)
    ]
    
    print("🎯 Model Performance Estimates:")
    print("-" * 40)
    
    total_combinations = len(models) * len(vectorization_methods)
    print(f"Total combinations: {total_combinations}")
    print()
    
    # Time estimates per model-vectorization combination
    time_estimates = []
    
    for model_name, model_speed, model_factor in models:
        for vec_name, vec_speed, vec_factor in vectorization_methods:
            # Base time calculation
            base_time = 30  # Base time in seconds for small dataset
            
            # Scale by dataset size (300k vs 1k samples = 300x)
            size_factor = (total_samples / 1000) ** 0.8  # Sub-linear scaling
            
            # Scale by model complexity
            model_time = base_time * model_factor * size_factor
            
            # Scale by vectorization complexity
            vec_time = base_time * vec_factor * size_factor
            
            # Total time for this combination
            total_time = model_time + vec_time
            
            # Add CV overhead (5-fold)
            cv_overhead = total_time * (cv_folds - 1) * 0.8
            final_time = total_time + cv_overhead
            
            time_estimates.append((f"{model_name} + {vec_name}", final_time))
    
    # Sort by time
    time_estimates.sort(key=lambda x: x[1])
    
    print("⏱️  Time Estimates per Combination:")
    print("-" * 50)
    
    total_estimated_time = 0
    for combo, time_sec in time_estimates:
        time_min = time_sec / 60
        time_hour = time_min / 60
        total_estimated_time += time_sec
        
        if time_hour >= 1:
            print(f"   {combo:<30} {time_hour:.1f}h")
        else:
            print(f"   {combo:<30} {time_min:.1f}m")
    
    print("-" * 50)
    
    # Total time estimation
    total_hours = total_estimated_time / 3600
    total_days = total_hours / 24
    
    print(f"📈 TOTAL ESTIMATED TIME:")
    print(f"   • Total seconds: {total_estimated_time:,.0f}s")
    print(f"   • Total minutes: {total_estimated_time/60:,.0f}m")
    print(f"   • Total hours: {total_hours:.1f}h")
    print(f"   • Total days: {total_days:.2f} days")
    print()
    
    # Memory estimation
    print("💾 MEMORY ESTIMATION:")
    print("-" * 30)
    
    # Estimate memory usage
    memory_per_sample = 0.001  # 1KB per sample (rough estimate)
    total_memory_gb = (total_samples * memory_per_sample) / (1024**3)
    
    print(f"   • Estimated memory usage: {total_memory_gb:.2f} GB")
    print(f"   • Recommended RAM: {total_memory_gb * 2:.1f} GB")
    print()
    
    # System check
    print("🖥️  SYSTEM CHECK:")
    print("-" * 20)
    
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    
    print(f"   • Available RAM: {available_gb:.1f} GB / {total_gb:.1f} GB")
    print(f"   • RAM usage: {memory.percent:.1f}%")
    
    if available_gb < total_memory_gb * 2:
        print(f"   ⚠️  WARNING: May need more RAM for optimal performance")
    else:
        print(f"   ✅ RAM should be sufficient")
    
    print()
    
    # Progress tracking
    print("📊 PROGRESS TRACKING:")
    print("-" * 25)
    
    current_time = datetime.now()
    estimated_completion = current_time + timedelta(seconds=total_estimated_time)
    
    print(f"   • Start time: {current_time.strftime('%H:%M:%S')}")
    print(f"   • Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
    print(f"   • Duration: {total_hours:.1f} hours")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 20)
    print("   1. Run overnight or during off-hours")
    print("   2. Monitor memory usage during training")
    print("   3. Consider reducing CV folds to 3 for faster training")
    print("   4. Use smaller sample size for initial testing")
    print("   5. Enable progress logging to track completion")
    print()
    
    # Quick test option
    print("🚀 QUICK TEST OPTION:")
    print("-" * 25)
    print("   For faster testing, consider:")
    print("   • Sample size: 10,000-50,000 samples")
    print("   • CV folds: 3 instead of 5")
    print("   • Estimated time: 10-30 minutes")
    print()
    
    return total_estimated_time

if __name__ == "__main__":
    try:
        total_time = estimate_training_time()
        print("✅ Estimation completed successfully!")
    except Exception as e:
        print(f"❌ Error during estimation: {e}")
