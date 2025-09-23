"""
Advanced LightGBM Optimization - Main Execution Script

This script demonstrates the complete pipeline for advanced LightGBM optimization:
1. Data loading and preprocessing
2. Advanced feature engineering
3. Hyperparameter optimization
4. Ensemble methods
5. Comprehensive evaluation
6. Model interpretability analysis
"""

import os
import sys
import yaml
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from feature_engineering import AdvancedFeatureEngineer
from hyperparameter_optimizer import HyperparameterOptimizer
from ensemble_methods import EnsembleMethods
from model_evaluator import ModelEvaluator
from lightgbm_advanced import AdvancedLightGBM

warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class AdvancedLightGBMPipeline:
    """
    Complete pipeline for advanced LightGBM optimization
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_loader = DataLoader(config_path)
        self.feature_engineer = AdvancedFeatureEngineer(self.config)
        self.hyperparameter_optimizer = HyperparameterOptimizer(self.config)
        self.ensemble_methods = EnsembleMethods(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        
        # Results storage
        self.results = {}
        self.models = {}
        
        print("🚀 Advanced LightGBM Pipeline Initialized")
        print(f"   Configuration: {config_path}")
        print(f"   Working Directory: {os.getcwd()}")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def run_complete_pipeline(self, dataset_name: str = 'fe_dt') -> dict:
        """
        Run the complete integrated pipeline with all features
        
        Args:
            dataset_name: Name of dataset to use ('raw', 'fe', 'dt', 'fe_dt')
            
        Returns:
            Dictionary with all results
        """
        print("🚀 Starting Complete Integrated Advanced LightGBM Pipeline")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Create results directory first
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(self.config['output']['results_dir']) / f"run_{timestamp}"
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create plots directory
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            print(f"📁 Results directory: {self.results_dir}")
            print(f"📊 Plots directory: {plots_dir}")
            
            # Integrated Dataset Comparison + Complete Pipeline
            print("\n📊 INTEGRATED DATASET COMPARISON + COMPLETE PIPELINE")
            print("=" * 70)
            
            # Define datasets to compare
            datasets = ['raw', 'fe', 'dt', 'fe_dt']
            comparison_results = {}
            
            print("\n📊 COMPARING ALL DATASETS WITH COMPLETE FEATURES")
            print("-" * 50)
            
            for dataset_name in datasets:
                print(f"\n🔍 Processing dataset: {dataset_name.upper()}")
                print("-" * 40)
                
                try:
                    # Load dataset
                    X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.load_dataset(dataset_name)
                    
                    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                    
                    # Advanced Feature Engineering
                    print("   🔧 Advanced Feature Engineering...")
                    X_train_processed, X_val_processed, X_test_processed = self.feature_engineer.create_comprehensive_features(
                        X_train, y_train, X_val, X_test
                    )
                    
                    # Find optimal LightGBM parameters with advanced settings
                    print("   🔍 Finding optimal parameters...")
                    best_n_estimators, cv_score = self._find_optimal_lightgbm(
                        X_train_processed, y_train,
                        n_estimators_range=range(100, 1001, 100),
                        cv_splits=5,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8
                    )
                    
                    # Run comprehensive hyperparameter optimization with Optuna
                    print(f"   🔍 Running Optuna optimization...")
                    study = self.hyperparameter_optimizer.optimize_with_optuna(
                        X_train_processed, y_train, X_val_processed, y_val, metric='accuracy'
                    )
                    
                    # Get best parameters from Optuna and fix conflicts
                    best_params = study.best_params.copy()
                    
                    # Fix LightGBM parameter conflicts
                    if best_params.get('force_col_wise') and best_params.get('force_row_wise'):
                        best_params['force_col_wise'] = True
                        best_params['force_row_wise'] = False
                    
                    # Add required parameters
                    best_params.update({
                        'n_estimators': best_n_estimators,
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'random_state': 42,
                        'verbose': -1,
                        'n_jobs': -1
                    })
                    
                    # Train Advanced LightGBM with optimized parameters
                    print("   🚀 Training Advanced LightGBM...")
                    lgb_model = AdvancedLightGBM(self.config, use_gpu=True)
                    lgb_model.train_model(X_train_processed, y_train, X_val_processed, y_val, params=best_params)
                    
                    # Comprehensive evaluation
                    print("   📊 Comprehensive evaluation...")
                    val_pred = lgb_model.predict(X_val_processed)
                    val_acc = accuracy_score(y_val, val_pred)
                    
                    test_pred = lgb_model.predict(X_test_processed)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    test_pred_proba = lgb_model.predict(X_test_processed, return_proba=True)
                    
                    # Calculate comprehensive metrics
                    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
                    comprehensive_metrics = {
                        'accuracy': test_acc,
                        'roc_auc': roc_auc_score(y_test, test_pred_proba),
                        'f1_score': f1_score(y_test, test_pred),
                        'precision': precision_score(y_test, test_pred),
                        'recall': recall_score(y_test, test_pred)
                    }
                    
                    # Store results
                    comparison_results[dataset_name] = {
                        'val_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'cv_score': cv_score,
                        'best_n_estimators': best_n_estimators,
                        'best_params': best_params,
                        'comprehensive_metrics': comprehensive_metrics,
                        'model': lgb_model,
                        'y_test': y_test,
                        'y_pred': test_pred,
                        'y_pred_proba': test_pred_proba,
                        'X_train_processed': X_train_processed,
                        'X_val_processed': X_val_processed,
                        'X_test_processed': X_test_processed
                    }
                    
                    print(f"   ✅ Validation Accuracy: {val_acc:.4f}")
                    print(f"   ✅ Test Accuracy: {test_acc:.4f}")
                    print(f"   ✅ CV Score: {cv_score:.4f}")
                    print(f"   ✅ ROC-AUC: {comprehensive_metrics['roc_auc']:.4f}")
                    print(f"   ✅ F1-Score: {comprehensive_metrics['f1_score']:.4f}")
                    print(f"   ✅ Best n_estimators: {best_n_estimators}")
                    
                except Exception as e:
                    print(f"   ❌ Error with dataset {dataset_name}: {e}")
                    comparison_results[dataset_name] = None
            
            # Create all comparison plots
            print("\n📊 Creating comprehensive comparison plots...")
            self._plot_dataset_comparison(comparison_results, plots_dir)
            self._plot_detailed_analysis(comparison_results, plots_dir)
            self._create_comprehensive_evaluation_plots(comparison_results, plots_dir)
            
            # Get the best dataset for final analysis (filter out None results)
            valid_results = {k: v for k, v in comparison_results.items() if v is not None}
            if not valid_results:
                print("❌ No valid results found! All datasets failed.")
                return {}
            
            best_dataset = max(valid_results.keys(), 
                             key=lambda x: valid_results[x]['test_accuracy'])
            
            print(f"\n🏆 Best dataset: {best_dataset.upper()}")
            print(f"   Test Accuracy: {comparison_results[best_dataset]['test_accuracy']:.4f}")
            
            # Use the best dataset for final complete pipeline
            best_result = comparison_results[best_dataset]
            X_train_processed = best_result['X_train_processed']
            X_val_processed = best_result['X_val_processed']
            X_test_processed = best_result['X_test_processed']
            y_test = best_result['y_test']
            
            # Store the best model
            self.models['advanced_lightgbm'] = best_result['model']
            self.results['advanced_lightgbm'] = best_result['model'].get_model_summary()
            
            # Additional Complete Pipeline Features
            print("\n🔧 ADDITIONAL COMPLETE PIPELINE FEATURES")
            print("-" * 50)
            
            # Ensemble Methods
            print("   🎭 Creating ensemble methods...")
            self._create_ensemble_methods(X_train_processed, y_train, X_val_processed, y_val)
            
            # Model Interpretability
            print("   🔍 Model interpretability analysis...")
            self._model_interpretability_analysis(X_train_processed, X_val_processed)
            
            # Additional comprehensive evaluation
            print("   📊 Additional comprehensive evaluation...")
            self._comprehensive_evaluation(X_test_processed, y_test)
            
            # Save all results
            print("   💾 Saving all results...")
            self._save_results()
            
            # Save dataset comparison results
            self._save_dataset_comparison_results(comparison_results)
            
            total_time = time.time() - start_time
            print(f"\n✅ Pipeline completed successfully in {total_time:.2f} seconds!")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise
    
    def run_dataset_comparison(self) -> dict:
        """
        Run LightGBM comparison across all datasets (raw, fe, dt, fe_dt)
        Similar to the XGBoost comparison in the notebook
        """
        print("🚀 Starting LightGBM Dataset Comparison")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Create results directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path(self.config['output']['results_dir']) / f"dataset_comparison_{timestamp}"
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create plots directory
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            print(f"📁 Results directory: {self.results_dir}")
            print(f"📊 Plots directory: {plots_dir}")
            
            # Define datasets to compare
            datasets = ['raw', 'fe', 'dt', 'fe_dt']
            results = {}
            
            print("\n📊 COMPARING ALL DATASETS")
            print("-" * 40)
            
            for dataset_name in datasets:
                print(f"\n🔍 Testing dataset: {dataset_name.upper()}")
                print("-" * 30)
                
                try:
                    # Load dataset
                    X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.load_dataset(dataset_name)
                    
                    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                    
                    # Find optimal LightGBM parameters with advanced settings
                    best_n_estimators, cv_score = self._find_optimal_lightgbm(
                        X_train, y_train,
                        n_estimators_range=range(100, 1001, 100),
                        cv_splits=5,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8
                    )
                    
                    # Run comprehensive hyperparameter optimization with Optuna
                    print(f"   🔍 Running Optuna optimization for {dataset_name}...")
                    study = self.hyperparameter_optimizer.optimize_hyperparameters(
                        X_train, y_train, X_val, y_val, n_trials=30
                    )
                    
                    # Get best parameters from Optuna
                    best_params = study.best_params
                    best_params.update({
                        'n_estimators': best_n_estimators,
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'random_state': 42,
                        'verbose': -1,
                        'n_jobs': -1
                    })
                    
                    # Train LightGBM model
                    lgb_model = AdvancedLightGBM(self.config, use_gpu=True)
                    
                    # Use optimized parameters
                    params = best_params
                    
                    # Add GPU parameters if available
                    if self.config.get('performance', {}).get('use_gpu', False):
                        params.update({
                            'device': 'gpu',
                            'gpu_platform_id': 0,
                            'gpu_device_id': 0,
                            'gpu_use_dp': True,
                            'gpu_max_memory_usage': 0.8
                        })
                    
                    # Train model
                    lgb_model.train_model(X_train, y_train, X_val, y_val, params=params)
                    
                    # Evaluate on validation set
                    val_pred = lgb_model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    
                    # Evaluate on test set
                    test_pred = lgb_model.predict(X_test)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    # Get probability predictions for comprehensive metrics
                    test_pred_proba = lgb_model.predict(X_test, return_proba=True)
                    
                    # Calculate comprehensive metrics
                    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
                    comprehensive_metrics = {
                        'accuracy': test_acc,
                        'roc_auc': roc_auc_score(y_test, test_pred_proba),
                        'f1_score': f1_score(y_test, test_pred),
                        'precision': precision_score(y_test, test_pred),
                        'recall': recall_score(y_test, test_pred)
                    }
                    
                    # Store results
                    results[dataset_name] = {
                        'val_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'cv_score': cv_score,
                        'best_n_estimators': best_n_estimators,
                        'best_params': best_params,
                        'comprehensive_metrics': comprehensive_metrics,
                        'model': lgb_model,
                        'y_test': y_test,
                        'y_pred': test_pred,
                        'y_pred_proba': test_pred_proba
                    }
                    
                    print(f"   ✅ Validation Accuracy: {val_acc:.4f}")
                    print(f"   ✅ Test Accuracy: {test_acc:.4f}")
                    print(f"   ✅ CV Score: {cv_score:.4f}")
                    print(f"   ✅ ROC-AUC: {comprehensive_metrics['roc_auc']:.4f}")
                    print(f"   ✅ F1-Score: {comprehensive_metrics['f1_score']:.4f}")
                    print(f"   ✅ Best n_estimators: {best_n_estimators}")
                    
                except Exception as e:
                    print(f"   ❌ Error with dataset {dataset_name}: {e}")
                    results[dataset_name] = {
                        'val_accuracy': 0.0,
                        'test_accuracy': 0.0,
                        'cv_score': 0.0,
                        'best_n_estimators': 0,
                        'model': None
                    }
            
            # Create comparison plots
            self._plot_dataset_comparison(results, plots_dir)
            
            # Create additional detailed plots
            self._plot_detailed_analysis(results, plots_dir)
            
            # Create comprehensive evaluation plots for each dataset
            self._create_comprehensive_evaluation_plots(results, plots_dir)
            
            # Save results
            self._save_dataset_comparison_results(results)
            
            total_time = time.time() - start_time
            print(f"\n✅ Dataset comparison completed in {total_time:.2f} seconds!")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Dataset comparison failed with error: {e}")
            raise
    
    def _plot_dataset_comparison(self, results, plots_dir):
        """Create comparison plot"""
        import matplotlib.pyplot as plt
        
        # Prepare data
        labels = ['Original', 'FE', 'Original + DT', 'FE + DT']
        dataset_keys = ['raw', 'fe', 'dt', 'fe_dt']
        
        # Filter out None results
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            print("❌ No valid results to plot!")
            return
            
        val_accs = [valid_results[key]['val_accuracy'] for key in dataset_keys if key in valid_results]
        test_accs = [valid_results[key]['test_accuracy'] for key in dataset_keys if key in valid_results]
        cv_scores = [valid_results[key]['cv_score'] for key in dataset_keys if key in valid_results]
        n_estimators = [valid_results[key]['best_n_estimators'] for key in dataset_keys if key in valid_results]
        
        # Set style
        plt.rcParams['font.family'] = 'Serif'
        
        # 1. Main Performance Comparison (Bar Chart)
        x = np.arange(len(labels))
        width = 0.3
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        rects1 = ax.bar(x - width/2, val_accs, width,
                        label='Validation Accuracy',
                        color='tab:blue', edgecolor='black', linewidth=1.2)
        rects2 = ax.bar(x + width/2, test_accs, width,
                        label='Test Accuracy',
                        color='tab:red', edgecolor='black', linewidth=1.2)
        
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title('LightGBM Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(ncol=2, loc="upper center")
        ax.grid(True, alpha=0.3)
        
        def autolabel(rects):
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:.3f}', xy=(rect.get_x()+rect.get_width()/2, h),
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "01_lightgbm_performance_comparison.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(plots_dir / "01_lightgbm_performance_comparison.pdf"), bbox_inches="tight")
        plt.close()
        
        # 2. Cross-Validation Scores Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, cv_scores, color='green', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.set_ylabel('Cross-Validation Score')
        ax.set_title('LightGBM Cross-Validation Scores by Dataset', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, cv_scores)):
            ax.annotate(f'{score:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom')
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "02_lightgbm_cv_scores.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 3. Optimal n_estimators Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, n_estimators, color='orange', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.set_ylabel('Optimal n_estimators')
        ax.set_title('LightGBM Optimal n_estimators by Dataset', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        for i, (bar, n_est) in enumerate(zip(bars, n_estimators)):
            ax.annotate(f'{n_est}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom')
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "03_lightgbm_optimal_estimators.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 4. Validation vs Test Accuracy Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (label, val_acc, test_acc) in enumerate(zip(labels, val_accs, test_accs)):
            ax.scatter(val_acc, test_acc, s=200, c=colors[i], alpha=0.7, 
                      edgecolors='black', linewidth=2, label=label)
            ax.annotate(label, (val_acc, test_acc), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add diagonal line (perfect correlation)
        min_val = min(min(val_accs), min(test_accs))
        max_val = max(max(val_accs), max(test_accs))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
        
        ax.set_xlabel('Validation Accuracy')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Validation vs Test Accuracy Correlation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "04_lightgbm_validation_vs_test.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 5. Performance Metrics Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create data matrix
        metrics_data = np.array([val_accs, test_accs, cv_scores]).T
        metrics_labels = ['Validation\nAccuracy', 'Test\nAccuracy', 'CV\nScore']
        
        im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
        
        # Set ticks and labels
        ax.set_xticks(range(len(metrics_labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(metrics_labels)
        ax.set_yticklabels(labels)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(metrics_labels)):
                text = ax.text(j, i, f'{metrics_data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('LightGBM Performance Heatmap', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy Score', rotation=270, labelpad=20)
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "05_lightgbm_performance_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 6. Performance Improvement Chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate improvements over baseline (raw dataset)
        baseline_val = val_accs[0]
        baseline_test = test_accs[0]
        
        val_improvements = [(acc - baseline_val) * 100 for acc in val_accs]
        test_improvements = [(acc - baseline_test) * 100 for acc in test_accs]
        
        x = np.arange(len(labels))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, val_improvements, width,
                        label='Validation Improvement (%)',
                        color='lightblue', edgecolor='black', linewidth=1.2)
        rects2 = ax.bar(x + width/2, test_improvements, width,
                        label='Test Improvement (%)',
                        color='lightcoral', edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Improvement (%)')
        ax.set_title('LightGBM Performance Improvement Over Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        def autolabel_improvement(rects):
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{h:+.1f}%', xy=(rect.get_x()+rect.get_width()/2, h),
                            ha='center', va='bottom' if h >= 0 else 'top')
        
        autolabel_improvement(rects1)
        autolabel_improvement(rects2)
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "06_lightgbm_improvement_chart.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        print("✅ All dataset comparison plots saved:")
        print("   1. Performance Comparison (Bar Chart)")
        print("   2. Cross-Validation Scores")
        print("   3. Optimal n_estimators")
        print("   4. Validation vs Test Accuracy Scatter")
        print("   5. Performance Heatmap")
        print("   6. Performance Improvement Chart")
    
    def _plot_detailed_analysis(self, results, plots_dir):
        """Create additional detailed analysis plots"""
        import matplotlib.pyplot as plt
        from math import pi
        
        # Prepare data
        labels = ['Original', 'FE', 'Original + DT', 'FE + DT']
        dataset_keys = ['raw', 'fe', 'dt', 'fe_dt']
        
        # Filter out None results
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            print("❌ No valid results to plot!")
            return
            
        val_accs = [valid_results[key]['val_accuracy'] for key in dataset_keys if key in valid_results]
        test_accs = [valid_results[key]['test_accuracy'] for key in dataset_keys if key in valid_results]
        cv_scores = [valid_results[key]['cv_score'] for key in dataset_keys if key in valid_results]
        n_estimators = [valid_results[key]['best_n_estimators'] for key in dataset_keys if key in valid_results]
        
        # 7. Radar Chart for Performance Metrics
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize n_estimators to 0-1 scale
        if max(n_estimators) > min(n_estimators):
            n_estimators_norm = [(n - min(n_estimators)) / (max(n_estimators) - min(n_estimators)) 
                                for n in n_estimators]
        else:
            n_estimators_norm = [0.5] * len(n_estimators)
        
        metrics = ['Validation\nAccuracy', 'Test\nAccuracy', 'CV\nScore', 'Normalized\nn_estimators']
        angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (label, val_acc, test_acc, cv_score, n_est_norm) in enumerate(
            zip(labels, val_accs, test_accs, cv_scores, n_estimators_norm)):
            
            values = [val_acc, test_acc, cv_score, n_est_norm]
            values = [float(v) for v in values]  # Ensure all values are float
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('LightGBM Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "07_lightgbm_radar_chart.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 8. Performance Trend Analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Trend 1: Accuracy progression
        x = np.arange(len(labels))
        ax1.plot(x, val_accs, 'o-', linewidth=3, markersize=8, label='Validation Accuracy', color='blue')
        ax1.plot(x, test_accs, 's-', linewidth=3, markersize=8, label='Test Accuracy', color='red')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('LightGBM Accuracy Trend Analysis', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (val, test) in enumerate(zip(val_accs, test_accs)):
            ax1.annotate(f'{val:.3f}', (i, val), textcoords="offset points", xytext=(0,10), ha='center')
            ax1.annotate(f'{test:.3f}', (i, test), textcoords="offset points", xytext=(0,-15), ha='center')
        
        # Trend 2: n_estimators vs performance
        ax2.scatter(n_estimators, val_accs, s=200, alpha=0.7, label='Validation Accuracy', color='blue')
        ax2.scatter(n_estimators, test_accs, s=200, alpha=0.7, label='Test Accuracy', color='red')
        
        # Add dataset labels
        for i, (n_est, val_acc, test_acc, label) in enumerate(zip(n_estimators, val_accs, test_accs, labels)):
            ax2.annotate(label, (n_est, val_acc), xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Optimal n_estimators')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('n_estimators vs Performance Relationship', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "08_lightgbm_trend_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 9. Performance Distribution Box Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create data for box plot
        all_accuracies = [val_accs, test_accs, cv_scores]
        box_labels = ['Validation\nAccuracy', 'Test\nAccuracy', 'CV\nScore']
        
        bp = ax.boxplot(all_accuracies, labels=box_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Accuracy Score')
        ax.set_title('LightGBM Performance Distribution Across Datasets', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(str(plots_dir / "09_lightgbm_distribution_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 10. Comprehensive Summary Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Accuracy comparison
        x = np.arange(len(labels))
        width = 0.35
        ax1.bar(x - width/2, val_accs, width, label='Validation', color='skyblue', alpha=0.8)
        ax1.bar(x + width/2, test_accs, width, label='Test', color='lightcoral', alpha=0.8)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: CV scores
        ax2.bar(labels, cv_scores, color='lightgreen', alpha=0.8)
        ax2.set_ylabel('CV Score')
        ax2.set_title('Cross-Validation Scores', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: n_estimators
        ax3.bar(labels, n_estimators, color='orange', alpha=0.8)
        ax3.set_ylabel('n_estimators')
        ax3.set_title('Optimal n_estimators', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Performance ratio (Test/Validation)
        ratios = [test/val if val > 0 else 0 for test, val in zip(test_accs, val_accs)]
        ax4.bar(labels, ratios, color='purple', alpha=0.8)
        ax4.set_ylabel('Test/Validation Ratio')
        ax4.set_title('Generalization Performance', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Generalization')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle('LightGBM Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(str(plots_dir / "10_lightgbm_comprehensive_summary.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        print("✅ Additional detailed analysis plots saved:")
        print("   7. Performance Radar Chart")
        print("   8. Trend Analysis")
        print("   9. Distribution Analysis")
        print("   10. Comprehensive Summary")
    
    def _create_comprehensive_evaluation_plots(self, results, plots_dir):
        """Create comprehensive evaluation plots for each dataset"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        import seaborn as sns
        
        print("\n📊 Creating comprehensive evaluation plots...")
        
        for dataset_name, result in results.items():
            if result is None or 'model' not in result or result['model'] is None:
                continue
                
            print(f"   📈 Creating plots for {dataset_name}...")
            
            # Get data
            y_test = result['y_test']
            y_pred = result['y_pred']
            y_pred_proba = result['y_pred_proba']
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Comprehensive Evaluation - {dataset_name.upper()} Dataset', fontsize=16, fontweight='bold')
            
            # 1. ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = result['comprehensive_metrics']['roc_auc']
            axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, 0].set_xlim([0.0, 1.0])
            axes[0, 0].set_ylim([0.0, 1.05])
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend(loc="lower right")
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            f1 = result['comprehensive_metrics']['f1_score']
            axes[0, 1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (F1 = {f1:.3f})')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend(loc="lower left")
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
            axes[0, 2].set_title('Confusion Matrix')
            axes[0, 2].set_xlabel('Predicted')
            axes[0, 2].set_ylabel('Actual')
            
            # 4. Prediction Distribution
            axes[1, 0].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Class 0', color='blue')
            axes[1, 0].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Class 1', color='red')
            axes[1, 0].set_xlabel('Predicted Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Prediction Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Metrics Comparison
            metrics = ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
            values = [
                result['comprehensive_metrics']['accuracy'],
                result['comprehensive_metrics']['roc_auc'],
                result['comprehensive_metrics']['f1_score'],
                result['comprehensive_metrics']['precision'],
                result['comprehensive_metrics']['recall']
            ]
            bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 6. Feature Importance (if available)
            try:
                model = result['model']
                if hasattr(model, 'get_feature_importance'):
                    importance = model.get_feature_importance()
                    if importance is not None:
                        feature_names = [f'Feature_{i}' for i in range(len(importance))]
                        axes[1, 2].barh(feature_names, importance, color='lightblue')
                        axes[1, 2].set_xlabel('Importance')
                        axes[1, 2].set_title('Feature Importance')
                    else:
                        axes[1, 2].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                                       ha='center', va='center', transform=axes[1, 2].transAxes)
                        axes[1, 2].set_title('Feature Importance')
                else:
                    axes[1, 2].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                                   ha='center', va='center', transform=axes[1, 2].transAxes)
                    axes[1, 2].set_title('Feature Importance')
            except:
                axes[1, 2].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Feature Importance')
            
            plt.tight_layout()
            plt.savefig(str(plots_dir / f"11_{dataset_name}_comprehensive_evaluation.png"), 
                       dpi=300, bbox_inches="tight")
            plt.close()
        
        print("✅ Comprehensive evaluation plots saved:")
        print("   11. Individual dataset comprehensive evaluation plots")
    
    def _run_integrated_dataset_comparison(self, plots_dir):
        """Run dataset comparison integrated into complete pipeline"""
        print("🚀 Starting Integrated Dataset Comparison")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Define datasets to compare
            datasets = ['raw', 'fe', 'dt', 'fe_dt']
            results = {}
            
            print("\n📊 COMPARING ALL DATASETS")
            print("-" * 40)
            
            for dataset_name in datasets:
                print(f"\n🔍 Testing dataset: {dataset_name.upper()}")
                print("-" * 30)
                
                try:
                    # Load dataset
                    X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.load_dataset(dataset_name)
                    
                    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                    
                    # Find optimal LightGBM parameters with advanced settings
                    best_n_estimators, cv_score = self._find_optimal_lightgbm(
                        X_train, y_train,
                        n_estimators_range=range(100, 1001, 100),
                        cv_splits=5,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8
                    )
                    
                    # Run comprehensive hyperparameter optimization with Optuna
                    print(f"   🔍 Running Optuna optimization for {dataset_name}...")
                    study = self.hyperparameter_optimizer.optimize_hyperparameters(
                        X_train, y_train, X_val, y_val, n_trials=30
                    )
                    
                    # Get best parameters from Optuna
                    best_params = study.best_params
                    best_params.update({
                        'n_estimators': best_n_estimators,
                        'objective': 'binary',
                        'metric': 'binary_logloss',
                        'boosting_type': 'gbdt',
                        'random_state': 42,
                        'verbose': -1,
                        'n_jobs': -1
                    })
                    
                    # Train LightGBM model
                    lgb_model = AdvancedLightGBM(self.config, use_gpu=True)
                    
                    # Use optimized parameters
                    params = best_params
                    
                    # Add GPU parameters if available
                    if self.config.get('performance', {}).get('use_gpu', False):
                        params.update({
                            'device': 'gpu',
                            'gpu_platform_id': 0,
                            'gpu_device_id': 0,
                            'gpu_use_dp': True,
                            'gpu_max_memory_usage': 0.8
                        })
                    
                    # Train model
                    lgb_model.train_model(X_train, y_train, X_val, y_val, params=params)
                    
                    # Evaluate on validation set
                    val_pred = lgb_model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    
                    # Evaluate on test set
                    test_pred = lgb_model.predict(X_test)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    # Get probability predictions for comprehensive metrics
                    test_pred_proba = lgb_model.predict(X_test, return_proba=True)
                    
                    # Calculate comprehensive metrics
                    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
                    comprehensive_metrics = {
                        'accuracy': test_acc,
                        'roc_auc': roc_auc_score(y_test, test_pred_proba),
                        'f1_score': f1_score(y_test, test_pred),
                        'precision': precision_score(y_test, test_pred),
                        'recall': recall_score(y_test, test_pred)
                    }
                    
                    # Store results
                    results[dataset_name] = {
                        'val_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'cv_score': cv_score,
                        'best_n_estimators': best_n_estimators,
                        'best_params': best_params,
                        'comprehensive_metrics': comprehensive_metrics,
                        'model': lgb_model,
                        'y_test': y_test,
                        'y_pred': test_pred,
                        'y_pred_proba': test_pred_proba
                    }
                    
                    print(f"   ✅ Validation Accuracy: {val_acc:.4f}")
                    print(f"   ✅ Test Accuracy: {test_acc:.4f}")
                    print(f"   ✅ CV Score: {cv_score:.4f}")
                    print(f"   ✅ ROC-AUC: {comprehensive_metrics['roc_auc']:.4f}")
                    print(f"   ✅ F1-Score: {comprehensive_metrics['f1_score']:.4f}")
                    print(f"   ✅ Best n_estimators: {best_n_estimators}")
                    
                except Exception as e:
                    print(f"   ❌ Error with dataset {dataset_name}: {e}")
                    results[dataset_name] = {
                        'val_accuracy': 0.0,
                        'test_accuracy': 0.0,
                        'cv_score': 0.0,
                        'best_n_estimators': 0,
                        'model': None
                    }
            
            # Create comparison plots
            self._plot_dataset_comparison(results, plots_dir)
            
            # Create additional detailed plots
            self._plot_detailed_analysis(results, plots_dir)
            
            # Create comprehensive evaluation plots for each dataset
            self._create_comprehensive_evaluation_plots(results, plots_dir)
            
            # Save results
            self._save_dataset_comparison_results(results)
            
            total_time = time.time() - start_time
            print(f"\n✅ Integrated dataset comparison completed in {total_time:.2f} seconds!")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Integrated dataset comparison failed with error: {e}")
            raise
    
    def _save_dataset_comparison_results(self, results):
        """Save dataset comparison results"""
        import json
        
        # Save results summary
        summary_path = self.results_dir / "dataset_comparison_results.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for dataset, data in results.items():
            json_results[dataset] = {
                'val_accuracy': float(data['val_accuracy']),
                'test_accuracy': float(data['test_accuracy']),
                'cv_score': float(data['cv_score']),
                'best_n_estimators': int(data['best_n_estimators'])
            }
        
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save detailed report
        report_path = self.results_dir / "dataset_comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write("LightGBM Dataset Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset, data in results.items():
                f.write(f"Dataset: {dataset.upper()}\n")
                f.write(f"  Validation Accuracy: {data['val_accuracy']:.4f}\n")
                f.write(f"  Test Accuracy: {data['test_accuracy']:.4f}\n")
                f.write(f"  CV Score: {data['cv_score']:.4f}\n")
                f.write(f"  Best n_estimators: {data['best_n_estimators']}\n")
                f.write("\n")
        
        print(f"✅ Results saved to {self.results_dir}")
    
    def _run_hyperparameter_optimization(self, X_train, y_train, X_val, y_val):
        """Run hyperparameter optimization"""
        print("🔍 Running Optuna optimization...")
        
        # Optuna optimization
        study = self.hyperparameter_optimizer.optimize_with_optuna(
            X_train, y_train, X_val, y_val, metric='accuracy'
        )
        
        # Store results
        self.results['hyperparameter_optimization'] = {
            'best_params': self.hyperparameter_optimizer.get_best_params(),
            'best_score': self.hyperparameter_optimizer.get_best_score(),
            'n_trials': len(study.trials)
        }
        
        print(f"✅ Best score: {self.hyperparameter_optimizer.get_best_score():.4f}")
        
        # Plot optimization history
        try:
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            self.hyperparameter_optimizer.plot_optimization_history(study, save_path=str(plots_dir / "01_hyperparameter_optimization.png"))
            print("   ✅ Hyperparameter optimization plot auto-saved")
        except Exception as e:
            print(f"⚠️  Could not plot optimization history: {e}")
    
    def _find_optimal_lightgbm(self, X_train, y_train, 
                              n_estimators_range=range(100, 1001, 100),
                              cv_splits=5, learning_rate=0.05, max_depth=8,
                              subsample=0.8, colsample_bytree=0.8):
        """
        Find optimal n_estimators using cross-validation like XGBoost
        """
        from sklearn.model_selection import StratifiedKFold
        import lightgbm as lgb
        import matplotlib.pyplot as plt
        
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        scores = []
        
        print(f"🔍 Testing {len(n_estimators_range)} different n_estimators values...")
        
        for n in n_estimators_range:
            # Create LightGBM parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            
            # Add GPU parameters if available
            if self.config.get('performance', {}).get('use_gpu', False):
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'gpu_use_dp': True,
                    'gpu_max_memory_usage': 0.8
                })
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Create datasets
                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)
                
                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=n,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100),
                        lgb.log_evaluation(0)
                    ]
                )
                
                # Make predictions
                pred = model.predict(X_val_cv, num_iteration=model.best_iteration)
                pred_binary = (pred > 0.5).astype(int)
                
                # Calculate accuracy
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val_cv, pred_binary)
                cv_scores.append(score)
            
            scores.append(np.mean(cv_scores))
            print(f"   n_estimators={n}: CV accuracy={np.mean(cv_scores):.4f}")
        
        # Plot results
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(list(n_estimators_range), scores, 'bo-')
        plt.title(f'Chọn n_estimators tối ưu cho LightGBM (CV={cv_splits}-fold)')
        plt.xlabel('n_estimators')
        plt.ylabel('Cross-Validation Accuracy')
        plt.grid(True)
        plt.savefig(str(plots_dir / "00_lightgbm_optimization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find best parameters
        best_n = list(n_estimators_range)[np.argmax(scores)]
        print(f"✅ n_estimators tối ưu (CV): {best_n}")
        
        return best_n, max(scores)

    def _train_advanced_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train advanced LightGBM model with improved optimization"""
        print("🚀 Training Advanced LightGBM with improved optimization...")
        
        # Find optimal n_estimators using cross-validation
        best_n_estimators, cv_score = self._find_optimal_lightgbm(
            X_train, y_train,
            n_estimators_range=range(50, 501, 50),
            cv_splits=3,
            learning_rate=0.1,
            max_depth=5,
            subsample=1.0,
            colsample_bytree=1.0
        )
        
        # Create advanced LightGBM model with optimized parameters
        lgb_model = AdvancedLightGBM(self.config, use_gpu=True)
        
        # Get optimized parameters from Optuna and update with best n_estimators
        best_params = self.hyperparameter_optimizer.get_best_params()
        if best_params:
            best_params.update({
                'n_estimators': best_n_estimators,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 1.0,
                'colsample_bytree': 1.0
            })
        else:
            # Fallback parameters if Optuna didn't run
            best_params = {
                'n_estimators': best_n_estimators,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'num_leaves': 31,
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
        
        print(f"📊 Using optimized parameters: n_estimators={best_n_estimators}, CV_score={cv_score:.4f}")
        
        # Train model
        lgb_model.train_model(X_train, y_train, X_val, y_val, params=best_params)
        
        # Store model
        self.models['advanced_lightgbm'] = lgb_model
        self.results['advanced_lightgbm'] = lgb_model.get_model_summary()
        
        print("✅ Advanced LightGBM training completed")
    
    def _create_ensemble_methods(self, X_train, y_train, X_val, y_val):
        """Create ensemble methods"""
        print("🎭 Creating ensemble methods...")
        
        # Create base models
        self.ensemble_methods.create_base_models(use_gpu=True)
        
        # Create advanced ensemble
        ensemble_results = self.ensemble_methods.create_advanced_ensemble(
            X_train, y_train, X_val, y_val
        )
        
        # Store results
        self.results['ensemble_methods'] = ensemble_results
        
        print("✅ Ensemble methods created")
    
    def _comprehensive_evaluation(self, X_test, y_test):
        """Perform comprehensive evaluation"""
        print("📊 Performing comprehensive evaluation...")
        
        evaluation_results = {}
        
        # Evaluate Advanced LightGBM
        if 'advanced_lightgbm' in self.models:
            lgb_model = self.models['advanced_lightgbm']
            y_pred = lgb_model.predict(X_test)
            y_pred_proba = lgb_model.predict(X_test, return_proba=True)
            
            # Store predictions for plotting
            self.y_test = y_test
            self.y_pred = y_pred
            self.y_pred_proba = y_pred_proba
            
            metrics = self.model_evaluator.calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba
            )
            evaluation_results['advanced_lightgbm'] = metrics
            
            # Plot evaluation metrics và tự động lưu
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # ROC Curve
            self.model_evaluator.plot_roc_curve(y_test, y_pred_proba, "Advanced LightGBM", save_path=str(plots_dir / "02_roc_curve.png"))
            print("   ✅ ROC curve plot auto-saved")
            
            # Precision-Recall Curve
            self.model_evaluator.plot_precision_recall_curve(y_test, y_pred_proba, "Advanced LightGBM", save_path=str(plots_dir / "03_precision_recall_curve.png"))
            print("   ✅ Precision-Recall curve plot auto-saved")
            
            # Confusion Matrix
            self.model_evaluator.plot_confusion_matrix(y_test, y_pred, "Advanced LightGBM", save_path=str(plots_dir / "04_confusion_matrix.png"))
            print("   ✅ Confusion matrix plot auto-saved")
            
            # Prediction Distribution
            self.model_evaluator.plot_prediction_distribution(y_test, y_pred_proba, "Advanced LightGBM", save_path=str(plots_dir / "05_prediction_distribution.png"))
            print("   ✅ Prediction distribution plot auto-saved")
        
        # Evaluate ensemble methods
        if 'ensemble_methods' in self.results:
            ensemble_results = self.results['ensemble_methods']
            
            for name, result in ensemble_results.items():
                try:
                    model = result['model']
                    y_pred = model.predict(X_test)
                    
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        y_pred_proba = y_pred.astype(float)
                    
                    metrics = self.model_evaluator.calculate_comprehensive_metrics(
                        y_test, y_pred, y_pred_proba
                    )
                    evaluation_results[name] = metrics
                    
                except Exception as e:
                    print(f"⚠️  Error evaluating {name}: {e}")
        
        # Store evaluation results
        self.results['evaluation'] = evaluation_results
        
        # Plot comparison
        if len(evaluation_results) > 1:
            # Metrics Comparison
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            self.model_evaluator.plot_metrics_comparison(evaluation_results, save_path=str(plots_dir / "06_metrics_comparison.png"))
            print("   ✅ Metrics comparison plot auto-saved")
            
            # Radar Chart
            self.model_evaluator.plot_radar_chart(evaluation_results, save_path=str(plots_dir / "07_radar_chart.png"))
            print("   ✅ Radar chart plot auto-saved")
        
        print("✅ Comprehensive evaluation completed")
    
    def _model_interpretability_analysis(self, X_train, X_val):
        """Perform model interpretability analysis"""
        print("🔍 Performing model interpretability analysis...")
        
        if 'advanced_lightgbm' in self.models:
            lgb_model = self.models['advanced_lightgbm']
            
            # Setup SHAP explainer
            lgb_model.setup_shap_explainer(X_train, X_val)
            
            # Plot feature importance và tự động lưu
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            lgb_model.plot_feature_importance(top_n=20, save_path=str(plots_dir / "08_feature_importance.png"))
            print("   ✅ Feature importance plot auto-saved")
            
            # Plot SHAP summary và tự động lưu
            lgb_model.plot_shap_summary(X_val, max_display=20, save_path=str(plots_dir / "09_shap_summary.png"))
            print("   ✅ SHAP summary plot auto-saved")
            
            # Plot SHAP waterfall for first instance và tự động lưu
            lgb_model.plot_shap_waterfall(X_val, instance_idx=0, save_path=str(plots_dir / "10_shap_waterfall.png"))
            print("   ✅ SHAP waterfall plot auto-saved")
            
            # Plot training history và tự động lưu
            lgb_model.plot_training_history(save_path=str(plots_dir / "11_training_history.png"))
            print("   ✅ Training history plot auto-saved")
        
        print("✅ Model interpretability analysis completed")
    
    def _save_results(self):
        """Save all results"""
        print("💾 Saving results...")
        
        # Use existing results directory (created in __init__)
        # Create plots directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📊 Plots directory: {plots_dir}")
        
        # Save plots
        self._save_plots(plots_dir)
        
        # Save models
        if 'advanced_lightgbm' in self.models:
            model_path = self.results_dir / "advanced_lightgbm_model.txt"
            self.models['advanced_lightgbm'].save_model(str(model_path))
        
        # Save ensemble models
        self.ensemble_methods.save_ensemble_models(str(self.results_dir / "ensemble_models"))
        
        # Save evaluation report
        if 'evaluation' in self.results:
            report_path = self.results_dir / "evaluation_report.txt"
            self.model_evaluator.generate_evaluation_report(
                self.results['evaluation'], str(report_path)
            )
        
        # Save configuration
        config_path = self.results_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # Save results summary
        summary_path = self.results_dir / "results_summary.json"
        import json
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, np.floating)):
                            json_results[key][k] = float(v)
                        elif hasattr(v, '__class__') and any(x in str(v.__class__) for x in ['VotingClassifier', 'StackingClassifier', 'WeightedEnsemble']):
                            # Skip non-serializable objects
                            json_results[key][k] = f"<{v.__class__.__name__}>"
                        elif hasattr(v, '__class__') and 'numpy' in str(v.__class__):
                            # Convert numpy arrays to lists
                            json_results[key][k] = v.tolist() if hasattr(v, 'tolist') else str(v)
                        else:
                            try:
                                json.dumps(v)  # Test if serializable
                                json_results[key][k] = v
                            except (TypeError, ValueError):
                                json_results[key][k] = f"<{v.__class__.__name__}>"
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to {self.results_dir}")
    
    def _save_plots(self, plots_dir: Path):
        """Đơn giản hóa: chỉ đếm số biểu đồ đã được lưu tự động"""
        print("📊 Checking auto-saved plots...")

        # Đếm số biểu đồ đã được lưu tự động
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            print(f"📈 Found {len(plot_files)} auto-saved plots:")
            for plot_file in sorted(plot_files):
                print(f"   - {plot_file.name}")
        else:
            print("⚠️  No plots were auto-saved")

        return len(plot_files)
    
    def run_quick_demo(self, dataset_name: str = 'fe') -> dict:
        """
        Run a quick demonstration of the pipeline
        
        Args:
            dataset_name: Name of dataset to use
            
        Returns:
            Dictionary with results
        """
        print("🚀 Running Quick Demo Pipeline")
        print("=" * 50)
        
        try:
            # Load data
            X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.load_dataset(dataset_name)
            
            # Quick feature engineering
            X_train_processed, X_val_processed, X_test_processed = self.feature_engineer.create_comprehensive_features(
                X_train, y_train, X_val, X_test
            )
            
            # Quick hyperparameter optimization (fewer trials)
            print("🔍 Quick hyperparameter optimization...")
            study = self.hyperparameter_optimizer.optimize_with_optuna(
                X_train_processed, y_train, X_val_processed, y_val, metric='accuracy'
            )
            
            # Train model
            print("🚀 Training model...")
            lgb_model = AdvancedLightGBM(self.config, use_gpu=True)
            best_params = self.hyperparameter_optimizer.get_best_params()
            lgb_model.train_model(X_train_processed, y_train, X_val_processed, y_val, params=best_params)
            
            # Evaluate
            print("📊 Evaluating model...")
            y_pred = lgb_model.predict(X_test_processed)
            y_pred_proba = lgb_model.predict(X_test_processed, return_proba=True)
            
            metrics = self.model_evaluator.calculate_comprehensive_metrics(
                y_test, y_pred, y_pred_proba
            )
            
            print("✅ Quick demo completed!")
            print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Test F1-Score: {metrics['f1']:.4f}")
            print(f"   Test AUC-ROC: {metrics['auc_roc']:.4f}")
            
            return {
                'metrics': metrics,
                'model': lgb_model,
                'best_params': best_params
            }
            
        except Exception as e:
            print(f"❌ Quick demo failed: {e}")
            raise


def main():
    """Main execution function"""
    print("🚀 Advanced LightGBM Optimization Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AdvancedLightGBMPipeline()
    
    # Check command line arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            print("Running quick demo...")
            results = pipeline.run_quick_demo('fe')
        elif sys.argv[1] == '--compare':
            print("Running dataset comparison...")
            results = pipeline.run_dataset_comparison()
        elif sys.argv[1] in ['raw', 'fe', 'dt', 'fe_dt']:
            dataset_name = sys.argv[1]
            print(f"Running complete pipeline with {dataset_name} dataset...")
            results = pipeline.run_complete_pipeline(dataset_name)
        else:
            print("Invalid argument. Use --quick, --compare, or dataset name (raw/fe/dt/fe_dt)")
            return None
    else:
        print("Running FULL pipeline with integrated dataset comparison by default...")
        results = pipeline.run_complete_pipeline('fe_dt')
    
    print("\n🎉 Pipeline execution completed!")
    return results


if __name__ == "__main__":
    main()
