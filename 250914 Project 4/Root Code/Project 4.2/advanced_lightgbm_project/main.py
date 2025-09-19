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
    
    def run_complete_pipeline(self, dataset_name: str = 'fe') -> dict:
        """
        Run the complete optimization pipeline
        
        Args:
            dataset_name: Name of dataset to use ('raw', 'fe', 'dt', 'fe_dt')
            
        Returns:
            Dictionary with all results
        """
        print("🚀 Starting Complete Advanced LightGBM Pipeline")
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
            
            # Step 1: Data Loading
            print("\n📊 STEP 1: DATA LOADING")
            print("-" * 40)
            X_train, y_train, X_val, y_val, X_test, y_test = self.data_loader.load_dataset(dataset_name)
            
            # Step 2: Advanced Feature Engineering
            print("\n🔧 STEP 2: ADVANCED FEATURE ENGINEERING")
            print("-" * 40)
            X_train_processed, X_val_processed, X_test_processed = self.feature_engineer.create_comprehensive_features(
                X_train, y_train, X_val, X_test
            )
            
            # Step 3: Hyperparameter Optimization
            print("\n🎯 STEP 3: HYPERPARAMETER OPTIMIZATION")
            print("-" * 40)
            self._run_hyperparameter_optimization(X_train_processed, y_train, X_val_processed, y_val)
            
            # Step 4: Advanced LightGBM Training
            print("\n🚀 STEP 4: ADVANCED LIGHTGBM TRAINING")
            print("-" * 40)
            self._train_advanced_lightgbm(X_train_processed, y_train, X_val_processed, y_val)
            
            # Step 5: Ensemble Methods
            print("\n🎭 STEP 5: ENSEMBLE METHODS")
            print("-" * 40)
            self._create_ensemble_methods(X_train_processed, y_train, X_val_processed, y_val)
            
            # Step 6: Comprehensive Evaluation
            print("\n📊 STEP 6: COMPREHENSIVE EVALUATION")
            print("-" * 40)
            self._comprehensive_evaluation(X_test_processed, y_test)
            
            # Step 7: Model Interpretability
            print("\n🔍 STEP 7: MODEL INTERPRETABILITY")
            print("-" * 40)
            self._model_interpretability_analysis(X_train_processed, X_val_processed)
            
            # Step 8: Save Results
            print("\n💾 STEP 8: SAVE RESULTS")
            print("-" * 40)
            self._save_results()
            
            total_time = time.time() - start_time
            print(f"\n✅ Pipeline completed successfully in {total_time:.2f} seconds!")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise
    
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
    
    def _train_advanced_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train advanced LightGBM model"""
        print("🚀 Training Advanced LightGBM...")
        
        # Create advanced LightGBM model
        lgb_model = AdvancedLightGBM(self.config, use_gpu=True)
        
        # Get optimized parameters
        best_params = self.hyperparameter_optimizer.get_best_params()
        
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
    
    # Check if we should run quick demo or full pipeline
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        print("Running quick demo...")
        results = pipeline.run_quick_demo('fe')
    else:
        print("Running complete pipeline...")
        results = pipeline.run_complete_pipeline('fe')
    
    print("\n🎉 Pipeline execution completed!")
    return results


if __name__ == "__main__":
    main()
