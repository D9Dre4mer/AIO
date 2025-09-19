"""
Quick Start Script for Advanced LightGBM Optimization

This script provides a simple interface to run the optimization pipeline
with different configurations and datasets.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import AdvancedLightGBMPipeline


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Advanced LightGBM Optimization Pipeline')
    
    parser.add_argument('--dataset', '-d', 
                       choices=['raw', 'fe', 'dt', 'fe_dt'],
                       default='fe',
                       help='Dataset to use (default: fe)')
    
    parser.add_argument('--mode', '-m',
                       choices=['quick', 'full', 'demo'],
                       default='quick',
                       help='Run mode (default: quick)')
    
    parser.add_argument('--config', '-c',
                       default='config/config.yaml',
                       help='Configuration file path (default: config/config.yaml)')
    
    parser.add_argument('--trials', '-t',
                       type=int,
                       default=50,
                       help='Number of optimization trials (default: 50)')
    
    parser.add_argument('--gpu', '-g',
                       action='store_true',
                       help='Force GPU usage')
    
    parser.add_argument('--no-ensemble',
                       action='store_true',
                       help='Skip ensemble methods')
    
    parser.add_argument('--output-dir', '-o',
                       default='results',
                       help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    print("🚀 Advanced LightGBM Optimization Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Trials: {args.trials}")
    print(f"GPU: {'Yes' if args.gpu else 'Auto'}")
    print(f"Ensemble: {'No' if args.no_ensemble else 'Yes'}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    try:
        # Load and modify configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update configuration based on arguments
        config['optimization']['n_trials'] = args.trials
        config['output']['results_dir'] = args.output_dir
        
        if args.gpu:
            config['performance']['use_gpu'] = True
        
        if args.no_ensemble:
            config['ensemble']['voting'] = False
            config['ensemble']['stacking'] = False
            config['ensemble']['blending'] = False
        
        # Save modified config
        modified_config_path = 'config/modified_config.yaml'
        with open(modified_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Initialize pipeline
        pipeline = AdvancedLightGBMPipeline(modified_config_path)
        
        # Run pipeline based on mode
        if args.mode == 'quick':
            print("\n🚀 Running Quick Mode...")
            results = pipeline.run_quick_demo(args.dataset)
            
        elif args.mode == 'demo':
            print("\n🚀 Running Demo Mode...")
            # Run with minimal configuration for demo
            config['optimization']['n_trials'] = 10
            config['optimization']['timeout'] = 300
            config['feature_engineering']['max_features'] = 20
            
            # Save demo config
            demo_config_path = 'config/demo_config.yaml'
            with open(demo_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            pipeline = AdvancedLightGBMPipeline(demo_config_path)
            results = pipeline.run_complete_pipeline(args.dataset)
            
        else:  # full
            print("\n🚀 Running Full Mode...")
            results = pipeline.run_complete_pipeline(args.dataset)
        
        # Display results summary
        print("\n📊 RESULTS SUMMARY")
        print("=" * 40)
        
        if 'metrics' in results:
            # Quick mode results
            metrics = results['metrics']
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
            
        elif 'evaluation' in results:
            # Full mode results
            evaluation = results['evaluation']
            print("Model Performance Comparison:")
            for model_name, metrics in evaluation.items():
                print(f"\n{model_name}:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  F1-Score:  {metrics['f1']:.4f}")
                print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
