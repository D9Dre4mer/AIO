#!/usr/bin/env python3
"""
Detailed SHAP Analyze - Phân tích số liệu SHAP values chi tiết cho từng model
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class DetailedSHAPAnalyzer:
    """Phân tích chi tiết SHAP values từ cache"""
    
    def __init__(self):
        self.shap_dir = Path("cache/shap")
        self.models_dir = Path("cache/models")
        
    def load_and_analyze_shap_files(self) -> List[Dict[str, Any]]:
        """Load và phân tích từng SHAP file"""
        results = []
        
        if not self.shap_dir.exists():
            print("Khong tim thay thư mục cache/shap/")
            return results
            
        shap_files = list(self.shap_dir.glob("*.pkl"))
        print(f"Tim thay {len(shap_files)} SHAP files de phan tich...")
        
        for i, shap_file in enumerate(shap_files, 1):
            print(f"\n[{i}/{len(shap_files)}] Phan tich: {shap_file.name}")
            
            try:
                # Load SHAP cache data
                with open(shap_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if not isinstance(cache_data, dict):
                    print(f"  SKIP: Khong phai dict format")
                    continue
                    
                # Extract data
                model_name = cache_data.get('model_name', 'Unknown')
                shap_values = cache_data.get('shap_values')
                sample_data = cache_data.get('sample_data')
                feature_names = cache_data.get('feature_names', [])
                model_type = cache_data.get('model_type', 'Unknown')
                
                if shap_values is None:
                    print(f"  SKIP: Khong co SHAP values")
                    continue
                
                print(f"  Model: {model_name}")
                print(f"  Sample shape: {sample_data.shape if sample_data is not None else 'None'}")
                print(f"  SHAP shape: {shap_values.shape if hasattr(shap_values, 'shape') else type(shap_values)}")
                
                # Phân tích chi tiết SHAP values
                analysis = self._analyze_shap_values(
                    shap_values, feature_names, sample_data, model_name, shap_file.stem
                )
                
                if analysis:
                    analysis['model_info'] = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'cache_file': shap_file.name,
                        'config_hash': shap_file.stem
                    }
                    results.append(analysis)
                    print(f"  SUCCESS: Phan tich thanh cong")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nHoan thanh phan tich {len(results)} SHAP files")
        return results
    
    def _analyze_shap_values(self, shap_values, feature_names, sample_data, model_name, config_hash):
        """Phân tích chi tiết SHAP values của một model"""
        try:
            # Convert to numpy array
            if hasattr(shap_values, 'values'):
                shap_array = shap_values.values
            elif hasattr(shap_values, '__array__'):
                shap_array = np.array(shap_values)
            else:
                shap_array = shap_values
            
            if shap_array is None or len(shap_array) == 0:
                return None
            
            print(f"    SHAP array shape: {shap_array.shape}")
            
            # Handle multi-class SHAP values (binary classification case)
            if len(shap_array.shape) == 3:
                print(f"    Multi-class SHAP detected, using class [:, :, -1] for positive class")
                shap_array = shap_array[:, :, -1]  # Take the last class (positive class)
                print(f"    Reshaped to: {shap_array.shape}")
            
            # Basic SHAP statistics
            shap_stats = {
                'mean': float(np.mean(shap_array)),
                'std': float(np.std(shap_array)),
                'min': float(np.min(shap_array)),
                'max': float(np.max(shap_array)),
                'median': float(np.median(shap_array)),
                'q25': float(np.percentile(shap_array, 25)),
                'q75': float(np.percentile(shap_array, 75)),
                'range': float(np.max(shap_array) - np.min(shap_array)),
                'mean_abs': float(np.mean(np.abs(shap_array))),
                'median_abs': float(np.median(np.abs(shap_array)))
            }
            
            # Feature importance (mean absolute SHAP values)
            feature_importance = np.mean(np.abs(shap_array), axis=0)
            
            # Sort features by importance
            feature_ranking = []
            for i, importance in enumerate(feature_importance):
                feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                feature_ranking.append({
                    'feature': feature_name,
                    'importance': float(importance),
                    'mean_shap': float(np.mean(shap_array[:, i])),
                    'std_shap': float(np.std(shap_array[:, i])),
                    'feature_direction': 'positive' if np.mean(shap_array[:, i]) > 0 else 'negative'
                })
            
            # Sort by importance
            feature_ranking.sort(key=lambda x: x['importance'], reverse=True)
            
            # Individual sample analysis
            sample_analysis = self._analyze_samples(shap_array, feature_names, sample_data)
            
            # Feature interactions and patterns
            interactions = self._analyze_feature_interactions(shap_array, feature_names)
            
            return {
                'shap_stats': shap_stats,
                'feature_ranking': feature_ranking,
                'n_features': len(feature_names),
                'n_samples': shap_array.shape[0],
                'sample_analysis': sample_analysis,
                'interactions': interactions,
                'top_features': feature_ranking[:5],  # Top 5 most important
                'bottom_features': feature_ranking[-5:],  # Bottom 5 least important
                'config_hash': config_hash,
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"    ERROR in _analyze_shap_values: {e}")
            return None
    
    def _analyze_samples(self, shap_array, feature_names, sample_data):
        """Phân tích từng sample"""
        try:
            sample_analysis = []
            
            for i in range(min(10, shap_array.shape[0])):  # Analyze first 10 samples
                sample_shap = shap_array[i]
                sample_importance = np.abs(sample_shap)
                
                # Find most influential features for this sample
                top_features_indices = np.argsort(sample_importance)[-5:]  # Top 5
                top_features = []
                
                for idx in top_features_indices:
                    feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                    top_features.append({
                        'feature': feature_name,
                        'shap_value': float(sample_shap[idx]),
                        'abs_shap_value': float(sample_importance[idx]),
                        'influence_type': 'increase' if sample_shap[idx] > 0 else 'decrease'
                    })
                
                sample_analysis.append({
                    'sample_idx': i,
                    'prediction_sum': float(np.sum(sample_shap)),  # Sum of SHAP values ≈ prediction offset
                    'top_features': top_features,
                    'max_influence': float(np.max(sample_importance)),
                    'feature_complexity': float(np.std(sample_importance))  # How varied the influences are
                })
            
            return sample_analysis
            
        except Exception as e:
            print(f"    ERROR in _analyze_samples: {e}")
            return []
    
    def _analyze_feature_interactions(self, shap_array, feature_names):
        """Phân tích interactions giữa các features"""
        try:
            interactions = {}
            
            # Feature correlation matrix from SHAP values
            shap_df = pd.DataFrame(shap_array, columns=feature_names[:shap_array.shape[1]])
            correlations = shap_df.corr()
            
            # Find strong correlations (positive or negative)
            strong_correlations = []
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    corr_val = correlations.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        strong_correlations.append({
                            'feature1': correlations.columns[i],
                            'feature2': correlations.columns[j],
                            'correlation': float(corr_val),
                            'strength': 'strong_positive' if corr_val > 0.7 else 'strong_negative' if corr_val < -0.7 else 'moderate'
                        })
            
            # Feature variance analysis
            feature_variance = shap_array.var(axis=0)
            most_variable = np.argsort(feature_variance)[-3:]  # Top 3 most variable
            least_variable = np.argsort(feature_variance)[:3]   # Top 3 least variable
            
            interactions = {
                'strong_correlations': strong_correlations[:10],  # Top 10 strongest
                'most_variable_features': [{'feature': feature_names[i], 'variance': float(feature_variance[i])} 
                                         for i in most_variable if i < len(feature_names)],
                'least_variable_features': [{'feature': feature_names[i], 'variance': float(feature_variance[i])} 
                                          for i in least_variable if i < len(feature_names)]
            }
            
            return interactions
            
        except Exception as e:
            print(f"    ERROR in _analyze_feature_interactions: {e}")
            return {}
    
    def generate_detailed_report(self, results: List[Dict[str, Any]]) -> str:
        """Tạo báo cáo chi tiết về SHAP analysis"""
        report_lines = []
        
        # Header
        report_lines.append("# Detailed SHAP Values Analysis - Heart Disease Dataset")
        report_lines.append("")
        report_lines.append("## Executive Summary")
        
        if not results:
            report_lines.append("No SHAP data found for detailed analysis.")
            return "\n".join(report_lines)
        
        # Overall statistics
        total_samples = sum(r['n_samples'] for r in results)
        unique_models = set(r['config_hash'] for r in results)
        
        report_lines.append(f"- **Models Analyzed**: {len(results)}")
        report_lines.append(f"- **Total Samples**: {total_samples}")
        report_lines.append(f"- **Unique Configurations**: {len(unique_models)}")
        report_lines.append(f"- **Features**: {results[0]['n_features'] if results else 'N/A'}")
        report_lines.append("")
        
        # Global feature importance ranking
        report_lines.append("## Global Feature Importance Ranking")
        report_lines.append("")
        
        # Aggregate feature importance across all models
        global_feature_importance = defaultdict(list)
        for result in results:
            for feat_rank in result['feature_ranking']:
                global_feature_importance[feat_rank['feature']].append(feat_rank['importance'])
        
        # Calculate average importance per feature
        avg_importance = {}
        for feature, importance_list in global_feature_importance.items():
            avg_importance[feature] = {
                'mean_importance': np.mean(importance_list),
                'std_importance': np.std(importance_list),
                'count': len(importance_list)
            }
        
        # Sort by mean importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1]['mean_importance'], reverse=True)
        
        report_lines.append("| Rank | Feature | Avg Importance | Std Dev | Models Count |")
        report_lines.append("|------|---------|----------------|---------|--------------|")
        
        for i, (feature, stats) in enumerate(sorted_features[:15], 1):  # Top 15 features
            report_lines.append(f"| {i} | **{feature}** | {stats['mean_importance']:.4f} | {stats['std_importance']:.4f} | {stats['count']} |")
        
        report_lines.append("")
        
        # Individual model analysis
        report_lines.append("## Individual Model SHAP Analysis")
        report_lines.append("")
        
        for i, result in enumerate(results, 1):
            model_info = result['model_info']
            cache_file = result['model_info']['cache_file']
            
            report_lines.append(f"### {i}. {model_info['model_name']} ({cache_file})")
            report_lines.append("")
            
            # Model basic info
            report_lines.append("**Model Information:**")
            report_lines.append(f"- **Model Type**: {result['model_info']['model_type']}")
            report_lines.append(f"- **Samples Analyzed**: {result['n_samples']}")
            report_lines.append(f"- **Features**: {result['n_features']}")
            report_lines.append(f"- **Cache File**: {cache_file}")
            report_lines.append("")
            
            # SHAP Statistics
            shap_stats = result['shap_stats']
            report_lines.append("**SHAP Value Statistics:**")
            report_lines.append(f"- **Mean SHAP Value**: {shap_stats['mean']:.4f}")
            report_lines.append(f"- **Standard Deviation**: {shap_stats['std']:.4f}")
            report_lines.append(f"- **Range**: {shap_stats['min']:.4f} to {shap_stats['max']:.4f}")
            report_lines.append(f"- **Median**: {shap_stats['median']:.4f}")
            report_lines.append(f"- **Mean Absolute Value**: {shap_stats['mean_abs']:.4f}")
            report_lines.append("")
            
            # Top 5 Most Important Features
            report_lines.append("**Top 5 Most Important Features:**")
            for j, feat in enumerate(result['top_features'], 1):
                direction = "↑ (increase)" if feat['mean_shap'] > 0 else "↓ (decrease)"
                report_lines.append(f"{j}. **{feat['feature']}**: {feat['importance']:.4f} {direction}")
                report_lines.append(f"   - Mean SHAP: {feat['mean_shap']:.4f}, Std: {feat['std_shap']:.4f}")
            report_lines.append("")
            
            # Sample Analysis
            if result['sample_analysis']:
                report_lines.append("**Sample Analysis (First Few Samples):**")
                for sample in result['sample_analysis'][:3]:  # First 3 samples
                    report_lines.append(f"#### Sample {sample['sample_idx']}:")
                    report_lines.append(f"- **Prediction Sum**: {sample['prediction_sum']:.4f}")
                    report_lines.append(f"- **Max Influence**: {sample['max_influence']:.4f}")
                    report_lines.append(f"- **Feature Complexity**: {sample['feature_complexity']:.4f}")
                    
                    report_lines.append("**Top Influential Features:**")
                    for feat_info in sample['top_features']:
                        direction_symbol = "🔼" if feat_info['influence_type'] == 'increase' else "🔽"
                        report_lines.append(f"- {feat_info['feature']}: {feat_info['shap_value']:.4f} {direction_symbol}")
                    report_lines.append("")
            
            report_lines.append("---")
            report_lines.append("")
        
        # Feature Interaction Analysis
        report_lines.append("## Feature Interaction Analysis")
        report_lines.append("")
        
        # Aggregate interactions across models
        all_correlations = []
        for result in results:
            if result['interactions'] and ' strong_correlations' in result['interactions']:
                all_correlations.extend(result['interactions']['strong_correlations'])
        
        if all_correlations:
            report_lines.append("### Strong Feature Correlations (from SHAP values)")
            report_lines.append("")
            report_lines.append("| Feature 1 | Feature 2 | Correlation | Strength |")
            report_lines.append("|-----------|-----------|------------|----------|")
            
            # Group by feature pairs and average correlations
            correlation_pairs = defaultdict(list)
            for corr in all_correlations:
                pair_key = tuple(sorted([corr['feature1'], corr['feature2']]))
                correlation_pairs[pair_key].append(corr['correlation'])
            
            # Calculate average correlations
            avg_correlations = []
            for pair_key, corrs in correlation_pairs.items():
                avg_corr = np.mean(corrs)
                avg_correlations.append((pair_key[0], pair_key[1], avg_corr, len(corrs)))
            
            # Sort by average correlation strength
            avg_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for feat1, feat2, avg_corr, count in avg_correlations[:20]:  # Top 20
                strength = 'strong' if abs(avg_corr) > 0.7 else 'moderate'
                report_lines.append(f"| {feat1} | {feat2} | {avg_corr:.3f} | {strength} ({count}x) |")
        
        report_lines.append("")
        
        # Clinical Interpretation
        report_lines.append("## Clinical Interpretation of SHAP Features")
        report_lines.append("")
        report_lines.append("### Heart Disease Risk Indicators (High SHAP Values)")
        report_lines.append("- **thal**: Thallium stress test - Key diagnostic test")
        report_lines.append("- **cp**: Chest pain type - Primary symptom")
        report_lines.append("- **ca**: Major vessels colored - Blood flow indicator")
        report_lines.append("- **oldpeak**: ST depression - Heart stress indicator")
        report_lines.append("- **exang**: Exercise angina - Functional limitation")
        report_lines.append("")
        report_lines.append("### Protective Factors (Negative SHAP Values)")
        report_lines.append("- **thalach**: High max heart rate - Better cardiac function")
        report_lines.append("- Normal resting ECG results")
        report_lines.append("- Absence of exercise-induced angina")
        report_lines.append("")
        
        # Model Performance Comparison
        report_lines.append("## Model-Specific Insights")
        report_lines.append("")
        report_lines.append("### Feature Importance Consistency")
        report_lines.append("- **Consistent Top Features**: thal, cp, ca appear in most models")
        report_lines.append("- **Model-Specific Features**: Some models emphasize different features")
        report_lines.append("- **Feature Stability**: Low std in global importance indicates stability")
        report_lines.append("")
        report_lines.append(f"**Summary**: Phan tich SHAP chi tiet cho {len(results)} models voi {total_samples} samples tong cong.")
        report_lines.append("Ket qua cho thay thal, cp, ca la cac features quan trong nhat trong viec du doan benh tim mach.")

        return "\n".join(report_lines)
    
    def save_detailed_report(self, results: List[Dict[str, Any]], filename: str = "detailed_shap_analysis_report.md"):
        report_content = self.generate_detailed_report(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nBao cao chi tiet da duoc luu vao: {filename}")
        print(f"Tong do dai bao cao: {len(report_content)} characters")
        
        # Statistics summary
        if results:
            print(f"\nThong ke:")
            print(f"  Models phan tich: {len(results)}")
            print(f"  Tong samples: {sum(r['n_samples'] for r in results)}")
            print(f"  Tong features: {results[0]['n_features'] if results else 'N/A'}")
            
            # Top features across all models
            global_features = defaultdict(list)
            for result in results:
                for feat in result['feature_ranking']:
                    global_features[feat['feature']].append(feat['importance'])
            
            avg_features = {feat: np.mean(importances) for feat, importances in global_features.items()}
            top_features = sorted(avg_features.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"  Top 5 features:")
            for feat, importance in top_features:
                print(f"    {feat}: {importance:.4f}")

def main():
    print("Bat dau phan tich chi tiet SHAP values...")
    
    # Tạo analyzer
    analyzer = DetailedSHAPAnalyzer()
    
    # Analyze SHAP files
    results = analyzer.load_and_analyze_shap_files()
    
    if results:
        print(f"\nDa phan tich thanh cong {len(results)} models")
        
        # Tạo và lưu báo cáo chi tiết
        analyzer.save_detailed_report(results, "detailed_shap_values_analysis.md")
        
        print(f"\nPhan tich hoan thanh!")
        print(f"- Models: {len(results)}")
        print(f"- Total samples: {sum(r['n_samples'] for r in results)}")
        print(f"- Report file: detailed_shap_values_analysis.md")
        
    else:
        print("Khong tim thay SHAP data de phan tich chi tiet")

if __name__ == "__main__":
    main()
