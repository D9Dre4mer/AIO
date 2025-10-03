#!/usr/bin/env python3
"""
Script để tạo confusion matrix thực tế từ eval_predictions trong cache
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import confusion_matrix
import json

class ConfusionMatrixGenerator:
    """Tạo confusion matrix từ cache data"""
    
    def __init__(self, cache_dir: str = "cache/models"):
        self.cache_dir = Path(cache_dir)
        
    def load_eval_predictions(self, cache_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Đọc eval_predictions từ cache"""
        try:
            # Đọc eval_predictions.parquet
            eval_file = Path(cache_path) / "eval_predictions.parquet"
            if eval_file.exists():
                eval_df = pd.read_parquet(eval_file)
            else:
                print(f"Khong tim thay eval_predictions.parquet trong {cache_path}")
                return None, None, {}
            
            # Đọc label_mapping.json
            label_file = Path(cache_path) / "label_mapping.json"
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    label_mapping = json.load(f)
            else:
                print(f"Khong tim thay label_mapping.json trong {cache_path}")
                label_mapping = {}
            
            # Trích xuất y_true và y_pred
            if 'y_true' in eval_df.columns:
                y_true = eval_df['y_true'].values
            elif 'true_labels' in eval_df.columns:
                y_true = eval_df['true_labels'].values
            else:
                print(f"Khong tim thay cot y_true trong eval_predictions")
                return None, None, {}
            
            if 'y_pred' in eval_df.columns:
                y_pred = eval_df['y_pred'].values
            elif 'predictions' in eval_df.columns:
                y_pred = eval_df['predictions'].values
            else:
                # Tìm probability columns
                proba_cols = [col for col in eval_df.columns if col.startswith('proba__class_')]
                if proba_cols:
                    proba_values = eval_df[proba_cols].values
                    y_pred = np.argmax(proba_values, axis=1)
                else:
                    print(f"Khong tim thay predictions trong eval_predictions")
                    return None, None, {}
            
            print(f"Doc thanh cong: {len(y_true)} samples")
            print(f"   y_true shape: {y_true.shape}")
            print(f"   y_pred shape: {y_pred.shape}")
            print(f"   Unique labels: {sorted(set(y_true) | set(y_pred))}")
            
            return y_true, y_pred, label_mapping
            
        except Exception as e:
            print(f"Loi doc eval_predictions tu {cache_path}: {e}")
            return None, None, {}
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              label_mapping: Dict, model_name: str, scaler_name: str) -> Dict[str, Any]:
        """Tạo confusion matrix từ y_true và y_pred"""
        try:
            # Tạo confusion matrix
            unique_labels = sorted(list(set(np.concatenate([y_true, y_pred]))))
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            
            # Tạo label names
            if label_mapping:
                # Convert string keys to int if needed
                int_label_mapping = {}
                for k, v in label_mapping.items():
                    try:
                        int_label_mapping[int(k)] = v
                    except:
                        int_label_mapping[k] = v
                
                label_names = [int_label_mapping.get(label_id, f"Class_{label_id}") 
                              for label_id in unique_labels]
            else:
                label_names = [f"Class_{label_id}" for label_id in unique_labels]
            
            # Tính metrics
            if cm.shape == (2, 2):  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                }
            else:
                # Multiclass - tính accuracy tổng
                accuracy = np.trace(cm) / np.sum(cm)
                metrics = {'accuracy': accuracy}
            
            result = {
                'confusion_matrix': cm.tolist(),
                'label_names': label_names,
                'unique_labels': unique_labels,
                'metrics': metrics,
                'model_name': model_name,
                'scaler_name': scaler_name
            }
            
            return result
            
        except Exception as e:
            print(f"Loi tao confusion matrix: {e}")
            return {}
    
    def generate_confusion_matrix_text(self, cm_result: Dict[str, Any]) -> str:
        """Tạo text representation của confusion matrix"""
        if not cm_result:
            return "Khong co du lieu"
        
        cm = np.array(cm_result['confusion_matrix'])
        label_names = cm_result['label_names']
        metrics = cm_result['metrics']
        
        text = []
        text.append("**Confusion Matrix:**")
        text.append("```")
        
        # Header
        header = "                Predicted"
        text.append(header)
        pred_header = "                " + "  ".join([f"{name:>6}" for name in label_names])
        text.append(pred_header)
        
        # Rows
        for i, true_label in enumerate(label_names):
            row = f"Actual {true_label:>6}  " + "  ".join([f"{cm[i,j]:>6}" for j in range(len(label_names))])
            text.append(row)
        
        text.append("```")
        text.append("")
        
        # Metrics
        text.append("**Metrics:**")
        if 'tn' in metrics:  # Binary classification
            text.append(f"- **True Negatives (TN)**: {metrics['tn']}")
            text.append(f"- **False Positives (FP)**: {metrics['fp']}")
            text.append(f"- **False Negatives (FN)**: {metrics['fn']}")
            text.append(f"- **True Positives (TP)**: {metrics['tp']}")
        
        text.append(f"- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        if 'precision' in metrics:
            text.append(f"- **Precision**: {metrics['precision']:.4f}")
            text.append(f"- **Recall**: {metrics['recall']:.4f}")
            text.append(f"- **F1-Score**: {metrics['f1_score']:.4f}")
        
        return "\n".join(text)
    
    def process_all_models(self) -> Dict[str, Any]:
        """Xử lý tất cả models và tạo confusion matrix"""
        print("Bat dau tao confusion matrix tu cache data...")
        
        results = {}
        
        # Danh sách các models cần xử lý cho cleveland dataset
        models_to_process = [
            ('adaboost', 'MinMaxScaler'),
            ('adaboost', 'RobustScaler'),
            ('adaboost', 'StandardScaler'),
            ('catboost', 'MinMaxScaler'),
            ('catboost', 'RobustScaler'),
            ('catboost', 'StandardScaler'),
            ('decision_tree', 'MinMaxScaler'),
            ('decision_tree', 'RobustScaler'),
            ('decision_tree', 'StandardScaler'),
            ('gradient_boosting', 'MinMaxScaler'),
            ('gradient_boosting', 'RobustScaler'),
            ('gradient_boosting', 'StandardScaler'),
            ('knn', 'MinMaxScaler'),
            ('knn', 'RobustScaler'),
            ('knn', 'StandardScaler'),
            ('lightgbm', 'MinMaxScaler'),
            ('lightgbm', 'RobustScaler'),
            ('lightgbm', 'StandardScaler'),
            ('logistic_regression', 'MinMaxScaler'),
            ('logistic_regression', 'RobustScaler'),
            ('logistic_regression', 'StandardScaler'),
            ('naive_bayes', 'MinMaxScaler'),
            ('naive_bayes', 'RobustScaler'),
            ('naive_bayes', 'StandardScaler'),
            ('random_forest', 'MinMaxScaler'),
            ('random_forest', 'RobustScaler'),
            ('random_forest', 'StandardScaler'),
            ('stacking_ensemble_logistic_regression', 'MinMaxScaler'),
            ('stacking_ensemble_logistic_regression', 'RobustScaler'),
            ('stacking_ensemble_logistic_regression', 'StandardScaler'),
            ('svm', 'MinMaxScaler'),
            ('svm', 'RobustScaler'),
            ('svm', 'StandardScaler'),
            ('voting_ensemble_hard', 'MinMaxScaler'),
            ('voting_ensemble_hard', 'RobustScaler'),
            ('voting_ensemble_hard', 'StandardScaler'),
            ('xgboost', 'MinMaxScaler'),
            ('xgboost', 'RobustScaler'),
            ('xgboost', 'StandardScaler')
        ]
        
        for model_name, scaler_name in models_to_process:
            print(f"\nXu ly {model_name} + {scaler_name}")
            
            # Tìm cache directory
            model_dir = self.cache_dir / model_name / f"numeric_dataset_{scaler_name}"
            if not model_dir.exists():
                print(f"Khong tim thay {model_dir}")
                continue
            
            # Tìm subdirectory với hash
            subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"Khong tim thay subdirectory trong {model_dir}")
                continue
            
            cache_path = subdirs[0]  # Lấy subdirectory đầu tiên
            print(f"   Cache path: {cache_path}")
            
            # Đọc eval_predictions
            y_true, y_pred, label_mapping = self.load_eval_predictions(str(cache_path))
            if y_true is None or y_pred is None:
                continue
            
            # Tạo confusion matrix
            cm_result = self.create_confusion_matrix(y_true, y_pred, label_mapping, model_name, scaler_name)
            if cm_result:
                results[f"{model_name}_{scaler_name}"] = cm_result
                print(f"   Tao thanh cong confusion matrix")
            else:
                print(f"   Khong the tao confusion matrix")
        
        return results

def main():
    """Hàm chính"""
    print("Bat dau tao confusion matrix tu cache data...")
    
    # Tạo generator
    generator = ConfusionMatrixGenerator()
    
    # Xử lý tất cả models
    results = generator.process_all_models()
    
    if results:
        print(f"\nDa tao confusion matrix cho {len(results)} models")
        
        # Tạo báo cáo với confusion matrix
        report_lines = []
        report_lines.append("# Confusion Matrix Thuc Te - Cleveland Heart Disease Dataset")
        report_lines.append("")
        report_lines.append("## Tong Quan")
        report_lines.append("- **Dataset**: Cleveland Heart Disease Dataset")
        report_lines.append("- **Binary Classification**: 0 = No Heart Disease, 1 = Heart Disease")
        report_lines.append("- **Models**: 12 models với 3 scalers mỗi model (36 combinations)")
        report_lines.append("- **Scalers**: MinMaxScaler, RobustScaler, StandardScaler")
        report_lines.append("")
        
        # Thêm confusion matrix cho từng model
        for key, cm_result in results.items():
            model_name = cm_result['model_name'].replace('_', ' ').title()
            scaler_name = cm_result['scaler_name']
            
            report_lines.append(f"## {model_name} ({scaler_name})")
            report_lines.append("")
            
            # Thêm confusion matrix text
            cm_text = generator.generate_confusion_matrix_text(cm_result)
            report_lines.append(cm_text)
            report_lines.append("")
        
        # Thêm phần tổng kết và phân tích theo models
        report_lines.append("---")
        report_lines.append("")
        report_lines.append("## Tong Ket va Phan Tich")
        report_lines.append("")
        
        # Nhóm theo models để phân tích
        model_groups = {}
        for key, cm_result in results.items():
            model_name = cm_result['model_name']
            if model_name not in model_groups:
                model_groups[model_name] = {}
            model_groups[model_name][cm_result['scaler_name']] = cm_result
        
        # Tạo bảng tổng kết
        report_lines.append("### Bang Tong Ket Theo Models")
        report_lines.append("")
        report_lines.append("| Model | MinMaxScaler | RobustScaler | StandardScaler | Best Scaler |")
        report_lines.append("|-------|--------------|--------------|----------------|-------------|")
        
        for model_name, scalers_data in model_groups.items():
            minmax_acc = scalers_data.get('MinMaxScaler', {}).get('metrics', {}).get('accuracy', 0)
            robust_acc = scalers_data.get('RobustScaler', {}).get('metrics', {}).get('accuracy', 0)
            standard_acc = scalers_data.get('StandardScaler', {}).get('metrics', {}).get('accuracy', 0)
            
            minmax_str = f"{minmax_acc:.3f}" if minmax_acc > 0 else "N/A"
            robust_str = f"{robust_acc:.3f}" if robust_acc > 0 else "N/A"
            standard_str = f"{standard_acc:.3f}" if standard_acc > 0 else "N/A"
            
            # Tìm best scaler
            best_scaler = "N/A"
            best_acc = 0
            for scaler_name, acc in [('MinMaxScaler', minmax_acc), ('RobustScaler', robust_acc), ('StandardScaler', standard_acc)]:
                if acc > best_acc:
                    best_acc = acc
                    best_scaler = scaler_name
            
            model_display = model_name.replace('_', ' ').title()
            report_lines.append(f"| {model_display} | {minmax_str} | {robust_str} | {standard_str} | {best_scaler} |")
        
        report_lines.append("")
        
        # Top performers
        report_lines.append("### Top Performers")
        report_lines.append("")
        all_results = []
        for key, cm_result in results.items():
            accuracy = cm_result.get('metrics', {}).get('accuracy', 0)
            all_results.append((accuracy, cm_result['model_name'], cm_result['scaler_name']))
        
        all_results.sort(reverse=True)
        
        report_lines.append("| Rank | Model | Scaler | Accuracy |")
        report_lines.append("|------|-------|--------|----------|")
        
        for i, (acc, model, scaler) in enumerate(all_results[:10], 1):  # Top 10
            if acc > 0:
                model_display = model.replace('_', ' ').title()
                report_lines.append(f"| {i} | {model_display} | {scaler} | {acc:.4f} ({acc*100:.2f}%) |")
        
        report_lines.append("")
        
        # Lưu báo cáo
        report_content = "\n".join(report_lines)
        with open('confusion_matrix_real_data.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Bao cao da duoc luu vao: confusion_matrix_real_data.md")
        
        # Hiển thị preview
        print("\nPreview bao cao:")
        print("-" * 50)
        print(report_content[:1000] + "..." if len(report_content) > 1000 else report_content)
        
    else:
        print("Khong co du lieu de tao confusion matrix")

if __name__ == "__main__":
    main()
