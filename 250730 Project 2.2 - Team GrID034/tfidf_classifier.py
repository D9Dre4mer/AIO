"""
Mô-đun phân loại spam dựa trên TF-IDF sử dụng scikit-learn.
Phiên bản nâng cấp với preprocessing, feature engineering và auto-tuning.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class TFIDFClassifier:
    """Bộ phân loại TF-IDF nâng cấp sử dụng SVM với preprocessing và auto-tuning."""
    
    def __init__(self, max_features: int = 2000):
        """
        Khởi tạo bộ phân loại.
        
        Args:
            max_features: Số đặc trưng tối đa cho TF-IDF (giới hạn kích thước từ vựng để tăng tốc).
        """
        self.max_features = max_features
        
        # Build pipeline với các cải tiến
        self.model = self._build_optimized_pipeline()
        
        # Thêm các thuộc tính để tracking
        self._is_fitted = False
        self._best_params = None
    
    def _clean_text(self, text: str) -> str:
        """
        Làm sạch và chuẩn hóa văn bản.
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Văn bản đã được làm sạch
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Chuyển về lowercase
        text = text.lower()
        
        # Loại bỏ các ký tự đặc biệt, chỉ giữ chữ cái và số
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Loại bỏ nhiều spaces liên tiếp
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _build_optimized_pipeline(self):
        """Xây dựng pipeline tối ưu với các cải tiến từ Streamlit app."""
        
        # Base pipeline với các tham số tối ưu
        base_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',           # Loại bỏ stop words
                max_df=0.9,                     # Loại bỏ từ xuất hiện quá nhiều (>90% docs)
                min_df=2,                       # Loại bỏ từ xuất hiện quá ít (<2 docs)
                ngram_range=(1, 2),             # Sử dụng unigrams và bigrams
                lowercase=True,                 # Chuyển về lowercase
                strip_accents='unicode'         # Loại bỏ dấu
            )),
            ('svd', TruncatedSVD(
                n_components=50,                # Giảm chiều để tiết kiệm memory
                random_state=42
            )),
            ('clf', LinearSVC(
                max_iter=5000,                  # Tăng max_iter cho convergence tốt hơn
                random_state=42,
                C=1.0                          # Regularization parameter
            ))
        ])
        
        # Parameter grid cho GridSearchCV
        param_grid = {
            'svd__n_components': [50, 100],     # Thử các giá trị SVD components
            'clf__C': [0.1, 1, 10]              # Thử các giá trị regularization
        }
        
        # Wrap với GridSearchCV để auto-tuning
        optimized_model = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=3,                               # 3-fold cross validation
            scoring='f1_macro',                 # Sử dụng F1-macro cho balanced evaluation
            n_jobs=1,                           # Sử dụng 1 CPU core để tránh lỗi parallel processing trên Windows
            verbose=0                           # Tắt verbose để không spam console
        )
        
        return optimized_model
    
    def fit(self, train_texts: List[str], train_labels: List[str]) -> None:
        """
        Huấn luyện mô hình trên dữ liệu.
        
        Args:
            train_texts: Danh sách tin nhắn đã được tiền xử lý.
            train_labels: Danh sách nhãn ('spam' hoặc 'ham').
        """
        # Làm sạch tất cả texts trước khi training
        cleaned_texts = [self._clean_text(text) for text in train_texts]
        
        # Fit model với GridSearchCV auto-tuning
        self.model.fit(cleaned_texts, train_labels)
        
        # Lưu best parameters và đánh dấu đã fitted
        self._best_params = self.model.best_params_
        self._is_fitted = True
    
    def predict(self, text: str, return_proba: bool = False) -> Dict[str, Any]:
        """
        Dự đoán nhãn cho một văn bản.
        
        Args:
            text: Văn bản đầu vào (tiền xử lý trước khi gọi).
            return_proba: Nếu True, trả về cả xác suất.
        
        Returns:
            Dict chứa 'prediction' và tùy chọn 'probabilities'.
        """
        if not self._is_fitted:
            raise ValueError("Model chưa được training. Gọi fit() trước!")
        
        # Làm sạch text trước khi predict
        cleaned_text = self._clean_text(text)
        
        # Predict
        pred = self.model.predict([cleaned_text])[0]
        result = {'prediction': pred}
        
        # Thêm probabilities nếu được yêu cầu
        if return_proba:
            try:
                # LinearSVC không có predict_proba, sử dụng decision_function
                if hasattr(self.model, 'decision_function'):
                    decision_scores = self.model.decision_function([cleaned_text])[0]
                    
                    # Convert decision scores thành probabilities
                    # Đối với binary classification
                    if len(self.model.classes_) == 2:
                        # Sử dụng sigmoid function
                        prob_positive = 1 / (1 + np.exp(-decision_scores))
                        prob_negative = 1 - prob_positive
                        proba = np.array([prob_negative, prob_positive])
                    else:
                        # Multi-class: sử dụng softmax
                        exp_scores = np.exp(decision_scores - np.max(decision_scores))
                        proba = exp_scores / np.sum(exp_scores)
                    
                    # Tạo dict probabilities
                    probabilities = {}
                    for i, class_name in enumerate(self.model.classes_):
                        probabilities[class_name] = proba[i]
                    
                    result['probabilities'] = probabilities
                
                elif hasattr(self.model, 'predict_proba'):
                    # Fallback nếu có predict_proba
                    proba = self.model.predict_proba([cleaned_text])[0]
                    probabilities = {}
                    for i, class_name in enumerate(self.model.classes_):
                        probabilities[class_name] = proba[i]
                    result['probabilities'] = probabilities
                
                else:
                    # Không thể tính probabilities
                    result['probabilities'] = None
                    
            except Exception as e:
                # Fallback: return None nếu có lỗi
                result['probabilities'] = None
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin về model đã được training.
        
        Returns:
            Dict chứa thông tin model
        """
        if not self._is_fitted:
            return {
                'status': 'not_fitted',
                'message': 'Model chưa được training'
            }
        
        info = {
            'status': 'fitted',
            'max_features': self.max_features,
            'best_params': self._best_params,
            'best_cv_score': getattr(self.model, 'best_score_', None),
            'classes': list(self.model.classes_) if hasattr(self.model, 'classes_') else None,
            'pipeline_steps': [
                'text_cleaning',
                'tfidf_vectorization', 
                'svd_dimensionality_reduction',
                'svm_classification',
                'gridsearch_optimization'
            ],
            'features': {
                'stop_words_removal': True,
                'ngrams': '(1,2)',
                'min_df': 2,
                'max_df': 0.9,
                'svd_components': self._best_params.get('svd__n_components') if self._best_params else 100,
                'svm_C': self._best_params.get('clf__C') if self._best_params else 1.0
            }
        }
        
        return info
    
    def evaluate_training(self) -> Dict[str, Any]:
        """
        Trả về thông tin về quá trình training.
        
        Returns:
            Dict chứa metrics từ cross-validation
        """
        if not self._is_fitted:
            return {'error': 'Model chưa được training'}
        
        return {
            'best_cv_score': getattr(self.model, 'best_score_', None),
            'best_params': self._best_params,
            'cv_results_keys': list(self.model.cv_results_.keys()) if hasattr(self.model, 'cv_results_') else None
        }
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, float]:
        """
        Lấy top features quan trọng nhất từ SVM coefficients.
        
        Args:
            top_k: Số lượng top features
            
        Returns:
            Dict mapping feature -> importance score
        """
        if not self._is_fitted:
            return {}
        
        try:
            # Lấy best estimator từ GridSearchCV
            best_estimator = self.model.best_estimator_
            
            # Lấy feature names từ TF-IDF vectorizer
            tfidf_step = best_estimator.named_steps['tfidf']
            feature_names = tfidf_step.get_feature_names_out()
            
            # Lấy SVM coefficients
            svm_step = best_estimator.named_steps['clf']
            
            if hasattr(svm_step, 'coef_'):
                # Lấy absolute values của coefficients
                if len(svm_step.coef_.shape) == 1:
                    # Binary classification
                    coefficients = np.abs(svm_step.coef_)
                else:
                    # Multi-class: lấy max absolute coefficient across classes
                    coefficients = np.max(np.abs(svm_step.coef_), axis=0)
                
                # Transform từ SVD space về TF-IDF space
                svd_step = best_estimator.named_steps['svd']
                # Approximate feature importance từ SVD components
                feature_importance_scores = np.abs(svd_step.components_).T.dot(coefficients)
                
                # Lấy top k features
                top_indices = np.argsort(feature_importance_scores)[::-1][:top_k]
                
                feature_importance = {}
                for idx in top_indices:
                    if idx < len(feature_names):
                        feature_importance[feature_names[idx]] = feature_importance_scores[idx]
                
                return feature_importance
            
        except Exception as e:
            # Return empty dict nếu có lỗi
            pass
        
        return {}
    
    def predict_batch(self, texts: List[str], return_proba: bool = False) -> List[Dict[str, Any]]:
        """
        Dự đoán cho một batch texts (tiện ích thêm).
        
        Args:
            texts: Danh sách texts
            return_proba: Có return probabilities không
            
        Returns:
            List các prediction results
        """
        return [self.predict(text, return_proba=return_proba) for text in texts]