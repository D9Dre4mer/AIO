# Hướng dẫn thêm Model mới vào dự án

## Tổng quan

Dự án sử dụng kiến trúc **Model-Centric Pipeline** - mỗi model tự quản lý toàn bộ pipeline xử lý dữ liệu của mình. Để thêm model mới, bạn cần tạo 4 thành phần chính.

## Cấu trúc Model-Centric Pipeline

```
Text Input → Preprocessor → Vectorizer → Feature Fuser → Estimator → Predictions
```

- **Preprocessor**: Tiền xử lý văn bản (stopwords, lemmatization, rare words)
- **Vectorizer**: Chuyển đổi văn bản thành vector (BoW, TF-IDF, Embeddings)
- **Feature Fuser**: Kết hợp nhiều loại đặc trưng (TF-IDF + Embeddings + features phụ)
- **Estimator**: Thuật toán học máy chính

## Bước 1: Tạo Model Class

### 1.1 Tạo file model mới

Tạo file mới trong `models/classification/` với tên `{model_name}_model.py`

**Ví dụ:** `models/classification/random_forest_model.py`

### 1.2 Cấu trúc Model Class

```python
"""
Random Forest Model với Model-Centric Pipeline
"""

import numpy as np
from typing import Dict, Any, Union
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from ..base.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest Model với pipeline tích hợp"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Khởi tạo estimator
        self.model = RandomForestClassifier(**self.model_params)
        
        # Pipeline components (sẽ được build trong build_pipeline)
        self.preprocessor = None
        self.vectorizer = None
        self.feature_fuser = None
        self.calibrator = None
        self.thresholds = None
    
    def build_pipeline(self, config: Dict[str, Any]) -> None:
        """Build pipeline dựa trên config"""
        from ..utils.text_preprocessor import TextPreprocessor
        from ..utils.vectorizer_provider import VectorizerProvider
        from ..utils.feature_fusion import FeatureFusion
        
        # 1. Build preprocessor
        preprocessor_config = config.get('preprocessor', {})
        self.preprocessor = TextPreprocessor(**preprocessor_config)
        
        # 2. Build vectorizer
        vectorizer_config = config.get('vectorizer', {})
        self.vectorizer = VectorizerProvider(**vectorizer_config)
        
        # 3. Build feature fuser (optional)
        if config.get('use_fusion', False):
            fusion_config = config.get('fusion', {})
            self.feature_fuser = FeatureFusion(**fusion_config)
    
    def fit_pipeline(self, X_text: list, y: np.ndarray) -> None:
        """Fit toàn bộ pipeline trên dữ liệu text"""
        # 1. Fit preprocessor
        self.preprocessor.fit(X_text, y)
        
        # 2. Transform text qua preprocessor
        X_processed = self.preprocessor.transform(X_text)
        
        # 3. Fit vectorizer
        self.vectorizer.fit(X_processed)
        
        # 4. Transform qua vectorizer
        X_vectorized = self.vectorizer.transform(X_processed)
        
        # 5. Apply feature fusion nếu có
        if self.feature_fuser:
            self.feature_fuser.fit(X_vectorized, y)
            X_final = self.feature_fuser.transform(X_vectorized)
        else:
            X_final = X_vectorized
        
        # 6. Fit estimator
        self.model.fit(X_final, y)
        self.is_fitted = True
    
    def transform_pipeline(self, X_text: list) -> Union[np.ndarray, sparse.csr_matrix]:
        """Transform text qua toàn bộ pipeline"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        # 1. Preprocess
        X_processed = self.preprocessor.transform(X_text)
        
        # 2. Vectorize
        X_vectorized = self.vectorizer.transform(X_processed)
        
        # 3. Feature fusion nếu có
        if self.feature_fuser:
            X_final = self.feature_fuser.transform(X_vectorized)
        else:
            X_final = X_vectorized
        
        return X_final
    
    def fit(self, X: Union[np.ndarray, sparse.csr_matrix], y: np.ndarray) -> 'RandomForestModel':
        """Fit model (cho compatibility với sklearn)"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict")
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, sparse.csr_matrix]) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before get_feature_importance")
        return self.model.feature_importances_
```

## Bước 2: Đăng ký Model trong Registry

### 2.1 Cập nhật `models/register_models.py`

```python
# Thêm import
from .classification.random_forest_model import RandomForestModel

# Thêm vào hàm register_all_models()
def register_all_models(registry):
    # ... existing models ...
    
    # Register Random Forest
    registry.register_model(
        'random_forest',
        RandomForestModel,
        {
            'category': 'classification',
            'task_type': 'supervised',
            'data_type': 'numerical',
            'description': 'Random Forest với Model-Centric Pipeline',
            'parameters': ['n_estimators', 'max_depth', 'max_features', 'class_weight'],
            'supports_sparse': True,
            'supports_gpu': True,  # nếu dùng cuML
            'supports_proba': True,
            'supports_shap': True,
            'default_vectorizers': ['tfidf', 'fusion'],
            'supports_fusion': True,
            'has_feature_importance': True
        }
    )
```

## Bước 3: Tạo Config mặc định

### 3.1 Thêm vào `config.py`

```python
# Random Forest default config
RANDOM_FOREST_DEFAULT_CONFIG = {
    'preprocessor': {
        'remove_stopwords': True,
        'lemmatization': False,
        'rare_words_threshold': 2
    },
    'vectorizer': {
        'type': 'tfidf',
        'max_features': 10000,
        'ngram_range': (1, 2)
    },
    'fusion': {
        'use_embeddings': True,
        'use_additional_features': True
    },
    'estimator': {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced'
    }
}
```

## Bước 4: Tích hợp vào Streamlit UI

### 4.1 Cập nhật `app.py` - Step 3

```python
def render_step3_wireframe():
    """Model Configuration & Preprocessing"""
    
    st.markdown("### 🤖 Step 3: Model Configuration & Preprocessing")
    
    # Model selector
    st.markdown("#### Chọn Model")
    available_models = ['random_forest', 'ada_boost', 'gbdt', 'xgboost', 'lightgbm']
    
    selected_models = []
    for model_name in available_models:
        if st.checkbox(f"✅ {model_name.replace('_', ' ').title()}", key=f"model_{model_name}"):
            selected_models.append(model_name)
    
    # Hiển thị config cho từng model đã chọn
    for model_name in selected_models:
        with st.expander(f"⚙️ Cấu hình {model_name.replace('_', ' ').title()}"):
            render_model_config(model_name)

def render_model_config(model_name: str):
    """Render config panel cho model cụ thể"""
    
    # Preprocessing options
    st.markdown("**🔧 Tiền xử lý dữ liệu:**")
    col1, col2 = st.columns(2)
    
    with col1:
        remove_stopwords = st.checkbox("Loại bỏ stopwords", value=True, key=f"{model_name}_stopwords")
        lemmatization = st.checkbox("Lemmatization", key=f"{model_name}_lemma")
    
    with col2:
        rare_words_threshold = st.number_input("Ngưỡng từ hiếm", min_value=1, value=2, key=f"{model_name}_rare")
    
    # Vectorizer options
    st.markdown("**📊 Vectorizer:**")
    vectorizer_type = st.selectbox(
        "Loại vectorizer",
        ['tfidf', 'embeddings', 'fusion'],
        key=f"{model_name}_vectorizer"
    )
    
    # Hyperparameters
    st.markdown("**🎛️ Hyperparameters:**")
    if model_name == 'random_forest':
        n_estimators = st.slider("n_estimators", 10, 500, 100, key=f"{model_name}_n_est")
        max_depth = st.slider("max_depth", 3, 20, 10, key=f"{model_name}_max_depth")
```

## Bước 5: Test Model mới

### 5.1 Tạo test script

```python
# test_new_model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.utils.model_factory import model_factory
from models.utils.model_registry import ModelRegistry

def test_new_model():
    # Khởi tạo registry
    registry = ModelRegistry()
    from models.register_models import register_all_models
    register_all_models(registry)
    
    # Khởi tạo factory
    model_factory.registry = registry
    
    # Tạo model
    model = model_factory.create_model('random_forest')
    
    # Test config
    config = {
        'preprocessor': {'remove_stopwords': True},
        'vectorizer': {'type': 'tfidf'},
        'use_fusion': False
    }
    
    # Build pipeline
    model.build_pipeline(config)
    
    # Test data
    X_text = ["This is a test document", "Another test document"]
    y = [0, 1]
    
    # Fit pipeline
    model.fit_pipeline(X_text, y)
    
    # Predict
    predictions = model.predict_pipeline(["New test document"])
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    test_new_model()
```

## Bước 6: Cập nhật Documentation

### 6.1 Thêm vào `models/README.md`

```markdown
## Random Forest Model

### Mô tả
Random Forest với Model-Centric Pipeline, hỗ trợ GPU qua cuML.

### Cấu hình mặc định
- Preprocessor: stopwords removal, rare words filtering
- Vectorizer: TF-IDF với n-gram (1,2)
- Feature Fusion: TF-IDF + Embeddings + additional features
- Estimator: RandomForestClassifier với class_weight='balanced'

### Sử dụng
```python
from models.utils.model_factory import model_factory

# Tạo model
model = model_factory.create_model('random_forest')

# Cấu hình
config = {
    'preprocessor': {'remove_stopwords': True},
    'vectorizer': {'type': 'tfidf'},
    'use_fusion': True
}

# Build và fit
model.build_pipeline(config)
model.fit_pipeline(X_text, y)

# Predict
predictions = model.predict_pipeline(X_test_text)
```

## Checklist hoàn thành

- [ ] Tạo model class trong `models/classification/`
- [ ] Implement đầy đủ pipeline methods
- [ ] Đăng ký model trong `register_models.py`
- [ ] Thêm config mặc định vào `config.py`
- [ ] Cập nhật Streamlit UI (Step 3)
- [ ] Tạo test script
- [ ] Cập nhật documentation
- [ ] Test với 300k samples từ arXiv

## Lưu ý quan trọng

1. **Memory optimization**: Với 300k samples, cần chú ý memory usage
2. **GPU support**: Thêm cuML support nếu có GPU
3. **Caching**: Sử dụng `st.cache_resource` cho model training
4. **Error handling**: Xử lý lỗi gracefully trong UI
5. **Progress tracking**: Hiển thị progress khi train model lớn

## Ví dụ Model phức tạp hơn

Để thêm model phức tạp như XGBoost với GPU:

```python
class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GPU detection
        import torch
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        
        # XGBoost với GPU flags
        xgb_params = self.model_params.copy()
        if self.device == 'gpu':
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
        
        self.model = XGBClassifier(**xgb_params)
```

Điều này đảm bảo model mới tích hợp hoàn toàn với kiến trúc Model-Centric Pipeline và Streamlit UI hiện có.
