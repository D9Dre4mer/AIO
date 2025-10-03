# Phương Pháp Trực Quan Hóa Step 5 - SHAP Visualization & Model Interpretation

## Tổng Quan Step 5

**Step 5** là bước trực quan hóa và phân tích mô hình trong Comprehensive Machine Learning Platform. Đây là bước quan trọng giúp người dùng hiểu sâu về cách các mô hình ML đưa ra quyết định thông qua SHAP (SHapley Additive exPlanations) và các ma trận nhầm lẫn (Confusion Matrices).

---

## 1. Kiến Trúc Step 5 (`wizard_ui/steps/step5_shap_visualization.py`)

### 1.1 SHAPVisualizationStep Class

```python
class SHAPVisualizationStep:
    """Step 5: SHAP Visualization & Model Interpretation"""
    
    def __init__(self):
        """Initialize Step 5"""
        self.session_manager = SessionManager()
        self.confusion_matrix_cache = confusion_matrix_cache
    
    def render(self) -> None:
        """Render the complete Step 5 interface"""
        st.title("📊 Step 5: SHAP Visualization & Model Interpretation")
        
        # Create tabs for different visualization sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Model Selection", 
            "📊 SHAP Analysis", 
            "📈 Confusion Matrices", 
            "💾 Results Summary"
        ])
```

**Đặc điểm chính:**
- **4-Tab Interface**: Model Selection, SHAP Analysis, Confusion Matrices, Results Summary
- **Session Management**: Tích hợp với SessionManager để lưu trữ kết quả
- **Cache Integration**: Sử dụng confusion_matrix_cache để truy cập dữ liệu
- **Interactive UI**: Giao diện tương tác với Streamlit

### 1.2 Workflow Step 5

```
1. Model Selection → 2. SHAP Analysis → 3. Confusion Matrices → 4. Results Summary
     ↓                    ↓                    ↓                    ↓
Select Models    Generate SHAP Plots    Create Confusion     Download Results
Load Cache       Feature Importance     Matrix Plots         Comprehensive Report
Configure        Model Interpretation   Classification       Performance
                 Memory Management      Metrics              Comparison
```

---

## 2. Tab 1: Model Selection

### 2.1 Model Selection Interface

```python
def _render_model_selection(self, available_caches: List[Dict[str, Any]]):
    """Render model selection interface"""
    st.subheader("🎯 Select Models for Analysis")
    
    # Filter models with eval_predictions
    models_with_predictions = [cache for cache in available_caches if cache['has_eval_predictions']]
    
    # Model selection
    model_options = []
    for cache in models_with_predictions:
        model_name = f"{cache['model_key']} ({cache['dataset_id']})"
        model_options.append((model_name, cache))
    
    selected_model_names = st.multiselect(
        "Select models to analyze:",
        [option[0] for option in model_options],
        default=[option[0] for option in model_options[:3]],  # Select first 3 by default
        help="Choose which trained models to include in the analysis"
    )
```

**Tính năng Model Selection:**
- **Filtering**: Chỉ hiển thị models có eval_predictions
- **Multi-select**: Chọn nhiều models cùng lúc
- **Default Selection**: Tự động chọn 3 models đầu tiên
- **Model Info Display**: Hiển thị thông tin chi tiết về từng model

### 2.2 Model Information Display

```python
# Display selected models info
st.markdown("**📋 Selected Models:**")

for i, model in enumerate(selected_models, 1):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write(f"**{i}.** {model['model_key']}")
    
    with col2:
        st.write(f"Dataset: {model['dataset_id']}")
    
    with col3:
        st.write(f"Accuracy: {model.get('accuracy', 'N/A')}")
    
    with col4:
        has_shap = "✅" if model['has_shap_sample'] else "❌"
        st.write(f"SHAP: {has_shap}")
```

**Thông tin hiển thị:**
- **Model Key**: Tên và loại model
- **Dataset ID**: Dataset được sử dụng
- **Accuracy**: Độ chính xác của model
- **SHAP Availability**: Có sẵn SHAP sample hay không

### 2.3 SHAP Configuration

```python
# SHAP configuration
st.markdown("**⚙️ SHAP Configuration:**")

col1, col2 = st.columns(2)

with col1:
    enable_shap = st.checkbox(
        "Enable SHAP Analysis",
        value=SHAP_ENABLE,
        help="Generate SHAP visualizations for selected models"
    )
    
    sample_size = st.number_input(
        "Sample Size for SHAP",
        min_value=100,
        max_value=10000,
        value=SHAP_SAMPLE_SIZE,
        help="Number of samples to use for SHAP analysis"
    )

with col2:
    output_dir = st.text_input(
        "Output Directory",
        value=SHAP_OUTPUT_DIR,
        help="Directory to save SHAP plots"
    )
    
    plot_types = st.multiselect(
        "Plot Types",
        ["summary", "bar", "dependence", "waterfall"],
        default=["summary", "bar", "dependence"],
        help="Types of SHAP plots to generate"
    )
```

**SHAP Configuration Options:**
- **Enable/Disable**: Bật/tắt SHAP analysis
- **Sample Size**: Số lượng samples cho SHAP (100-10000)
- **Output Directory**: Thư mục lưu plots
- **Plot Types**: Các loại plots (summary, bar, dependence, waterfall)

---

## 3. Tab 2: SHAP Analysis

### 3.1 SHAP Analysis Interface

```python
def _render_shap_analysis(self):
    """Render SHAP analysis interface"""
    st.subheader("📊 SHAP Analysis")
    
    # SHAP analysis controls
    st.markdown("**🔧 SHAP Analysis Controls:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Generate SHAP Analysis", type="primary"):
            self._generate_shap_analysis(selected_models, shap_config)
    
    with col2:
        if st.button("📊 Preview SHAP Sample"):
            self._preview_shap_sample(selected_models[0])
```

### 3.2 SHAP Generation Process

```python
def _generate_shap_analysis(self, selected_models: List[Dict], shap_config: Dict):
    """Generate SHAP analysis for selected models"""
    with st.spinner("Generating SHAP analysis..."):
        try:
            shap_results = {}
            
            for model_info in selected_models:
                model_key = model_info['model_key']
                
                st.write(f"🔍 Analyzing {model_key}...")
                
                try:
                    # Load model and data from cache
                    from cache_manager import cache_manager
                    
                    cache_data = cache_manager.load_model_cache(
                        model_info['model_key'],
                        model_info['dataset_id'],
                        model_info['config_hash']
                    )
                    
                    model = cache_data['model']
                    shap_sample = cache_data.get('shap_sample')
                    
                    if shap_sample is None:
                        st.warning(f"⚠️ No SHAP sample available for {model_key}")
                        continue
                    
                    # Prepare sample data
                    sample_size = min(shap_config['sample_size'], len(shap_sample))
                    X_sample = shap_sample.iloc[:sample_size]
                    
                    # Generate comprehensive SHAP analysis
                    result = generate_comprehensive_shap_analysis(
                        model=model,
                        X_sample=X_sample,
                        feature_names=cache_data.get('feature_names'),
                        output_dir=shap_config['output_dir'],
                        model_name=model_key,
                        plot_types=shap_config['plot_types']
                    )
                    
                    shap_results[model_key] = result
                    st.success(f"✅ SHAP analysis completed for {model_key}")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing {model_key}: {str(e)}")
                    continue
```

**SHAP Generation Steps:**
1. **Load Model**: Tải model từ cache
2. **Load SHAP Sample**: Tải dữ liệu sample cho SHAP
3. **Prepare Data**: Chuẩn bị dữ liệu với sample size phù hợp
4. **Generate Analysis**: Tạo comprehensive SHAP analysis
5. **Save Results**: Lưu kết quả vào session

### 3.3 SHAP Visualization Functions (`visualization.py`)

#### A. SHAP Explainer Creation

```python
def create_shap_explainer(model, X_sample):
    """Create appropriate SHAP explainer for the model"""
    try:
        import shap
        
        # Extract underlying sklearn model
        underlying_model = extract_underlying_model(model)
        if underlying_model is None:
            print("ERROR: Cannot extract underlying sklearn model")
            return None
        
        model_name = model.__class__.__name__
        underlying_name = underlying_model.__class__.__name__
        
        print(f"Creating SHAP explainer for {model_name} -> {underlying_name}")
        
        # Determine model type and create appropriate explainer
        model_type_str = str(type(underlying_model)).lower()
        
        # Try TreeExplainer for tree-based models
        if any(keyword in model_type_str for keyword in ['randomforest', 'lightgbm', 'xgboost', 'gradientboosting', 'decisiontree']):
            try:
                print(f"Attempting TreeExplainer for {underlying_name}")
                
                # Memory safety check
                if len(X_sample) > 100:
                    print(f"Warning: Sample size {len(X_sample)} too large, using first 100 samples")
                    X_sample = X_sample[:100]
                
                explainer = shap.TreeExplainer(underlying_model)
                
                # Test explainer with small sample
                test_sample = X_sample[:5] if len(X_sample) >= 5 else X_sample
                try:
                    _ = explainer.shap_values(test_sample)
                    print(f"SUCCESS: Created TreeExplainer for {underlying_name}")
                    return explainer
                except Exception as test_error:
                    print(f"WARNING: TreeExplainer test failed for {underlying_name}: {test_error}")
                    del explainer
                    import gc
                    gc.collect()
                    raise test_error
                    
            except Exception as e:
                print(f"WARNING: TreeExplainer failed for {underlying_name}: {e}")
                import gc
                gc.collect()
                # Fallback to Explainer with predict_proba
                pass
        
        # Try Explainer with predict_proba for other models
        try:
            print(f"Attempting Explainer with predict_proba for {underlying_name}")
            
            # Memory safety check
            if len(X_sample) > 50:
                print(f"Warning: Sample size {len(X_sample)} too large, using first 50 samples")
                X_sample = X_sample[:50]
            
            def predict_proba_wrapper(X):
                return underlying_model.predict_proba(X)
            
            # Create explainer with smaller background
            background_sample = X_sample[:10] if len(X_sample) >= 10 else X_sample
            explainer = shap.Explainer(predict_proba_wrapper, background_sample)
            
            # Test explainer
            test_sample = X_sample[:3] if len(X_sample) >= 3 else X_sample
            try:
                _ = explainer(test_sample)
                print(f"SUCCESS: Created Explainer with predict_proba for {underlying_name}")
                return explainer
            except Exception as test_error:
                print(f"WARNING: Explainer test failed for {underlying_name}: {test_error}")
                del explainer
                import gc
                gc.collect()
                raise test_error
            
        except Exception as e:
            print(f"ERROR: All SHAP explainer methods failed for {underlying_name}: {e}")
            import gc
            gc.collect()
            return None
```

**SHAP Explainer Strategy:**
- **TreeExplainer**: Cho tree-based models (RandomForest, LightGBM, XGBoost, etc.)
- **Explainer**: Cho linear models với predict_proba wrapper
- **Memory Safety**: Giới hạn sample size để tránh memory issues
- **Fallback Strategy**: Thử nhiều phương pháp khác nhau
- **Garbage Collection**: Cleanup memory sau mỗi thử nghiệm

#### B. SHAP Summary Plot (Beeswarm)

```python
def generate_shap_summary_plot(explainer, X_sample, feature_names=None, max_display=20, save_path=None):
    """Generate SHAP summary plot (beeswarm plot)"""
    try:
        import shap
        
        # Get SHAP values using definitive method
        shap_values, shap_type = get_shap_values_definitive(explainer, X_sample)
        if shap_values is None:
            return None
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=max_display, show=False)
        
        ax.set_title("SHAP Summary Plot (Beeswarm)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: SHAP summary plot saved to: {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error generating SHAP summary plot: {e}")
        return None
```

**SHAP Summary Plot Features:**
- **Beeswarm Visualization**: Hiển thị distribution của SHAP values
- **Feature Importance**: Sắp xếp features theo importance
- **Color Coding**: Màu sắc thể hiện feature values
- **High Resolution**: 300 DPI cho chất lượng cao
- **Custom Titles**: Tiêu đề tùy chỉnh

#### C. SHAP Bar Plot

```python
def generate_shap_bar_plot(explainer, X_sample, feature_names=None, max_display=20, save_path=None):
    """Generate SHAP bar plot (mean absolute SHAP values)"""
    try:
        import shap
        
        # Get SHAP values using definitive method
        shap_values, shap_type = get_shap_values_definitive(explainer, X_sample)
        if shap_values is None:
            return None
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         max_display=max_display, plot_type="bar", show=False)
        
        ax.set_title("SHAP Bar Plot (Mean Absolute SHAP Values)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SUCCESS: SHAP bar plot saved to: {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error generating SHAP bar plot: {e}")
        return None
```

**SHAP Bar Plot Features:**
- **Mean Absolute Values**: Hiển thị mean absolute SHAP values
- **Feature Ranking**: Sắp xếp features theo importance
- **Clear Visualization**: Bar chart dễ đọc
- **Consistent Formatting**: Formatting nhất quán với summary plot

#### D. SHAP Dependence Plot

```python
def generate_shap_dependence_plot(explainer, X_sample, feature_names=None, 
                                 feature_index=0, interaction_index=None, save_path=None):
    """Generate SHAP dependence plot for a specific feature"""
    try:
        import shap
        
        # Get SHAP values using definitive method
        shap_values, shap_type = get_shap_values_definitive(explainer, X_sample)
        if shap_values is None:
            return None
        
        # Create dependence plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ensure feature_index is an integer
        if isinstance(feature_index, str):
            try:
                feature_index = int(feature_index)
            except ValueError:
                feature_index = 0
        
        try:
            if interaction_index is not None:
                shap.dependence_plot(feature_index, shap_values, X_sample, 
                                    interaction_index=interaction_index, show=False)
            else:
                shap.dependence_plot(feature_index, shap_values, X_sample, show=False)
            
            # Force set title for all axes after plot creation
            feature_name = feature_names[feature_index] if feature_names and len(feature_names) > feature_index else f"Feature {feature_index}"
            for ax in fig.get_axes():
                if not ax.get_title():  # Only set if title is empty
                    ax.set_title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, fontweight='bold')
            
            # Check if plot has meaningful content
            axes = fig.get_axes()
            has_content = False
            for ax in axes:
                if len(ax.get_lines()) > 0 or len(ax.collections) > 0:
                    # Check if collections have actual data points
                    for collection in ax.collections:
                        if hasattr(collection, 'get_offsets') and len(collection.get_offsets()) > 0:
                            has_content = True
                            break
                    if has_content:
                        break
            
            if not has_content:
                print("Warning: Dependence plot has no meaningful content, creating custom plot")
                # Clear and create custom dependence plot
                fig.clear()
                ax = fig.add_subplot(111)
                
                # Extract feature values and SHAP values
                feature_values = X_sample[:, feature_index]
                shap_feature_values = shap_values[:, feature_index]
                
                # Create scatter plot
                scatter = ax.scatter(feature_values, shap_feature_values, 
                                   alpha=0.7, s=50, c=shap_feature_values, 
                                   cmap='RdBu_r', edgecolors='black', linewidth=0.5)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='SHAP Value')
                
                # Set labels and title
                feature_name = feature_names[feature_index] if feature_names and len(feature_names) > feature_index else f"Feature {feature_index}"
                ax.set_xlabel(feature_name, fontsize=12)
                ax.set_ylabel(f"SHAP Value for {feature_name}", fontsize=12)
                ax.set_title(f"SHAP Dependence Plot: {feature_name}", fontsize=14, fontweight='bold')
                
                # Add grid
                ax.grid(True, alpha=0.3)
```

**SHAP Dependence Plot Features:**
- **Feature Interaction**: Hiển thị tương tác giữa features
- **Custom Fallback**: Tạo custom plot nếu SHAP plot không có nội dung
- **Color Mapping**: Sử dụng RdBu_r colormap
- **Grid Support**: Thêm grid để dễ đọc
- **Flexible Indexing**: Hỗ trợ cả string và integer feature index

### 3.4 Comprehensive SHAP Analysis

```python
def generate_comprehensive_shap_analysis(model, X_sample, feature_names=None, 
                                       model_name="Model", output_dir="info/Result/"):
    """Generate comprehensive SHAP analysis with multiple plots"""
    try:
        import shap
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create SHAP explainer
        explainer = create_shap_explainer(model, X_sample)
        if explainer is None:
            return None
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate plots
        plots = {}
        
        # 1. Summary plot (beeswarm)
        summary_path = os.path.join(output_dir, f"{model_name}_shap_summary_{timestamp}.png")
        summary_plot = generate_shap_summary_plot(explainer, X_sample, feature_names, 
                                                 save_path=summary_path)
        if summary_plot:
            plots['summary'] = summary_path
        
        # 2. Bar plot
        bar_path = os.path.join(output_dir, f"{model_name}_shap_bar_{timestamp}.png")
        bar_plot = generate_shap_bar_plot(explainer, X_sample, feature_names, 
                                        save_path=bar_path)
        if bar_plot:
            plots['bar'] = bar_path
        
        # 3. Dependence plots for top features
        if feature_names:
            # Get top 3 most important features
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            top_features = np.argsort(mean_shap)[-3:][::-1]
            
            for i, feature_idx in enumerate(top_features):
                dep_path = os.path.join(output_dir, f"{model_name}_shap_dependence_{feature_names[feature_idx]}_{timestamp}.png")
                dep_plot = generate_shap_dependence_plot(explainer, X_sample, feature_names, 
                                                      feature_index=feature_idx, save_path=dep_path)
                if dep_plot:
                    plots[f'dependence_{feature_names[feature_idx]}'] = dep_path
```

**Comprehensive Analysis Features:**
- **Multiple Plot Types**: Summary, Bar, Dependence plots
- **Top Features**: Tự động chọn top 3 features quan trọng nhất
- **Timestamp Naming**: File names với timestamp để tránh conflict
- **Directory Management**: Tự động tạo output directory
- **Error Handling**: Xử lý lỗi cho từng plot type

---

## 4. Tab 3: Confusion Matrices

### 4.1 Confusion Matrix Interface

```python
def _render_confusion_matrices(self):
    """Render confusion matrices interface"""
    st.subheader("📈 Confusion Matrices from Cache")
    
    # Confusion matrix configuration
    st.markdown("**⚙️ Confusion Matrix Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        normalize = st.selectbox(
            "Normalization",
            ["true", "pred", "all", None],
            index=0,
            help="How to normalize the confusion matrix"
        )
    
    with col2:
        save_plots = st.checkbox(
            "Save Plots",
            value=True,
            help="Save confusion matrix plots to disk"
        )
        
        show_metrics = st.checkbox(
            "Show Metrics",
            value=True,
            help="Display classification metrics"
        )
```

### 4.2 Confusion Matrix Generation (`confusion_matrix_cache.py`)

```python
class ConfusionMatrixCache:
    """Generates confusion matrices from cached evaluation predictions"""
    
    def __init__(self, cache_root_dir: str = "cache/models/"):
        """Initialize confusion matrix cache"""
        self.cache_root_dir = Path(cache_root_dir)
        self.cache_manager = cache_manager
    
    def generate_confusion_matrix_from_cache(self, model_key: str, dataset_id: str, 
                                           config_hash: str, 
                                           normalize: str = "true",
                                           labels_order: Optional[List[str]] = None,
                                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate confusion matrix from cached eval_predictions"""
        try:
            # Load cached data
            cache_data = self.cache_manager.load_model_cache(model_key, dataset_id, config_hash)
            
            if cache_data['eval_predictions'] is None:
                raise ValueError(f"No eval_predictions found in cache for {model_key}")
            
            eval_df = cache_data['eval_predictions']
            
            # Extract true labels and predictions
            # Handle different column naming conventions
            if 'y_true' in eval_df.columns:
                y_true = eval_df['y_true'].values
            elif 'true_labels' in eval_df.columns:
                y_true = eval_df['true_labels'].values
            else:
                raise ValueError("No true labels column found. Expected 'y_true' or 'true_labels'")
            
            y_pred = self._extract_predictions(eval_df)
            
            # Get label mapping
            label_mapping = cache_data.get('label_mapping', {})
            if not label_mapping:
                # Create default mapping using integer keys
                unique_labels = sorted(set(y_true) | set(y_pred))
                label_mapping = {int(i): f"Class_{i}" for i in unique_labels}
            else:
                # Convert string keys to integer keys if needed
                label_mapping = {int(k): v for k, v in label_mapping.items()}
            
            # Apply label ordering if provided
            if labels_order:
                label_mapping = {k: v for k, v in label_mapping.items() if v in labels_order}
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=list(label_mapping.keys()))
            
            # Normalize if requested
            if normalize:
                cm_normalized = self._normalize_confusion_matrix(cm, normalize)
            else:
                cm_normalized = cm
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
```

**Confusion Matrix Features:**
- **Cache Integration**: Tải dữ liệu từ model cache
- **Flexible Column Names**: Hỗ trợ nhiều tên cột khác nhau
- **Label Mapping**: Mapping labels từ integer sang string
- **Normalization Options**: true, pred, all, hoặc None
- **High-Quality Plots**: 10x8 figure size với high DPI

### 4.3 Confusion Matrix Visualization

```python
# Create plot
fig, ax = plt.subplots(figsize=(10, 8))

# Create annotations with raw + normalized values
annotations = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        raw = cm[i, j]
        if normalize:
            norm = cm_normalized[i, j]
            annotations[i, j] = f"{raw}\n({norm:.2%})"
        else:
            annotations[i, j] = f"{raw}"

# Create heatmap
sns.heatmap(cm_normalized, annot=annotations, fmt='', cmap='Blues',
            xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
            yticklabels=[label_mapping[i] for i in range(len(label_mapping))],
            ax=ax)

# Set title and labels
title = f"Confusion Matrix: {model_key}"
if normalize:
    title += f" (Normalized: {normalize})"
ax.set_title(title, fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)

plt.tight_layout()
```

**Visualization Features:**
- **Dual Annotations**: Hiển thị cả raw counts và normalized percentages
- **Color Mapping**: Sử dụng Blues colormap
- **Custom Labels**: Labels từ label mapping
- **Dynamic Titles**: Tiêu đề với normalization info
- **Professional Formatting**: Font sizes và weights nhất quán

### 4.4 Classification Metrics

```python
def _compute_classification_metrics(self, y_true, y_pred, label_mapping):
    """Compute comprehensive classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    class_metrics = {}
    for class_id, class_name in label_mapping.items():
        # Convert to binary for this class
        y_true_binary = (y_true == class_id).astype(int)
        y_pred_binary = (y_pred == class_id).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:  # Only if class exists in true labels
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics
    }
```

**Metrics Features:**
- **Overall Metrics**: Accuracy, Macro F1, Weighted F1
- **Per-Class Metrics**: Precision, Recall, F1 cho từng class
- **Binary Conversion**: Chuyển đổi sang binary classification cho từng class
- **Zero Division Handling**: Xử lý zero division errors
- **Comprehensive Coverage**: Bao phủm tất cả các metrics quan trọng

---

## 5. Tab 4: Results Summary

### 5.1 Results Summary Interface

```python
def _render_results_summary(self):
    """Render results summary interface"""
    st.subheader("💾 Results Summary")
    
    # Get all results
    shap_results = self.session_manager.get_step_data(5).get('shap_results', {})
    cm_results = self.session_manager.get_step_data(5).get('confusion_matrix_results', {})
    selected_models = self.session_manager.get_step_data(5).get('selected_models', [])
    
    # Summary statistics
    st.markdown("**📊 Analysis Summary:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models Analyzed", len(selected_models))
    
    with col2:
        st.metric("SHAP Results", len(shap_results))
    
    with col3:
        st.metric("Confusion Matrices", len(cm_results))
    
    with col4:
        total_plots = len(shap_results) + len(cm_results)
        st.metric("Total Plots", total_plots)
```

### 5.2 Model Performance Comparison

```python
# Model performance comparison
if cm_results:
    st.markdown("**🏆 Model Performance Comparison:**")
    
    performance_data = []
    for model_key, result in cm_results.items():
        metrics = result['metrics']
        performance_data.append({
            'Model': model_key,
            'Accuracy': metrics['accuracy'],
            'Macro F1': metrics['macro_f1'],
            'Weighted F1': metrics['weighted_f1']
        })
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('Accuracy', ascending=False)
        
        st.dataframe(performance_df, width='stretch')
        
        # Best model
        best_model = performance_df.iloc[0]
        st.success(f"🏆 Best performing model: **{best_model['Model']}** "
                  f"(Accuracy: {best_model['Accuracy']:.3f})")
```

**Performance Comparison Features:**
- **Comprehensive Metrics**: Accuracy, Macro F1, Weighted F1
- **Sorted Results**: Sắp xếp theo Accuracy giảm dần
- **Best Model Highlight**: Highlight model tốt nhất
- **Interactive Table**: DataFrame với Streamlit styling

### 5.3 Download Options

```python
# Download options
st.markdown("**💾 Download Options:**")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Download SHAP Results"):
        self._download_shap_results(shap_results)

with col2:
    if st.button("📈 Download Confusion Matrices"):
        self._download_confusion_matrices(cm_results)

# Generate comprehensive report
if st.button("📋 Generate Comprehensive Report", type="primary"):
    self._generate_comprehensive_report(shap_results, cm_results, selected_models)
```

**Download Features:**
- **SHAP Results**: Download SHAP analysis results
- **Confusion Matrices**: Download confusion matrix results
- **Comprehensive Report**: Generate và download comprehensive report
- **CSV Format**: Tất cả downloads đều ở format CSV
- **Timestamp Naming**: File names với timestamp

---

## 6. SHAP Cache Management (`shap_cache_manager.py`)

### 6.1 SHAPCacheManager Class

```python
class SHAPCacheManager:
    """Manages SHAP explainer and values caching with memory leak protection"""
    
    def __init__(self, cache_dir: str = "cache/shap/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory leak protection
        self._lock = threading.Lock()
        self._active_explainers = {}  # Track active explainers
        self._max_memory_mb = 2000  # Max memory usage in MB
        self._max_sample_size = 500  # Max sample size to prevent memory issues
        
        # Cleanup old cache files on init
        self._cleanup_old_cache()
```

**Memory Management Features:**
- **Threading Lock**: Thread-safe operations
- **Active Explainer Tracking**: Theo dõi active explainers
- **Memory Limits**: Giới hạn memory usage (2GB)
- **Sample Size Limits**: Giới hạn sample size (500)
- **Automatic Cleanup**: Tự động cleanup cache files cũ

### 6.2 Memory Safety Operations

```python
@contextmanager
def _memory_safe_operation(self):
    """Context manager for memory-safe operations"""
    try:
        # Force garbage collection before checking memory
        gc.collect()
        
        # Check memory before operation
        if not self._check_memory_usage():
            print("Warning: High memory usage detected, attempting cleanup...")
            # Try aggressive cleanup
            gc.collect()
            import time
            time.sleep(0.1)  # Brief pause for cleanup
            
            # Check again after cleanup
            if not self._check_memory_usage():
                print("Warning: Still high memory usage after cleanup, skipping SHAP operation")
                yield False
                return
            else:
                print("INFO: Memory usage reduced after cleanup, proceeding with SHAP operation")
        
        yield True
    finally:
        # Aggressive cleanup after operation
        gc.collect()
        import time
        time.sleep(0.05)  # Brief pause for cleanup
```

**Memory Safety Features:**
- **Context Manager**: Context manager cho memory-safe operations
- **Pre-operation Check**: Kiểm tra memory trước khi thực hiện
- **Aggressive Cleanup**: Cleanup aggressive khi memory cao
- **Post-operation Cleanup**: Cleanup sau khi thực hiện
- **Graceful Degradation**: Skip operations nếu memory quá cao

### 6.3 Cache Key Generation

```python
def generate_shap_cache_key(self, model, sample_data, model_name=""):
    """Generate cache key for SHAP explainer and values with safety checks"""
    try:
        # Create hash from model and sample data (with safety checks)
        try:
            model_params = model.get_params() if hasattr(model, 'get_params') else {}
        except Exception:
            model_params = {}
        
        model_str = str(type(model)) + str(model_params)
        
        # Create hash from sample data (with safety checks)
        try:
            if hasattr(sample_data, 'shape'):
                # For numpy arrays or dataframes
                sample_str = str(sample_data.shape) + str(sample_data.dtype)
            else:
                # For other data types
                sample_str = str(type(sample_data)) + str(len(sample_data))
        except Exception:
            sample_str = "unknown_sample"
        
        # Combine and hash
        combined_str = model_str + sample_str + model_name
        cache_key = hashlib.md5(combined_str.encode()).hexdigest()[:16]
        
        return cache_key
        
    except Exception as e:
        print(f"Error generating cache key: {e}")
        return f"error_{hashlib.md5(str(e).encode()).hexdigest()[:8]}"
```

**Cache Key Features:**
- **Model Fingerprinting**: Hash từ model type và parameters
- **Sample Fingerprinting**: Hash từ sample data shape và type
- **Safety Checks**: Xử lý exceptions trong quá trình hash
- **Consistent Length**: Cache keys có độ dài nhất quán (16 chars)
- **Error Handling**: Fallback cache key nếu có lỗi

---

## 7. Kết Luận

### 7.1 Điểm Mạnh của Step 5 Visualization

**A. SHAP Analysis:**
- ✅ **Multiple Explainer Types**: TreeExplainer, Explainer với predict_proba
- ✅ **Memory Safety**: Memory management và garbage collection
- ✅ **Multiple Plot Types**: Summary, Bar, Dependence plots
- ✅ **Feature Importance**: Tự động chọn top features
- ✅ **High-Quality Output**: 300 DPI plots với professional formatting

**B. Confusion Matrices:**
- ✅ **Cache Integration**: Tải dữ liệu từ model cache
- ✅ **Flexible Normalization**: true, pred, all, hoặc None
- ✅ **Comprehensive Metrics**: Accuracy, F1, Precision, Recall
- ✅ **Per-Class Analysis**: Metrics cho từng class
- ✅ **Professional Visualization**: Seaborn heatmaps với annotations

**C. User Experience:**
- ✅ **4-Tab Interface**: Organized workflow
- ✅ **Interactive Selection**: Multi-select models
- ✅ **Real-time Feedback**: Progress indicators và status updates
- ✅ **Download Options**: CSV downloads với timestamp naming
- ✅ **Comprehensive Reports**: Detailed analysis reports

**D. Technical Excellence:**
- ✅ **Memory Management**: Advanced memory leak protection
- ✅ **Error Handling**: Comprehensive error handling
- ✅ **Cache System**: Intelligent caching với compatibility scoring
- ✅ **Thread Safety**: Thread-safe operations
- ✅ **Performance Optimization**: Memory limits và cleanup strategies

### 7.2 Tính Năng Đặc Biệt

1. **Intelligent Explainer Selection**: Tự động chọn explainer phù hợp cho từng model type
2. **Memory Leak Protection**: Advanced memory management với garbage collection
3. **Fallback Strategies**: Multiple fallback strategies cho SHAP explainers
4. **Custom Plot Generation**: Tạo custom plots khi SHAP plots không có nội dung
5. **Comprehensive Metrics**: Detailed classification metrics với per-class analysis
6. **Professional Visualization**: High-quality plots với consistent formatting
7. **Cache Integration**: Seamless integration với model cache system
8. **Download System**: Multiple download options với timestamp naming

### 7.3 Best Practices Được Áp Dụng

- **Memory Safety**: Context managers và garbage collection
- **Error Handling**: Comprehensive try-catch blocks
- **User Experience**: Progress indicators và real-time feedback
- **Code Organization**: Modular design với clear separation of concerns
- **Performance Optimization**: Memory limits và cleanup strategies
- **Professional Output**: High-quality visualizations với consistent formatting
- **Cache Management**: Intelligent caching với compatibility scoring
- **Thread Safety**: Thread-safe operations với locking mechanisms

Step 5 thể hiện một cách tiếp cận chuyên nghiệp và toàn diện trong việc tạo ra các visualizations có ý nghĩa và dễ hiểu cho việc giải thích mô hình ML, giúp người dùng hiểu sâu về cách các mô hình đưa ra quyết định.

---

*Step 5 Visualization Methodology Documentation*
*Comprehensive Machine Learning Platform*
*Cập nhật: 2025-01-27*
