# Step 5: SHAP Visualization & Model Interpretation - Hướng dẫn sử dụng

## 📋 Tổng quan

Step 5 là bước cuối cùng trong pipeline machine learning, cung cấp các công cụ phân tích và giải thích mô hình:

- **🔍 SHAP Analysis**: Giải thích các dự đoán của mô hình
- **📊 Confusion Matrix**: Ma trận nhầm lẫn và đánh giá hiệu suất
- **📈 Model Comparison**: So sánh hiệu suất các mô hình

## 🎯 Yêu cầu tiên quyết

Trước khi sử dụng Step 5, bạn cần hoàn thành:

1. **Step 1**: Upload dataset
2. **Step 2**: Chọn cột input và label
3. **Step 3**: Cấu hình models và optimization
4. **Step 4**: Training models và lưu vào cache

## 🔍 SHAP Analysis

### Mục đích
SHAP (SHapley Additive exPlanations) giải thích các dự đoán của mô hình bằng cách tính toán đóng góp của từng feature.

### Cách sử dụng

#### 1. Kích hoạt SHAP Analysis
- ✅ Tick vào checkbox "Enable SHAP Analysis"
- Mặc định: Đã được kích hoạt

#### 2. Cấu hình tham số
- **Sample Size**: Số lượng samples để phân tích (100-10000)
  - Mặc định: 1000
  - Khuyến nghị: 500-2000 cho dataset lớn
  
- **Output Directory**: Thư mục lưu plots
  - Mặc định: "info/Result/"
  - Tự động tạo thư mục nếu chưa tồn tại

#### 3. Chọn Models
- **Available models**: Chỉ hiển thị tree-based models
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
  - AdaBoost
  - Gradient Boosting
- **Mặc định**: Random Forest, XGBoost, LightGBM

#### 4. Chọn Plot Types
- **Summary**: Tổng quan feature importance
- **Bar**: Bar chart của feature importance
- **Dependence**: Biểu đồ phụ thuộc giữa features
- **Waterfall**: Waterfall plot cho từng prediction
- **Mặc định**: Summary, Bar, Dependence

#### 5. Features cho Dependence Plots
- **Auto**: Tự động chọn features quan trọng nhất
- **Top 3**: 3 features quan trọng nhất
- **Custom**: Chọn features thủ công

#### 6. Generate SHAP Analysis
- Click nút "🚀 Generate SHAP Analysis"
- Hệ thống sẽ:
  - Load models từ cache
  - Tạo SHAP explainer
  - Tính toán SHAP values
  - Generate plots theo loại đã chọn

### Kết quả mong đợi
- **Summary Plot**: Hiển thị feature importance tổng quan
- **Bar Plot**: Bar chart của feature importance
- **Dependence Plots**: Biểu đồ phụ thuộc giữa features
- **Waterfall Plots**: Giải thích từng prediction cụ thể

## 📊 Confusion Matrix

### Mục đích
Ma trận nhầm lẫn giúp đánh giá hiệu suất phân loại của mô hình.

### Cách sử dụng

#### 1. Chọn Model
- **Select Model**: Chọn từ danh sách models đã train
- Chỉ hiển thị models có evaluation data

#### 2. Cấu hình Normalization
- **None**: Không chuẩn hóa (số lượng thực tế)
- **True**: Chuẩn hóa theo true labels (tỷ lệ %)
- **Pred**: Chuẩn hóa theo predicted labels
- **All**: Chuẩn hóa theo tổng số samples

#### 3. Dataset Split
- **Test**: Sử dụng test set (khuyến nghị)
- **Validation**: Sử dụng validation set
- **Train**: Sử dụng training set

#### 4. Classification Threshold
- **Slider**: 0.0 - 1.0
- Mặc định: 0.5
- Điều chỉnh threshold cho binary classification

#### 5. Show Percentages
- ✅ Tick để hiển thị tỷ lệ phần trăm
- Mặc định: Bật

#### 6. Generate Confusion Matrix
- Click nút "📊 Generate Confusion Matrix"
- Hệ thống sẽ:
  - Load model và evaluation data
  - Tính toán confusion matrix
  - Hiển thị matrix với normalization đã chọn

### Kết quả mong đợi
- **Confusion Matrix**: Ma trận nhầm lẫn
- **Classification Report**: Báo cáo chi tiết metrics
- **Accuracy, Precision, Recall, F1-Score**

## 📈 Model Comparison

### Mục đích
So sánh hiệu suất của tất cả models đã train để chọn model tốt nhất.

### Cách sử dụng

#### 1. Xem Training Configurations
- **Optuna optimization**: Số models được optimize
- **Voting ensemble**: Số models trong voting
- **Stacking ensemble**: Số base models trong stacking
- **SHAP analysis**: Số models được phân tích SHAP

#### 2. Load Model Metrics
- Click nút "📈 Load Model Metrics"
- Hệ thống sẽ load metrics từ Step 4 training results

#### 3. Performance Metrics Table
- **Model Name**: Tên model
- **Vectorization**: Phương pháp vectorization
- **F1 Score**: F1 score
- **Test Accuracy**: Độ chính xác trên test set
- **Precision**: Precision score
- **Recall**: Recall score
- **Training Time**: Thời gian training (giây)
- **Overfitting Level**: Mức độ overfitting
- **CV Mean Accuracy**: Cross-validation accuracy
- **CV Std Accuracy**: Standard deviation của CV

#### 4. Download Results
- **Download Results CSV**: Tải về file CSV chứa tất cả metrics
- File name: "comprehensive_evaluation_results.csv"

#### 5. Summary Report
- **Total Models**: Tổng số models
- **Avg Accuracy**: Độ chính xác trung bình
- **Avg Training Time**: Thời gian training trung bình

### Kết quả mong đợi
- **Performance Table**: Bảng so sánh metrics
- **Rankings**: Xếp hạng models theo accuracy/F1
- **Summary Statistics**: Thống kê tổng quan
- **CSV Export**: File dữ liệu để phân tích thêm

## ⚠️ Lưu ý quan trọng

### Cache Requirements
- **SHAP Analysis**: Cần models có SHAP sample data
- **Confusion Matrix**: Cần models có evaluation data
- **Model Comparison**: Cần models có metrics data

### Performance Considerations
- **SHAP Sample Size**: Càng lớn càng chính xác nhưng chậm hơn
- **Tree-based Models**: SHAP hoạt động tốt nhất với tree models
- **Memory Usage**: SHAP analysis có thể tốn nhiều RAM

### Error Handling
- **No Cached Models**: Cần hoàn thành Step 4 trước
- **Missing SHAP Sample**: Retrain model với SHAP sample
- **Missing Evaluation Data**: Retrain model với evaluation data

## 🎯 Best Practices

### SHAP Analysis
1. **Chọn Sample Size phù hợp**: 500-2000 cho dataset lớn
2. **Sử dụng tree-based models**: Random Forest, XGBoost, LightGBM
3. **Bắt đầu với Summary Plot**: Để hiểu tổng quan
4. **Sử dụng Dependence Plots**: Để hiểu mối quan hệ features

### Confusion Matrix
1. **Sử dụng Test Set**: Để đánh giá chính xác
2. **So sánh Normalization**: Thử các methods khác nhau
3. **Điều chỉnh Threshold**: Cho binary classification
4. **Xem Classification Report**: Để hiểu chi tiết metrics

### Model Comparison
1. **Load Metrics sau training**: Để có dữ liệu mới nhất
2. **So sánh nhiều metrics**: Không chỉ accuracy
3. **Xem Training Time**: Cân nhắc tốc độ vs hiệu suất
4. **Export CSV**: Để phân tích thêm với Excel/Python

## 🔧 Troubleshooting

### Lỗi thường gặp

#### "No cached models found"
- **Nguyên nhân**: Chưa hoàn thành Step 4
- **Giải pháp**: Quay lại Step 4 và training models

#### "No SHAP sample available"
- **Nguyên nhân**: Model được cache không có SHAP sample
- **Giải pháp**: Retrain model trong Step 4

#### "No tree-based models found"
- **Nguyên nhân**: Chỉ có linear models (Logistic Regression, SVM)
- **Giải pháp**: Train thêm tree-based models trong Step 3

#### "SHAP analysis failed"
- **Nguyên nhân**: Model không tương thích với SHAP
- **Giải pháp**: Thử với Random Forest hoặc XGBoost

### Performance Issues
- **SHAP chậm**: Giảm sample size
- **Memory error**: Giảm sample size hoặc sử dụng model nhỏ hơn
- **Plot không hiển thị**: Kiểm tra matplotlib backend

## 📚 Tài liệu tham khảo

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Confusion Matrix Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

## 🎉 Kết luận

Step 5 cung cấp các công cụ mạnh mẽ để:
- **Hiểu mô hình**: SHAP analysis giải thích predictions
- **Đánh giá hiệu suất**: Confusion matrix và metrics
- **Chọn model tốt nhất**: Model comparison và ranking

Sử dụng Step 5 để có cái nhìn sâu sắc về mô hình và đưa ra quyết định dựa trên dữ liệu!
