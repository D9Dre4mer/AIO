# 📊 Hướng Dẫn Chi Tiết Tất Cả Biểu Đồ - Advanced LightGBM Project

## 🎯 Tổng Quan

Dự án Advanced LightGBM tạo ra **25 biểu đồ PNG + 1 biểu đồ PDF** (tổng cộng 26 files) để phân tích và đánh giá toàn diện hiệu suất mô hình. Các biểu đồ được chia thành 4 nhóm chính:

1. **🔧 Optimization & Training** (6 biểu đồ)
2. **📈 Performance Evaluation** (6 biểu đồ) 
3. **🧠 Model Interpretability** (4 biểu đồ)
4. **📊 Comprehensive Analysis** (9 biểu đồ)

---

## 🔧 1. OPTIMIZATION & TRAINING PLOTS

### 1.1 `00_lightgbm_optimization.png`
**Mục đích**: Tìm số lượng estimators (n_estimators) tối ưu cho LightGBM bằng Cross-Validation

**Cách đọc**:
- **Trục X**: Số lượng estimators (n_estimators) - từ 50 đến 500
- **Trục Y**: Cross-Validation Accuracy
- **Đường cong xanh**: Hiệu suất CV accuracy theo số estimators
- **Điểm cao nhất**: Số estimators tối ưu
- **Grid**: Lưới để dễ đọc giá trị

**Ý nghĩa**:
- Điểm cao nhất → số estimators tốt nhất cho mô hình
- Nếu đường cong nằm ngang → có thể dừng sớm để tránh overfitting
- Nếu đường cong giảm → có thể bị overfitting với quá nhiều estimators
- Giúp cân bằng giữa hiệu suất và thời gian training

### 1.2 `01_lightgbm_performance_comparison.png` & `01_lightgbm_performance_comparison.pdf`
**Mục đích**: So sánh hiệu suất Validation vs Test của các dataset khác nhau

**Cách đọc**:
- **Trục X**: Tên các dataset (Raw, FE, DT, FE+DT)
- **Trục Y**: Độ chính xác (Accuracy) - từ 0.5 đến 1.05
- **Thanh xanh**: Validation Accuracy
- **Thanh đỏ**: Test Accuracy
- **Số trên thanh**: Giá trị chính xác cụ thể

**Ý nghĩa**:
- **Raw**: Dataset gốc
- **FE**: Dataset với Feature Engineering
- **DT**: Dataset với Decision Tree features
- **FE+DT**: Dataset kết hợp cả hai
- Thanh gần nhau → mô hình generalizes tốt
- Thanh xa nhau → có thể bị overfitting hoặc underfitting

**Lưu ý**: File có cả định dạng PNG (cho web) và PDF (cho in ấn)

### 1.3 `02_lightgbm_cv_scores.png`
**Mục đích**: Hiển thị kết quả Cross-Validation của các mô hình

**Cách đọc**:
- **Trục X**: Tên các mô hình (Raw, FE, DT, FE+DT)
- **Trục Y**: Điểm số Cross-Validation (CV Score)
- **Thanh cột**: Điểm trung bình của mỗi mô hình
- **Thanh lỗi**: Độ lệch chuẩn (standard deviation)
- **Số trên thanh**: Giá trị chính xác cụ thể

**Ý nghĩa**:
- Thanh cao → mô hình tốt hơn
- Thanh lỗi ngắn → mô hình ổn định qua các fold
- Thanh lỗi dài → mô hình không ổn định
- So sánh hiệu suất giữa các phiên bản dữ liệu khác nhau

### 1.4 `03_lightgbm_optimal_estimators.png`
**Mục đích**: So sánh số lượng estimators tối ưu của các dataset khác nhau

**Cách đọc**:
- **Trục X**: Tên các dataset (Raw, FE, DT, FE+DT)
- **Trục Y**: Số lượng estimators tối ưu (n_estimators)
- **Thanh cam**: Số estimators được chọn cho mỗi dataset
- **Số trên thanh**: Giá trị n_estimators cụ thể

**Ý nghĩa**:
- **Raw**: Số estimators tối ưu cho dataset gốc
- **FE**: Số estimators tối ưu cho dataset với Feature Engineering
- **DT**: Số estimators tối ưu cho dataset với Decision Tree features
- **FE+DT**: Số estimators tối ưu cho dataset kết hợp
- Số cao → dataset phức tạp hơn, cần nhiều estimators
- Số thấp → dataset đơn giản hơn, ít estimators đã đủ

### 1.5 `04_lightgbm_validation_vs_test.png`
**Mục đích**: Phân tích tương quan giữa Validation Accuracy và Test Accuracy

**Cách đọc**:
- **Trục X**: Validation Accuracy
- **Trục Y**: Test Accuracy
- **Điểm màu**: Mỗi dataset một màu (Raw=đỏ, FE=xanh, DT=xanh lá, FE+DT=cam)
- **Đường chéo đứt nét**: Perfect Correlation (tương quan hoàn hảo)
- **Nhãn**: Tên dataset gần mỗi điểm

**Ý nghĩa**:
- Điểm gần đường chéo → mô hình generalizes tốt
- Điểm xa đường chéo → có thể bị overfitting hoặc underfitting
- Điểm trên đường chéo → Test > Validation (tốt)
- Điểm dưới đường chéo → Test < Validation (có thể overfitting)
- Màu sắc giúp phân biệt các dataset khác nhau

### 1.6 `05_lightgbm_performance_heatmap.png`
**Mục đích**: Heatmap hiệu suất của các dataset khác nhau trên các metric khác nhau

**Cách đọc**:
- **Trục X**: Các metric (Validation Accuracy, Test Accuracy, CV Score)
- **Trục Y**: Các dataset (Raw, FE, DT, FE+DT)
- **Màu sắc**: Điểm số (xanh = cao, đỏ = thấp, vàng = trung bình)
- **Số trong ô**: Giá trị cụ thể (0.5-1.0)
- **Colorbar**: Thang màu từ 0.5 đến 1.0

**Ý nghĩa**:
- Màu xanh đậm → hiệu suất cao trên metric đó
- Màu đỏ đậm → hiệu suất thấp trên metric đó
- Hàng xanh → dataset tốt trên tất cả metric
- Cột xanh → metric ổn định qua các dataset
- So sánh hiệu suất tổng thể của từng dataset

---

## 📈 2. PERFORMANCE EVALUATION PLOTS

### 2.1 `02_roc_curve.png`
**Mục đích**: Đánh giá khả năng phân loại của mô hình

**Cách đọc**:
- **Trục X**: False Positive Rate (1 - Specificity)
- **Trục Y**: True Positive Rate (Sensitivity/Recall)
- **Đường chéo đứt nét**: Random classifier (AUC = 0.5)
- **Đường cong màu cam**: ROC curve của mô hình
- **AUC Score**: Hiển thị trong legend (thường > 0.8)
- **Grid**: Lưới để dễ đọc giá trị

**Ý nghĩa**:
- AUC = 1.0 → Perfect classifier
- AUC = 0.5 → Random classifier
- AUC > 0.8 → Good classifier
- Đường cong càng gần góc trên trái càng tốt
- Diện tích dưới đường cong càng lớn càng tốt

### 2.2 `03_precision_recall_curve.png`
**Mục đích**: Đánh giá hiệu suất khi có class imbalance

**Cách đọc**:
- **Trục X**: Recall (True Positive Rate)
- **Trục Y**: Precision (Positive Predictive Value)
- **Đường cong màu xanh**: Precision-Recall curve
- **AP Score**: Average Precision score (hiển thị trong legend)
- **Grid**: Lưới để dễ đọc giá trị
- **Baseline**: Đường ngang cho random classifier

**Ý nghĩa**:
- AP = 1.0 → Perfect classifier
- AP > 0.8 → Good classifier
- Đường cong càng gần góc trên phải càng tốt
- Quan trọng khi có class imbalance
- Diện tích dưới đường cong càng lớn càng tốt

### 2.3 `04_confusion_matrix.png`
**Mục đích**: Hiển thị chi tiết kết quả phân loại

**Cách đọc**:
- **Hàng**: True labels (Actual)
- **Cột**: Predicted labels (Predicted)
- **Ô trên trái**: True Negative (TN) - Dự đoán đúng negative
- **Ô trên phải**: False Positive (FP) - Dự đoán sai positive
- **Ô dưới trái**: False Negative (FN) - Dự đoán sai negative
- **Ô dưới phải**: True Positive (TP) - Dự đoán đúng positive
- **Số trong ô**: Số lượng mẫu
- **Màu sắc**: Càng đậm càng nhiều mẫu

**Ý nghĩa**:
- Đường chéo chính cao → mô hình tốt
- TP và TN cao → mô hình chính xác
- FP và FN thấp → ít lỗi
- Có thể tính accuracy = (TP+TN)/(TP+TN+FP+FN)

### 2.4 `05_prediction_distribution.png`
**Mục đích**: Phân tích phân bố của predictions

**Cách đọc**:
- **Trục X**: Giá trị prediction probability
- **Trục Y**: Tần suất (frequency)
- **Histogram xanh**: Class 0 (Negative)
- **Histogram đỏ**: Class 1 (Positive)

**Ý nghĩa**:
- Phân bố tách biệt → mô hình phân loại tốt
- Phân bố chồng lấp → mô hình khó phân biệt
- Threshold tối ưu ở điểm giao nhau

### 2.5 `06_metrics_comparison.png`
**Mục đích**: So sánh các metric khác nhau

**Cách đọc**:
- **Trục X**: Các mô hình
- **Trục Y**: Điểm số metric
- **Thanh khác màu**: Các metric khác nhau
- **Legend**: Giải thích màu sắc

**Ý nghĩa**:
- Thanh cao → mô hình tốt trên metric đó
- Cân bằng giữa các metric → mô hình ổn định
- Một metric quá cao/ thấp → cần điều chỉnh

### 2.6 `06_lightgbm_improvement_chart.png`
**Mục đích**: Hiển thị sự cải thiện hiệu suất của các dataset so với baseline (Raw dataset)

**Cách đọc**:
- **Trục X**: Các dataset (Raw, FE, DT, FE+DT)
- **Trục Y**: Phần trăm cải thiện (%) - có thể âm hoặc dương
- **Thanh xanh nhạt**: Validation Improvement  (%)
- **Thanh hồng nhạt**: Test Improvement (%)
- **Đường ngang đen**: Baseline (0% improvement)
- **Số trên thanh**: Giá trị cải thiện cụ thể (+/-%)

**Ý nghĩa**:
- Thanh dương → dataset tốt hơn baseline
- Thanh âm → dataset kém hơn baseline
- Raw dataset luôn có 0% improvement (baseline)
- FE, DT, FE+DT so sánh với Raw dataset
- Cải thiện cao → kỹ thuật hiệu quả

### 2.7 `07_radar_chart.png`
**Mục đích**: Hiển thị hiệu suất tổng thể của mô hình

**Cách đọc**:
- **Trục**: Các metric đánh giá (Accuracy, Precision, Recall, F1, AUC-ROC)
- **Đường cong**: Hiệu suất của mô hình
- **Vùng tô màu**: Diện tích hiệu suất tổng thể
- **Giá trị 0-1**: 0 = kém nhất, 1 = tốt nhất

**Ý nghĩa**:
- Hình tròn đều → mô hình cân bằng
- Hình lệch → mô hình mạnh/yếu ở một số metric
- Diện tích lớn → hiệu suất tổng thể cao

### 2.8 `07_lightgbm_radar_chart.png`
**Mục đích**: So sánh hiệu suất của các dataset khác nhau trên radar chart

**Cách đọc**:
- **Trục**: 4 metric (Validation Accuracy, Test Accuracy, CV Score, Normalized n_estimators)
- **Đường cong**: Mỗi dataset một màu (Raw=đỏ, FE=xanh, DT=xanh lá, FE+DT=cam)
- **Vùng tô màu**: Hiệu suất của từng dataset (alpha=0.25)
- **Legend**: Giải thích màu sắc và dataset
- **Thang đo**: 0-1 (n_estimators được normalize)

**Ý nghĩa**:
- Hình tròn lớn → dataset tốt trên tất cả metric
- Hình lệch → dataset mạnh/yếu ở một số metric
- Vùng chồng lấp → dataset tương đương
- Vùng riêng biệt → dataset khác biệt rõ ràng
- So sánh tổng thể hiệu suất của từng dataset

---

## 🧠 3. MODEL INTERPRETABILITY PLOTS

### 3.1 `08_feature_importance.png`
**Mục đích**: Xác định đặc trưng nào ảnh hưởng nhất đến kết quả dự đoán

**Cách đọc**:
- **Trục Y**: Tên các đặc trưng (features)
- **Trục X**: Mức độ quan trọng (Gain score)
- **Thanh dài**: Đặc trưng quan trọng hơn
- **Màu sắc**: Thường xanh dương, càng đậm càng quan trọng

**Ý nghĩa**:
- Top 5-10 đặc trưng → tập trung vào những đặc trưng này
- Đặc trưng có điểm thấp → có thể loại bỏ
- Phân bố đều → dữ liệu cân bằng

### 3.2 `09_shap_summary.png`
**Mục đích**: Giải thích cách mô hình đưa ra quyết định

**Cách đọc**:
- **Trục Y**: Các đặc trưng (features)
- **Trục X**: SHAP values (tác động đến dự đoán)
- **Màu đỏ**: Giá trị cao của đặc trưng
- **Màu xanh**: Giá trị thấp của đặc trưng
- **Vị trí**: Bên phải = tăng xác suất, bên trái = giảm xác suất

**Ý nghĩa**:
- Điểm đỏ bên phải → đặc trưng có giá trị cao → tăng xác suất dự đoán
- Điểm xanh bên trái → đặc trưng có giá trị thấp → giảm xác suất dự đoán
- Độ rộng → mức độ ảnh hưởng của đặc trưng

### 3.3 `10_shap_waterfall.png`
**Mục đích**: Giải thích chi tiết một prediction cụ thể

**Cách đọc**:
- **Trục Y**: Các đặc trưng
- **Trục X**: SHAP values
- **Thanh xanh**: Tăng xác suất dự đoán
- **Thanh đỏ**: Giảm xác suất dự đoán
- **Đường tích lũy**: Tổng tác động

**Ý nghĩa**:
- Thanh dài → đặc trưng có tác động lớn
- Màu xanh → đặc trưng ủng hộ prediction
- Màu đỏ → đặc trưng phản đối prediction

### 3.4 `08_lightgbm_trend_analysis.png`
**Mục đích**: Phân tích xu hướng hiệu suất và mối quan hệ giữa n_estimators và performance

**Cách đọc**:
- **Layout 2x1**: Hai subplot dọc
- **Subplot trên - Accuracy Trend**:
  - **Trục X**: Các dataset (Raw, FE, DT, FE+DT)
  - **Trục Y**: Accuracy
  - **Đường xanh**: Validation Accuracy (hình tròn)
  - **Đường đỏ**: Test Accuracy (hình vuông)
  - **Số trên điểm**: Giá trị accuracy cụ thể
- **Subplot dưới - n_estimators vs Performance**:
  - **Trục X**: Optimal n_estimators
  - **Trục Y**: Accuracy
  - **Điểm xanh**: Validation Accuracy
  - **Điểm đỏ**: Test Accuracy
  - **Nhãn**: Tên dataset gần mỗi điểm

**Ý nghĩa**:
- Xu hướng tăng → dataset cải thiện hiệu suất
- Xu hướng giảm → dataset có vấn đề
- Mối quan hệ n_estimators vs accuracy → tìm điểm tối ưu
- Điểm tách biệt → dataset khác biệt rõ ràng

---

## 📊 4. COMPREHENSIVE ANALYSIS PLOTS

### 4.1 `09_lightgbm_distribution_analysis.png`
**Mục đích**: Phân tích phân bố hiệu suất của các metric khác nhau qua tất cả dataset

**Cách đọc**:
- **Trục X**: Các metric (Validation Accuracy, Test Accuracy, CV Score)
- **Trục Y**: Accuracy Score
- **Box plot**: Phân bố của từng metric
- **Màu sắc**: Mỗi metric một màu (xanh nhạt, hồng nhạt, xanh lá nhạt)
- **Whiskers**: Phạm vi giá trị
- **Median**: Đường giữa hộp

**Ý nghĩa**:
- Box lớn → metric có biến động cao
- Box nhỏ → metric ổn định
- Median cao → metric có hiệu suất tốt
- Whiskers dài → có outlier hoặc biến động lớn
- So sánh độ ổn định của các metric khác nhau

### 4.2 `10_lightgbm_comprehensive_summary.png`
**Mục đích**: Tổng hợp tất cả thông tin quan trọng về hiệu suất của các dataset

**Cách đọc**:
- **Layout 2x2**: Bốn subplot trong một hình
- **Góc trên trái - Accuracy Comparison**:
  - **Thanh xanh nhạt**: Validation Accuracy
  - **Thanh hồng nhạt**: Test Accuracy
  - **Trục X**: Các dataset (Raw, FE, DT, FE+DT)
- **Góc trên phải - Cross-Validation Scores**:
  - **Thanh xanh lá nhạt**: CV Score
  - **Trục X**: Các dataset
- **Góc dưới trái - Optimal n_estimators**:
  - **Thanh cam**: Số estimators tối ưu
  - **Trục X**: Các dataset
- **Góc dưới phải - Generalization Performance**:
  - **Thanh tím**: Tỷ lệ Test/Validation
  - **Đường đỏ đứt nét**: Perfect Generalization (ratio = 1)

**Ý nghĩa**:
- Cái nhìn tổng quan về hiệu suất của từng dataset
- So sánh trực tiếp giữa các dataset
- Đánh giá khả năng generalization
- Thông tin đầy đủ trong một biểu đồ

### 4.3 `11_training_history.png`
**Mục đích**: Theo dõi quá trình học của mô hình

**Cách đọc**:
- **Layout 1x2**: Hai subplot cạnh nhau
- **Biểu đồ trái - Training History**:
  - **Trục X**: Number of Iterations
  - **Trục Y**: Binary Log Loss
  - **Đường xanh**: Training Loss
  - **Đường đỏ**: Validation Loss
  - **Đường xanh lá đứt nét**: Best Iteration (early stopping)
- **Biểu đồ phải - Feature Importance**:
  - **Trục Y**: Top 10 Most Important Features
  - **Trục X**: Importance Score
  - **Thanh ngang**: Mức độ quan trọng của từng feature

**Ý nghĩa**:
- Overfitting: Training loss giảm, validation loss tăng
- Underfitting: Cả hai đường đều cao và không giảm
- Tốt: Cả hai đường đều giảm và gần nhau
- Early stopping: Dừng tại best iteration để tránh overfitting

### 4.4 `11_raw_comprehensive_evaluation.png`
**Mục đích**: Đánh giá toàn diện mô hình với dữ liệu gốc

**Cách đọc**:
- **Layout 2x3**: 6 subplot trong một hình
- **Góc trên trái**: ROC Curve với AUC score
- **Góc trên giữa**: Precision-Recall Curve với AP score
- **Góc trên phải**: Confusion Matrix
- **Góc dưới trái**: Feature Importance (top features)
- **Góc dưới giữa**: Prediction Distribution
- **Góc dưới phải**: Performance Metrics (Accuracy, Precision, Recall, F1, AUC)

**Ý nghĩa**:
- Baseline performance của mô hình với dữ liệu gốc
- So sánh với các phiên bản cải tiến (FE, DT, FE+DT)
- Điểm khởi đầu để đánh giá hiệu quả của feature engineering

### 4.5 `11_fe_comprehensive_evaluation.png`
**Mục đích**: Đánh giá toàn diện mô hình với Feature Engineering

**Cách đọc**:
- **Layout 2x3**: Tương tự như raw evaluation
- **Các subplot**: ROC, Precision-Recall, Confusion Matrix, Feature Importance, Prediction Distribution, Metrics
- **So sánh**: Có thể so sánh trực tiếp với raw evaluation
- **Cải thiện**: Thường thấy improvement trong các metrics

**Ý nghĩa**:
- Tác động của Feature Engineering techniques
- Cải thiện performance so với raw data
- Hiệu quả của các kỹ thuật FE được áp dụng
- Validation cho việc feature engineering có hiệu quả

### 4.6 `11_dt_comprehensive_evaluation.png`
**Mục đích**: Đánh giá toàn diện mô hình với Decision Tree features

**Cách đọc**:
- **Layout 2x3**: Tương tự như các evaluation khác
- **Các subplot**: ROC, Precision-Recall, Confusion Matrix, Feature Importance, Prediction Distribution, Metrics
- **So sánh**: Có thể so sánh với raw và FE evaluation
- **DT Features**: Tập trung vào tác động của Decision Tree features

**Ý nghĩa**:
- Tác động của Decision Tree features được thêm vào
- Cải thiện performance so với raw và FE
- Hiệu quả của việc sử dụng Decision Tree để tạo features
- Validation cho DT approach có hiệu quả

### 4.7 `11_fe_dt_comprehensive_evaluation.png`
**Mục đích**: Đánh giá toàn diện mô hình kết hợp FE + DT

**Cách đọc**:
- **Layout 2x3**: Tương tự như các evaluation khác
- **Các subplot**: ROC, Precision-Recall, Confusion Matrix, Feature Importance, Prediction Distribution, Metrics
- **Tổng hợp**: Kết hợp cả Feature Engineering và Decision Tree features
- **Best Performance**: Thường có performance tốt nhất

**Ý nghĩa**:
- Hiệu quả của việc kết hợp FE + DT techniques
- Best performance có thể đạt được với tất cả techniques
- Tác động tổng hợp và tương tác giữa các techniques
- Validation cho approach tổng hợp có hiệu quả nhất

---

## 🔧 5. CÁCH SỬ DỤNG BIỂU ĐỒ

### 5.1 Phân Tích Tổng Quan
1. **Bắt đầu với**: `10_lightgbm_comprehensive_summary.png`
2. **Xem chi tiết**: `11_*_comprehensive_evaluation.png`
3. **So sánh**: `01_lightgbm_performance_comparison.png`

### 5.2 Tối Ưu Hóa Mô Hình
1. **Theo dõi quá trình**: `00_lightgbm_optimization.png`
2. **Tìm tham số tối ưu**: `03_lightgbm_optimal_estimators.png`
3. **Đánh giá stability**: `02_lightgbm_cv_scores.png`

### 5.3 Đánh Giá Hiệu Suất
1. **ROC Analysis**: `02_roc_curve.png`
2. **Precision-Recall**: `03_precision_recall_curve.png`
3. **Confusion Matrix**: `04_confusion_matrix.png`
4. **Radar Chart**: `07_radar_chart.png`

### 5.4 Hiểu Mô Hình
1. **Feature Importance**: `08_feature_importance.png`
2. **SHAP Analysis**: `09_shap_summary.png`
3. **Individual Predictions**: `10_shap_waterfall.png`

---

## ⚠️ 6. LƯU Ý QUAN TRỌNG

### 6.1 Fallback Plots
- Một số biểu đồ có thể sử dụng dữ liệu mẫu khi thiếu dữ liệu thực
- Điều này đảm bảo luôn có biểu đồ để tham khảo

### 6.2 Tiêu Đề Tiếng Việt
- Tất cả biểu đồ đều có tiêu đề và nhãn tiếng Việt
- Giúp dễ hiểu và sử dụng hơn

### 6.3 Chất Lượng Cao
- Tất cả biểu đồ được lưu với độ phân giải 300 DPI
- Phù hợp cho báo cáo và thuyết trình

### 6.4 Thứ Tự Đọc Biểu Đồ
1. **Comprehensive Summary** → Tổng quan
2. **Performance Comparison** → So sánh
3. **Individual Evaluations** → Chi tiết
4. **Optimization Plots** → Tối ưu hóa
5. **Interpretability Plots** → Giải thích

---

## 📁 7. VỊ TRÍ LƯU TRỮ

Tất cả biểu đồ được lưu trong thư mục:
```
results/run_20250920_014031/plots/
```

**Cấu trúc file**:
- `00_*` → Optimization plots
- `01_*` → Performance comparison (có cả PNG và PDF)
- `02_*` → ROC curves
- `03_*` → Precision-Recall curves
- `04_*` → Confusion matrices
- `05_*` → Distribution analysis
- `06_*` → Metrics comparison
- `07_*` → Radar charts
- `08_*` → Feature importance
- `09_*` → SHAP analysis
- `10_*` → Comprehensive summaries
- `11_*` → Training history & evaluations

**Định dạng file**:
- **PNG files (25)**: Cho hiển thị web và xem nhanh
- **PDF files (1)**: Cho in ấn và báo cáo chất lượng cao

---

## 🚀 8. KẾT LUẬN

**25 biểu đồ PNG + 1 biểu đồ PDF** (tổng cộng 26 files) này cung cấp cái nhìn toàn diện về:

### 📊 **Dữ Liệu**
- Chất lượng và đặc điểm
- Phân bố và xu hướng
- Tương quan giữa các features

### 🤖 **Mô Hình**
- Quá trình học và tối ưu hóa
- Hiệu suất trên các metric khác nhau
- So sánh giữa các phương pháp

### 🔍 **Đặc Trưng**
- Tầm quan trọng của từng feature
- Tác động đến predictions
- Hiệu quả của feature engineering

### ⚡ **Tối Ưu Hóa**
- Quá trình tìm tham số tối ưu
- Cải thiện qua thời gian
- So sánh các techniques

### 🎯 **Đánh Giá**
- Hiệu suất tổng thể
- Điểm mạnh và điểm yếu
- Khả năng generalizes

**Sử dụng kết hợp tất cả biểu đồ để có cái nhìn đầy đủ và đưa ra quyết định tối ưu cho dự án Machine Learning của bạn!** 🎯

---

## 📞 9. HỖ TRỢ

Nếu cần hỗ trợ thêm về cách đọc hoặc sử dụng các biểu đồ, vui lòng tham khảo:
- `PROJECT_SUMMARY.md` - Tổng quan dự án
- `PLOT_GUIDE.md` - Hướng dẫn cơ bản
- `README.md` - Tài liệu đầy đủ
- `QUICK_START.md` - Hướng dẫn nhanh

**Chúc bạn thành công với dự án Machine Learning!** 🚀
