# 📊 Hướng Dẫn Đọc Biểu Đồ - Advanced LightGBM Project

## Tổng Quan
Dự án Advanced LightGBM tạo ra **7 loại biểu đồ** khác nhau để phân tích và đánh giá hiệu suất mô hình. Mỗi biểu đồ cung cấp thông tin quan trọng về quá trình huấn luyện, hiệu suất và đặc trưng của mô hình.

---

## 1. 📈 Biểu Đồ Lịch Sử Tối Ưu Hóa (01_optimization_history.png)

### Mục đích
- Theo dõi quá trình tìm kiếm tham số tối ưu nhất cho mô hình
- Hiển thị sự cải thiện của mô hình qua các lần thử nghiệm

### Cách đọc
- **Trục X**: Số lần thử nghiệm (trial number)
- **Trục Y**: Giá trị mục tiêu (objective value) - thường là accuracy hoặc loss
- **Đường cong**: Cho thấy xu hướng cải thiện của mô hình
- **Xuống dốc**: Mô hình đang cải thiện (tốt)
- **Nằm ngang**: Mô hình đã đạt tối ưu

### Ý nghĩa
- Đường cong càng xuống dốc càng tốt
- Nếu đường cong nằm ngang sớm → có thể cần tăng số lần thử nghiệm
- Nếu đường cong không ổn định → có thể cần điều chỉnh phạm vi tham số

---

## 2. 🎯 Biểu Đồ Lịch Sử Huấn Luyện (02_training_history.png)

### Mục đích
- Theo dõi quá trình học của mô hình qua các epoch
- Phát hiện overfitting và underfitting

### Cách đọc
**Biểu đồ bên trái - Training/Validation Loss:**
- **Đường xanh**: Lỗi huấn luyện (training loss)
- **Đường đỏ**: Lỗi kiểm tra (validation loss)
- **Đường xanh lá**: Lần lặp tốt nhất (best iteration)

**Biểu đồ bên phải - Feature Importance:**
- **Thanh ngang**: Top 10 đặc trưng quan trọng nhất
- **Chiều dài thanh**: Mức độ quan trọng của đặc trưng

### Ý nghĩa
- **Overfitting**: Training loss giảm nhưng validation loss tăng
- **Underfitting**: Cả hai đường đều cao và không giảm
- **Tốt**: Cả hai đường đều giảm và gần nhau
- **Early stopping**: Dừng tại điểm validation loss thấp nhất

---

## 3. 🔍 Biểu Đồ Tầm Quan Trọng Đặc Trưng (03_feature_importance.png)

### Mục đích
- Xác định đặc trưng nào ảnh hưởng nhất đến kết quả dự đoán
- Hỗ trợ feature selection và feature engineering

### Cách đọc
- **Trục Y**: Tên các đặc trưng (features)
- **Trục X**: Mức độ quan trọng (Gain score)
- **Thanh dài**: Đặc trưng quan trọng hơn
- **Màu sắc**: Thường xanh dương, càng đậm càng quan trọng

### Ý nghĩa
- **Top 5-10 đặc trưng**: Có thể tập trung vào những đặc trưng này
- **Đặc trưng có điểm thấp**: Có thể loại bỏ để giảm noise
- **Phân bố đều**: Dữ liệu cân bằng, không có đặc trưng nào quá chi phối

---

## 4. 🧠 Biểu Đồ SHAP Summary (04_shap_summary.png)

### Mục đích
- Giải thích cách mô hình đưa ra quyết định
- Hiểu tác động của từng đặc trưng đến dự đoán

### Cách đọc
- **Trục Y**: Các đặc trưng (features)
- **Trục X**: SHAP values (tác động đến dự đoán)
- **Màu đỏ**: Giá trị cao của đặc trưng
- **Màu xanh**: Giá trị thấp của đặc trưng
- **Vị trí**: Bên phải = tăng xác suất dự đoán, bên trái = giảm xác suất

### Ý nghĩa
- **Điểm đỏ bên phải**: Đặc trưng có giá trị cao → tăng xác suất dự đoán
- **Điểm xanh bên trái**: Đặc trưng có giá trị thấp → giảm xác suất dự đoán
- **Độ rộng**: Cho thấy mức độ ảnh hưởng của đặc trưng

---

## 5. 🏆 Biểu Đồ So Sánh Ensemble (05_ensemble_comparison.png)

### Mục đích
- So sánh hiệu suất của các mô hình ensemble khác nhau
- Lựa chọn phương pháp ensemble tốt nhất

### Cách đọc
- **Trục X**: Tên các mô hình ensemble
- **Trục Y**: Độ chính xác (Accuracy)
- **Thanh cao**: Mô hình tốt hơn
- **Số trên thanh**: Giá trị chính xác cụ thể

### Ý nghĩa
- **Voting Hard**: Dự đoán dựa trên đa số phiếu
- **Voting Soft**: Dự đoán dựa trên xác suất trung bình
- **Stacking**: Sử dụng meta-learner để kết hợp
- **Weighted**: Kết hợp có trọng số dựa trên hiệu suất

---

## 6. 📊 Biểu Đồ Phân Tích Dữ Liệu (06_data_analysis.png)

### Mục đích
- Tổng quan về dữ liệu và đặc trưng
- Kiểm tra chất lượng dữ liệu

### Cách đọc
**Góc trên trái - Phân bố nhãn:**
- **Pie chart**: Tỷ lệ các class trong dữ liệu
- **Cân bằng**: Dữ liệu không bị lệch class

**Góc trên phải - Ma trận tương quan:**
- **Heatmap**: Mối quan hệ giữa các đặc trưng
- **Màu đỏ**: Tương quan dương mạnh
- **Màu xanh**: Tương quan âm mạnh
- **Màu trắng**: Không có tương quan

**Góc dưới trái - Top 5 đặc trưng quan trọng:**
- **Thanh ngang**: 5 đặc trưng quan trọng nhất
- **Chiều dài**: Mức độ quan trọng

**Góc dưới phải - Thống kê dữ liệu:**
- **Số mẫu**: Tổng số dữ liệu
- **Số đặc trưng**: Số lượng features
- **Tỷ lệ Class 1**: Phần trăm dữ liệu positive
- **Giá trị thiếu**: Số lượng missing values

---

## 7. 🎯 Biểu Đồ Radar Hiệu Suất (07_model_performance.png)

### Mục đích
- Hiển thị hiệu suất tổng thể của mô hình
- So sánh các metric khác nhau

### Cách đọc
- **Trục**: Các metric đánh giá (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- **Đường cong**: Hiệu suất của mô hình
- **Vùng tô màu**: Diện tích hiệu suất tổng thể
- **Giá trị 0-1**: 0 = kém nhất, 1 = tốt nhất

### Ý nghĩa
- **Hình tròn đều**: Mô hình cân bằng trên tất cả metric
- **Hình lệch**: Mô hình mạnh/yếu ở một số metric cụ thể
- **Diện tích lớn**: Hiệu suất tổng thể cao
- **Điểm yếu**: Các metric có giá trị thấp cần cải thiện

---

## 🔧 Cách Sử Dụng Biểu Đồ

### 1. Phân Tích Tổng Quan
- Bắt đầu với **Data Analysis** để hiểu dữ liệu
- Xem **Feature Importance** để biết đặc trưng quan trọng
- Kiểm tra **Training History** để đánh giá quá trình học

### 2. Tối Ưu Hóa Mô Hình
- Sử dụng **Optimization History** để điều chỉnh hyperparameters
- Dựa vào **SHAP Summary** để cải thiện feature engineering
- So sánh **Ensemble Methods** để chọn phương pháp tốt nhất

### 3. Đánh Giá Hiệu Suất
- **Model Performance Radar** cho cái nhìn tổng thể
- **Ensemble Comparison** để chọn mô hình tốt nhất
- Kết hợp tất cả biểu đồ để đưa ra quyết định cuối cùng

---

## ⚠️ Lưu Ý Quan Trọng

### Fallback Plots
- Một số biểu đồ có thể sử dụng dữ liệu mẫu (fallback) khi thiếu dữ liệu thực
- Điều này đảm bảo luôn có biểu đồ để tham khảo

### Tiêu Đề Tiếng Việt
- Tất cả biểu đồ đều có tiêu đề và nhãn tiếng Việt
- Giúp dễ hiểu và sử dụng hơn

### Chất Lượng Cao
- Tất cả biểu đồ được lưu với độ phân giải 300 DPI
- Phù hợp cho báo cáo và thuyết trình

---

## 📁 Vị Trí Lưu Trữ

Tất cả biểu đồ được lưu trong thư mục:
```
results/run_YYYYMMDD_HHMMSS/plots/
```

Với tên file:
- `01_optimization_history.png`
- `02_training_history.png`
- `03_feature_importance.png`
- `04_shap_summary.png`
- `05_ensemble_comparison.png`
- `06_data_analysis.png`
- `07_model_performance.png`

---

## 🚀 Kết Luận

7 loại biểu đồ này cung cấp cái nhìn toàn diện về:
- **Dữ liệu**: Chất lượng và đặc điểm
- **Mô hình**: Quá trình học và hiệu suất
- **Đặc trưng**: Tầm quan trọng và tác động
- **Tối ưu hóa**: Quá trình cải thiện tham số
- **So sánh**: Hiệu suất các phương pháp khác nhau

Sử dụng kết hợp tất cả biểu đồ để có cái nhìn đầy đủ và đưa ra quyết định tối ưu cho dự án Machine Learning của bạn! 🎯
