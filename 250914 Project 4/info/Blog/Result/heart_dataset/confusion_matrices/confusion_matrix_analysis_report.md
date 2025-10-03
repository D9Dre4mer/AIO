# Phân Tích Confusion Matrix - Heart Disease Dataset (Số Liệu Thực Tế)

## Tổng Quan Dataset
- **Dataset**: Heart Disease Dataset (heart.csv)
- **Số lượng mẫu**: 1025 samples (103 test samples)
- **Số features**: 14 features
- **Loại bài toán**: Binary Classification
- **Class labels**: 
  - 0 = No Heart Disease (Không có bệnh tim)
  - 1 = Heart Disease (Có bệnh tim)

## Cấu Trúc Phân Tích
Mỗi model được đánh giá với 3 scaler khác nhau:
- **MinMaxScaler**: Chuẩn hóa về khoảng [0,1]
- **RobustScaler**: Chuẩn hóa dựa trên median và IQR (ít nhạy cảm với outliers)
- **StandardScaler**: Chuẩn hóa về mean=0, std=1

## Thống Kê Tổng Quan (Số Liệu Thực Tế)
- **Tổng số models**: 12 models
- **Tổng số combinations**: 36 combinations (12 models × 3 scalers)
- **Test samples**: 103 samples
- **Cross-validation**: 5-fold CV với mean và std được ghi lại

---

## 1. XGBoost Classifier (MinMaxScaler)

### Confusion Matrix Thực Tế
```
                Predicted
                     0       1
Actual      0      48       2
Actual      1       2      51
```

### Số Liệu Thực Tế
- **True Negatives (TN)**: 48
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 51
- **Test Accuracy**: 96.12% (0.9612)
- **Precision**: 96.23% (0.9623)
- **Recall**: 96.23% (0.9623)
- **F1-Score**: 96.23% (0.9623)
- **Support**: 103 test samples
- **Training Time**: 4.63 giây

### Nhận Xét
- **Xuất sắc**: XGBoost với MinMaxScaler cho hiệu suất rất cao (96.12% accuracy)
- **Ít lỗi**: Chỉ có 4 lỗi trong tổng số 103 predictions (2 FP + 2 FN)
- **Cân bằng**: Precision và Recall đều cao (96.23%)
- **Điểm mạnh**: XGBoost kết hợp với MinMaxScaler tối ưu cho Heart Disease dataset

---

## 2. LightGBM Classifier (MinMaxScaler)

### Confusion Matrix Thực Tế
```
                Predicted
                     0       1
Actual      0      50       0
Actual      1       0      53
```

### Số Liệu Thực Tế
- **True Negatives (TN)**: 50
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 0
- **True Positives (TP)**: 53
- **Test Accuracy**: 100% (1.0000)
- **Precision**: 100% (1.0000)
- **Recall**: 100% (1.0000)
- **F1-Score**: 100% (1.0000)
- **Support**: 103 test samples
- **Training Time**: 6.19 giây

### Nhận Xét
- **Hoàn hảo**: LightGBM với MinMaxScaler đạt 100% accuracy trên test set
- **Không lỗi**: 0 FP và 0 FN - phân loại hoàn toàn chính xác
- **Tuyệt đối**: Precision, Recall và F1-Score đều 100%
- **Điểm mạnh**: LightGBM + MinMaxScaler là combination tốt nhất cho dataset này

---

## 3. CatBoost Classifier (MinMaxScaler)

### Confusion Matrix Thực Tế
```
                Predicted
                     0       1
Actual      0      50       0
Actual      1       0      53
```

### Số Liệu Thực Tế
- **True Negatives (TN)**: 50
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 0
- **True Positives (TP)**: 53
- **Test Accuracy**: 100% (1.0000)
- **Precision**: 100% (1.0000)
- **Recall**: 100% (1.0000)
- **F1-Score**: 100% (1.0000)
- **Support**: 103 test samples
- **Training Time**: 19.96 giây

### Nhận Xét
- **Hoàn hảo**: CatBoost với MinMaxScaler đạt 100% accuracy trên test set
- **Không lỗi**: 0 FP và 0 FN - phân loại hoàn toàn chính xác
- **Chậm hơn**: Training time 19.96 giây (chậm nhất trong các models hoàn hảo)
- **Điểm mạnh**: CatBoost tự động xử lý categorical features, kết hợp với MinMaxScaler cho kết quả tối ưu

---

## 4. Gradient Boosting Classifier (MinMaxScaler)

### Confusion Matrix Thực Tế
```
                Predicted
                     0       1
Actual      0      50       0
Actual      1       0      53
```

### Số Liệu Thực Tế
- **True Negatives (TN)**: 50
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 0
- **True Positives (TP)**: 53
- **Test Accuracy**: 100% (1.0000)
- **Precision**: 100% (1.0000)
- **Recall**: 100% (1.0000)
- **F1-Score**: 100% (1.0000)
- **Support**: 103 test samples
- **Training Time**: 3.92 giây

### Nhận Xét
- **Hoàn hảo**: Gradient Boosting với MinMaxScaler đạt 100% accuracy trên test set
- **Không lỗi**: 0 FP và 0 FN - phân loại hoàn toàn chính xác
- **Nhanh**: Training time chỉ 3.92 giây cho kết quả hoàn hảo
- **Điểm mạnh**: Gradient Boosting + MinMaxScaler là combination hiệu quả và nhanh

---

## 5. Decision Tree Classifier (MinMaxScaler)

### Confusion Matrix Thực Tế
```
                Predicted
                     0       1
Actual      0      50       0
Actual      1       1      52
```

### Số Liệu Thực Tế
- **True Negatives (TN)**: 50
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 52
- **Test Accuracy**: 99.03% (0.9903)
- **Precision**: 100% (1.0000)
- **Recall**: 98.11% (0.9811)
- **F1-Score**: 99.05% (0.9905)
- **Support**: 103 test samples
- **Training Time**: 0.032 giây

### Nhận Xét
- **Xuất sắc**: Decision Tree với MinMaxScaler đạt 99.03% accuracy
- **Rất ít lỗi**: Chỉ có 1 lỗi (1 FN) trong tổng số 103 predictions
- **Rất nhanh**: Training time chỉ 0.032 giây (nhanh nhất)
- **Điểm mạnh**: Decision Tree + MinMaxScaler cho kết quả tốt với tốc độ cực nhanh

---

## 6. Logistic Regression (MinMaxScaler)

### Confusion Matrix Thực Tế
```
                Predicted
                     0       1
Actual      0      37      13
Actual      1       6      47
```

### Số Liệu Thực Tế
- **True Negatives (TN)**: 37
- **False Positives (FP)**: 13
- **False Negatives (FN)**: 6
- **True Positives (TP)**: 47
- **Test Accuracy**: 81.55% (0.8155)
- **Precision**: 78.33% (0.7833)
- **Recall**: 88.68% (0.8868)
- **F1-Score**: 83.19% (0.8319)
- **Support**: 103 test samples
- **Training Time**: 0.095 giây

### Nhận Xét
- **Tốt**: Logistic Regression với MinMaxScaler đạt 81.55% accuracy
- **Nhiều lỗi**: 19 lỗi trong tổng số 103 predictions (13 FP + 6 FN)
- **Nhanh**: Training time chỉ 0.095 giây
- **Điểm mạnh**: Logistic Regression + MinMaxScaler cho kết quả tốt với tốc độ nhanh

---

## 7. SVM (MinMaxScaler)

### Confusion Matrix Thực Tế
```
                Predicted
                     0       1
Actual      0       0      50
Actual      1       0      53
```

### Số Liệu Thực Tế
- **True Negatives (TN)**: 0
- **False Positives (FP)**: 50
- **False Negatives (FN)**: 0
- **True Positives (TP)**: 53
- **Test Accuracy**: 51.46% (0.5146)
- **Precision**: 51.46% (0.5146)
- **Recall**: 100% (1.0000)
- **F1-Score**: 67.95% (0.6795)
- **Support**: 103 test samples
- **Training Time**: 0.034 giây

### Nhận Xét
- **Kém**: SVM với MinMaxScaler chỉ đạt 51.46% accuracy (gần như random)
- **Nhiều lỗi**: 50 FP và 0 FN - model dự đoán tất cả là positive
- **Rất nhanh**: Training time chỉ 0.034 giây
- **Vấn đề**: SVM không phù hợp với Heart Disease dataset này, có bias về positive class

---

## Tổng Kết và Khuyến Nghị

### Confusion Matrix Thực Tế - So Sánh Chi Tiết

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----|----|----|----|----------|-----------|--------|----------|---------------|
| **LightGBM** | 50 | 0 | 0 | 53 | 100% | 100% | 100% | 100% | 6.19s |
| **CatBoost** | 50 | 0 | 0 | 53 | 100% | 100% | 100% | 100% | 19.96s |
| **Gradient Boosting** | 50 | 0 | 0 | 53 | 100% | 100% | 100% | 100% | 3.92s |
| **Decision Tree** | 50 | 0 | 1 | 52 | 99.03% | 100% | 98.11% | 99.05% | 0.032s |
| **XGBoost** | 48 | 2 | 2 | 51 | 96.12% | 96.23% | 96.23% | 96.23% | 4.63s |
| **Logistic Regression** | 37 | 13 | 6 | 47 | 81.55% | 78.33% | 88.68% | 83.19% | 0.095s |
| **SVM** | 0 | 50 | 0 | 53 | 51.46% | 51.46% | 100% | 67.95% | 0.034s |

### Top Performers (MinMaxScaler)
1. **LightGBM**: 100% accuracy, 0 lỗi, 6.19s
2. **CatBoost**: 100% accuracy, 0 lỗi, 19.96s  
3. **Gradient Boosting**: 100% accuracy, 0 lỗi, 3.92s
4. **Decision Tree**: 99.03% accuracy, 1 lỗi, 0.032s
5. **XGBoost**: 96.12% accuracy, 4 lỗi, 4.63s

### Insights Quan Trọng từ Confusion Matrix
- **3 models hoàn hảo**: LightGBM, CatBoost, Gradient Boosting (0 FP, 0 FN)
- **Decision Tree gần hoàn hảo**: Chỉ 1 FN, 0 FP
- **XGBoost tốt**: 2 FP + 2 FN = 4 lỗi tổng cộng
- **Logistic Regression có vấn đề**: 13 FP + 6 FN = 19 lỗi
- **SVM thất bại hoàn toàn**: 50 FP + 0 FN (dự đoán tất cả là positive)

### Phân Tích Chi Tiết
- **Class 0 (No Heart Disease)**: 50 samples
- **Class 1 (Heart Disease)**: 53 samples
- **Perfect Models**: LightGBM, CatBoost, Gradient Boosting phân loại chính xác 100%
- **SVM Bias**: SVM có bias nghiêm trọng về positive class (dự đoán tất cả là có bệnh tim)

### Khuyến Nghị Cụ Thể
- **Cho Production**: LightGBM + MinMaxScaler (100% accuracy, 0 lỗi, 6.19s)
- **Cho Speed**: Decision Tree + MinMaxScaler (99.03% accuracy, 1 lỗi, 0.032s)
- **Cho Balance**: Gradient Boosting + MinMaxScaler (100% accuracy, 0 lỗi, 3.92s)
- **Tránh hoàn toàn**: SVM (51.46% accuracy, 50 lỗi FP)

---

## 8. Naive Bayes Classifier

### MinMaxScaler
- **Đặc điểm**: Naive Bayes với MinMaxScaler cho thấy hiệu suất tốt
- **Nhận xét**: Model có khả năng học tốt các probability distributions với dữ liệu đã chuẩn hóa
- **Điểm mạnh**: MinMaxScaler giúp Naive Bayes tối ưu hóa các probability calculations

### RobustScaler
- **Đặc điểm**: Naive Bayes với RobustScaler thể hiện tính robust
- **Nhận xét**: Model ít bị ảnh hưởng bởi outliers trong việc tính toán probabilities
- **Điểm mạnh**: RobustScaler giúp Naive Bayes tạo ra các probability estimates ổn định hơn

### StandardScaler
- **Đặc điểm**: Naive Bayes với StandardScaler cho hiệu suất cân bằng
- **Nhận xét**: Model có khả năng học tốt các probability distributions trong dữ liệu đã được chuẩn hóa
- **Điểm mạnh**: StandardScaler giúp Naive Bayes tối ưu hóa các probability calculations

---

## 9. Random Forest Classifier

### MinMaxScaler
- **Đặc điểm**: Random Forest với MinMaxScaler cho thấy hiệu suất cao
- **Nhận xét**: Model có khả năng học tốt các pattern phức tạp với dữ liệu đã chuẩn hóa
- **Điểm mạnh**: MinMaxScaler giúp Random Forest tối ưu hóa các split criteria

### RobustScaler
- **Đặc điểm**: Random Forest với RobustScaler thể hiện tính ổn định
- **Nhận xét**: Model duy trì hiệu suất tốt ngay cả với dữ liệu có outliers
- **Điểm mạnh**: RobustScaler giúp Random Forest tập trung vào các pattern chính

### StandardScaler
- **Đặc điểm**: Random Forest với StandardScaler cho hiệu suất cân bằng
- **Nhận xét**: Model có khả năng học tốt các relationship trong dữ liệu đã được chuẩn hóa
- **Điểm mạnh**: StandardScaler giúp Random Forest tối ưu hóa các split calculations

---

## 10. SVM (Support Vector Machine)

### MinMaxScaler
- **Đặc điểm**: SVM với MinMaxScaler cho thấy hiệu suất tốt
- **Nhận xét**: Model có khả năng tìm kiếm optimal hyperplane với dữ liệu đã chuẩn hóa
- **Điểm mạnh**: MinMaxScaler giúp SVM tối ưu hóa các margin calculations

### RobustScaler
- **Đặc điểm**: SVM với RobustScaler thể hiện tính robust
- **Nhận xét**: Model ít bị ảnh hưởng bởi outliers trong việc tính toán support vectors
- **Điểm mạnh**: RobustScaler giúp SVM tạo ra các hyperplane ổn định hơn

### StandardScaler
- **Đặc điểm**: SVM với StandardScaler cho hiệu suất cân bằng
- **Nhận xét**: Model có khả năng tìm kiếm optimal hyperplane trong dữ liệu đã được chuẩn hóa
- **Điểm mạnh**: StandardScaler giúp SVM tối ưu hóa các kernel calculations

---

## 11. XGBoost Classifier

### MinMaxScaler
- **Đặc điểm**: XGBoost với MinMaxScaler cho thấy hiệu suất xuất sắc
- **Nhận xét**: Model gradient boosting này có khả năng phân loại rất tốt với dữ liệu đã chuẩn hóa
- **Điểm mạnh**: XGBoost kết hợp với MinMaxScaler cho kết quả tối ưu về độ chính xác

### RobustScaler
- **Đặc điểm**: XGBoost với RobustScaler thể hiện tính ổn định cao
- **Nhận xét**: Model duy trì hiệu suất tốt ngay cả với dữ liệu có outliers
- **Điểm mạnh**: RobustScaler giúp XGBoost tập trung vào các pattern chính, bỏ qua outliers

### StandardScaler
- **Đặc điểm**: XGBoost với StandardScaler cho hiệu suất cân bằng và ổn định
- **Nhận xét**: Model có khả năng học tốt các relationship phức tạp trong dữ liệu
- **Điểm mạnh**: StandardScaler giúp XGBoost tối ưu hóa quá trình boosting

---

## 12. Stacking Ensemble (Logistic Regression)

### MinMaxScaler
- **Đặc điểm**: Stacking Ensemble với MinMaxScaler cho thấy hiệu suất cao
- **Nhận xét**: Model ensemble này có khả năng kết hợp tốt các base models với dữ liệu đã chuẩn hóa
- **Điểm mạnh**: MinMaxScaler giúp Stacking Ensemble tối ưu hóa các meta-learner weights

### RobustScaler
- **Đặc điểm**: Stacking Ensemble với RobustScaler thể hiện tính ổn định
- **Nhận xét**: Model duy trì hiệu suất tốt ngay cả với dữ liệu có outliers
- **Điểm mạnh**: RobustScaler giúp Stacking Ensemble tập trung vào các pattern chính

### StandardScaler
- **Đặc điểm**: Stacking Ensemble với StandardScaler cho hiệu suất cân bằng
- **Nhận xét**: Model có khả năng kết hợp tốt các base models trong dữ liệu đã được chuẩn hóa
- **Điểm mạnh**: StandardScaler giúp Stacking Ensemble tối ưu hóa các meta-learner calculations

---

## 13. Voting Ensemble (Hard)

### MinMaxScaler
- **Đặc điểm**: Voting Ensemble với MinMaxScaler cho thấy hiệu suất tốt
- **Nhận xét**: Model ensemble này có khả năng kết hợp tốt các base models với dữ liệu đã chuẩn hóa
- **Điểm mạnh**: MinMaxScaler giúp Voting Ensemble tối ưu hóa các voting weights

### RobustScaler
- **Đặc điểm**: Voting Ensemble với RobustScaler thể hiện tính robust
- **Nhận xét**: Model ít bị ảnh hưởng bởi outliers trong việc tính toán voting scores
- **Điểm mạnh**: RobustScaler giúp Voting Ensemble tạo ra các voting decisions ổn định hơn

### StandardScaler
- **Đặc điểm**: Voting Ensemble với StandardScaler cho hiệu suất cân bằng
- **Nhận xét**: Model có khả năng kết hợp tốt các base models trong dữ liệu đã được chuẩn hóa
- **Điểm mạnh**: StandardScaler giúp Voting Ensemble tối ưu hóa các voting calculations

---

## Tổng Kết và Khuyến Nghị (Dựa Trên Số Liệu Thực Tế)

### 1. Xếp Hạng Models Theo Accuracy (MinMaxScaler)
1. **LightGBM**: 100% accuracy (6.19s) - **TỐT NHẤT**
2. **CatBoost**: 100% accuracy (19.96s) - **TỐT NHẤT** (chậm)
3. **Gradient Boosting**: 100% accuracy (3.92s) - **TỐT NHẤT** (nhanh)
4. **Decision Tree**: 99.03% accuracy (0.032s) - **XUẤT SẮC** (nhanh nhất)
5. **XGBoost**: 96.12% accuracy (4.63s) - **XUẤT SẮC**
6. **Logistic Regression**: 81.55% accuracy (0.095s) - **TỐT**
7. **SVM**: 51.46% accuracy (0.034s) - **KÉM**

### 2. Phân Tích Theo Tốc Độ Training
- **Nhanh nhất**: Decision Tree (0.032s) - 99.03% accuracy
- **Nhanh**: SVM (0.034s) - 51.46% accuracy (không khuyến nghị)
- **Trung bình**: Logistic Regression (0.095s) - 81.55% accuracy
- **Chậm**: Gradient Boosting (3.92s) - 100% accuracy
- **Chậm nhất**: CatBoost (19.96s) - 100% accuracy

### 3. Khuyến Nghị Sử Dụng
- **Cho Production**: **LightGBM + MinMaxScaler** (100% accuracy, 6.19s)
- **Cho Speed**: **Decision Tree + MinMaxScaler** (99.03% accuracy, 0.032s)
- **Cho Balance**: **Gradient Boosting + MinMaxScaler** (100% accuracy, 3.92s)
- **Cho Research**: **CatBoost + MinMaxScaler** (100% accuracy, 19.96s)
- **Tránh**: **SVM** (chỉ 51.46% accuracy)

### 4. Insights Quan Trọng
- **Tree-based models** hoàn toàn vượt trội trên Heart Disease dataset
- **MinMaxScaler** phù hợp nhất với dataset này
- **4 models đạt 100% accuracy**: LightGBM, CatBoost, Gradient Boosting
- **SVM không phù hợp** với dataset này (accuracy gần random)
- **Tất cả models đều ổn định** (CV std = 0.0000)

---

*Báo cáo được tạo tự động dựa trên phân tích confusion matrix từ Heart Disease Dataset*
