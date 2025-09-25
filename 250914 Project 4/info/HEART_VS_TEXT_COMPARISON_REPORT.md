# 📊 Heart vs Text Dataset Comparison Report

## 🎯 Tổng Quan

Báo cáo này so sánh kết quả test Optuna giữa hai loại dữ liệu:
- **Heart Dataset** (Numeric): 1025 samples, 13 features
- **Text Dataset** (Spam): 1000 samples, TF-IDF vectorized to 1000 features

## 📈 Kết Quả So Sánh

### 🏆 **Performance Rankings**

#### **Heart Dataset (Numeric)**
```
🥇 Perfect Scores (100%):
- Random Forest: 100.0%
- XGBoost: 100.0% 
- LightGBM: 100.0%
- CatBoost: 100.0%
- Gradient Boosting: 100.0%

🥈 Excellent Scores (95%+):
- Decision Tree: 98.5%

🥉 Good Scores (80-95%):
- AdaBoost: 89.8%
- Logistic Regression: 81.0%
- Linear SVC: 80.5%
- KNN: 86.3%
- Naive Bayes: 82.9%
- SVM: 77.6%
```

#### **Text Dataset (TF-IDF)**
```
🥇 Excellent Scores (94%+):
- Linear SVC: 94.0%
- AdaBoost: 94.0%

🥈 Very Good Scores (90-94%):
- CatBoost: 93.5%
- LightGBM: 93.0%
- Gradient Boosting: 93.0%
- Random Forest: 91.0%
- Decision Tree: 91.5%
- Logistic Regression: 91.5%

🥉 Good Scores (80-90%):
- XGBoost: 89.0%
- SVM: 81.5%
- KNN: 82.5%
- Naive Bayes: 65.5%
```

## 🔍 **Phân Tích Chi Tiết**

### 1. **Tree-based Models Performance**

| Model | Heart (Numeric) | Text (TF-IDF) | Difference |
|-------|----------------|---------------|------------|
| Random Forest | 100.0% | 91.0% | -9.0% |
| XGBoost | 100.0% | 89.0% | -11.0% |
| LightGBM | 100.0% | 93.0% | -7.0% |
| CatBoost | 100.0% | 93.5% | -6.5% |
| Gradient Boosting | 100.0% | 93.0% | -7.0% |
| Decision Tree | 98.5% | 91.5% | -7.0% |

**📊 Nhận xét**: Tree-based models hoạt động tốt hơn với numeric data, có thể do:
- Numeric features có cấu trúc rõ ràng hơn cho splitting
- Text features sau TF-IDF có thể có noise hoặc sparsity

### 2. **Linear Models Performance**

| Model | Heart (Numeric) | Text (TF-IDF) | Difference |
|-------|----------------|---------------|------------|
| Logistic Regression | 81.0% | 91.5% | +10.5% |
| Linear SVC | 80.5% | 94.0% | +13.5% |
| SVM | 77.6% | 81.5% | +3.9% |

**📊 Nhận xét**: Linear models hoạt động tốt hơn với text data:
- TF-IDF features phù hợp với linear models
- Text classification thường có linear decision boundaries

### 3. **Ensemble Models Performance**

| Model | Heart (Numeric) | Text (TF-IDF) | Difference |
|-------|----------------|---------------|------------|
| AdaBoost | 89.8% | 94.0% | +4.2% |

**📊 Nhận xét**: AdaBoost hoạt động tốt với cả hai loại data, đặc biệt tốt với text.

### 4. **Other Models Performance**

| Model | Heart (Numeric) | Text (TF-IDF) | Difference |
|-------|----------------|---------------|------------|
| KNN | 86.3% | 82.5% | -3.8% |
| Naive Bayes | 82.9% | 65.5% | -17.4% |

**📊 Nhận xét**: 
- KNN: Tương đối ổn định
- Naive Bayes: Hoạt động kém với text data (có thể do TF-IDF không phù hợp với Gaussian assumption)

## ⚡ **Training Time Comparison**

### **Heart Dataset (Numeric)**
```
Fastest: Decision Tree (0.01s)
Slowest: LightGBM (9.67s)
Average: ~2.5s per model
```

### **Text Dataset (TF-IDF)**
```
Fastest: Linear SVC (0.01s)
Slowest: CatBoost (12.31s)
Average: ~3.5s per model
```

**📊 Nhận xét**: Text processing mất thời gian hơn do:
- TF-IDF vectorization overhead
- Higher dimensionality (1000 vs 13 features)
- Sparse matrix operations

## 🎯 **Key Insights**

### 1. **Data Type Suitability**
- **Numeric Data**: Tree-based models excel (100% accuracy)
- **Text Data**: Linear models excel (94% accuracy)

### 2. **Model Selection Strategy**
- **For Numeric**: Prioritize Random Forest, XGBoost, LightGBM
- **For Text**: Prioritize Linear SVC, AdaBoost, CatBoost

### 3. **Feature Engineering Impact**
- TF-IDF vectorization tạo ra 1000 features từ text
- Numeric data chỉ có 13 features nhưng hiệu quả hơn
- Quality > Quantity trong features

### 4. **GPU Acceleration**
- XGBoost, LightGBM, CatBoost đều sử dụng GPU thành công
- GPU acceleration hoạt động tốt với cả numeric và text data

## 🏆 **Best Practices Recommendations**

### **For Numeric Data**
1. **Primary**: Random Forest, XGBoost, LightGBM
2. **Secondary**: Gradient Boosting, Decision Tree
3. **Avoid**: SVM (77.6% accuracy)

### **For Text Data**
1. **Primary**: Linear SVC, AdaBoost
2. **Secondary**: CatBoost, LightGBM, Gradient Boosting
3. **Avoid**: Naive Bayes (65.5% accuracy)

### **Universal Models**
- **AdaBoost**: Hoạt động tốt với cả hai loại data
- **Logistic Regression**: Reliable baseline
- **KNN**: Stable performance across data types

## 📊 **Success Rate Summary**

| Dataset Type | Models Tested | Success Rate | Best Accuracy |
|--------------|---------------|--------------|---------------|
| **Heart (Numeric)** | 12/12 | 100% | 100.0% (Multiple) |
| **Text (TF-IDF)** | 12/12 | 100% | 94.0% (Linear SVC) |

## 🎉 **Conclusion**

✅ **Hệ thống hoạt động hoàn hảo** với cả hai loại dữ liệu:
- **100% success rate** cho tất cả models
- **Optuna optimization** hoạt động tốt
- **GPU acceleration** được sử dụng hiệu quả
- **Automatic data detection** và processing hoạt động chính xác

🚀 **Hệ thống đã sẵn sàng** để xử lý cả numeric và text data một cách tự động và hiệu quả!

---
*Report generated on: 2025-09-25*
*Test environment: PJ3.1 conda environment*
*Datasets: heart.csv (numeric) + 2cls_spam_text_cls.csv (text)*
