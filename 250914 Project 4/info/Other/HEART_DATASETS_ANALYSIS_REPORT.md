# Heart Datasets Analysis Report

## Tổng quan
Phân tích so sánh hai bộ dữ liệu Heart Disease để tìm hiểu mối quan hệ và cách chuyển đổi giữa chúng.

## Datasets được phân tích
1. **Heart_disease_cleveland_new.csv**: 303 rows × 14 columns
2. **heart.csv**: 1025 rows × 14 columns

## Kết quả phân tích chính

### 1. Cấu trúc Dataset
- **Cùng cấu trúc cột**: Cả hai đều có cùng 14 cột với cùng tên và kiểu dữ liệu
- **Không có missing values**: Heart_disease_cleveland_new.csv hoàn toàn sạch
- **Missing values trong heart.csv**: 
  - ca=4: 18 cases (1.8%) - không hợp lệ về mặt y học
  - thal=0: 7 cases (0.7%) - không hợp lệ về mặt y học

### 2. Phân phối Target
- **Cleveland**: 54.1% class 0, 45.9% class 1 (164 vs 139)
- **Heart**: 48.7% class 0, 51.3% class 1 (499 vs 526)
- Heart dataset có phân phối cân bằng hơn

### 3. Phát hiện quan trọng về chuyển đổi

**Heart dataset KHÔNG phải là subset của Cleveland dataset:**
- Common rows: 0 (không có dòng nào giống nhau hoàn toàn)
- Cleveland duplicates: 0
- Heart duplicates: 723 (có rất nhiều duplicate)
- Heart unique rows: 302 (gần bằng Cleveland: 303)

### 4. Data Augmentation Pattern
- **Duplication factor**: 3.39x (1025 total / 302 unique)
- **TẤT CẢ rows đều bị duplicate** (không có row nào xuất hiện đúng 1 lần)
- Most frequent duplicates xuất hiện 8 lần

### 5. Khác biệt trong phạm vi dữ liệu
- **ca column**: Cleveland (0-3), Heart (0-4) - Heart có thêm giá trị 4
- **thal column**: Cleveland (1-3), Heart (0-3) - Heart có thêm giá trị 0
- **trestbps**: Cleveland có 50 unique values, Heart có 49 unique values

## Kết luận về cách chuyển đổi

### Quy trình thực tế:
```
Original UCI Dataset (với '?')
         ↓
Heart Dataset (encode '?' → 0,4) 
         ↓
Cleveland Dataset (loại bỏ missing values)
         ↓
Heart Dataset (augment/duplicate để tăng kích thước)
```

### Giải thích nghịch lý:
**Heart dataset KHÔNG phải được tạo từ Cleveland dataset!**

**Thực tế là:**
1. **Heart dataset** = Dataset gốc từ UCI với missing values được encode thành 0 và 4
2. **Cleveland dataset** = Phiên bản đã được **làm sạch** từ Heart dataset (loại bỏ missing values)
3. **Heart dataset** sau đó được **augment** (duplicate) để tạo dataset lớn hơn

### Bằng chứng:
1. **Missing codes trong Heart**: ca=4 (18 cases), thal=0 (7 cases)
2. **Cleveland sạch**: Không có missing values
3. **Augmentation**: Heart có 3.39x duplication
4. **Medical validity**: ca=4 và thal=0 không hợp lệ về mặt y học

## Khuyến nghị

### Sử dụng dataset nào?
- **Cho training**: Sử dụng Cleveland dataset (sạch, không có missing values)
- **Cho research**: Có thể sử dụng Heart dataset để nghiên cứu về missing value handling
- **Cho production**: Cleveland dataset được khuyến nghị

### Xử lý missing values:
- ca=4 và thal=0 trong Heart dataset nên được xử lý như missing values
- Có thể loại bỏ hoặc impute các giá trị này

## Files được tạo
1. `compare_heart_datasets.py` - Script phân tích so sánh
2. `check_missing_data.py` - Script kiểm tra missing values
3. `deep_missing_analysis.py` - Script phân tích sâu
4. `heart_datasets_comparison.png` - Visualization so sánh

## Ngày tạo
Ngày: 2024-12-19
Người phân tích: AI Assistant
