# 📚 Ghi Chú về Post-Processing và Cách Chia Dữ Liệu

## Mục Lục
1. [Post-Processing là gì?](#post-processing-là-gì)
2. [Cách Điều Chỉnh Prediction](#cách-điều-chỉnh-prediction)
3. [Cách Chia Dữ Liệu](#cách-chia-dữ-liệu)
4. [TimeSeriesSplit - Cách Chia Validation](#timeseriessplit---cách-chia-validation)
5. [Tại Sao Các Fold Không Bằng Nhau?](#tại-sao-các-fold-không-bằng-nhau)
6. [Tại Sao Phát Hiện Được Độ Lệch?](#tại-sao-phát-hiện-được-độ-lệch)
7. [Normalize trong Post-Processing](#normalize-trong-post-processing)

---

## Post-Processing là gì?

### Định nghĩa
Post-processing là bước **điều chỉnh predictions** sau khi model chính (PatchTST) đã dự đoán, nhằm giảm sai số và bias.

### Vấn đề cần giải quyết

#### 1. Model có bias:
- PatchTST baseline có thể có bias hệ thống (ví dụ: dự đoán cao hơn thực tế)
- Ví dụ: Baseline dự đoán 120, thực tế 100 → bias = +20

#### 2. Predictions chưa tối ưu:
- Model có thể có pattern sai số nhất quán
- Ví dụ: Luôn dự đoán cao hơn 15% so với thực tế

### Cách Post-Processing hoạt động

#### Bước 1: Thu thập dữ liệu từ Cross-Validation

```python
# Sử dụng TimeSeriesSplit để tạo nhiều folds
tscv = TimeSeriesSplit(n_splits=3)
X_post = []  # Lưu predictions từ PatchTST
y_post = []  # Lưu ground truth thực tế

for train_idx, val_idx in tscv.split(full_data):
    # Train PatchTST trên train_fold
    model_fold = PatchTST(...)
    nf_fold.fit(df=train_fold, val_size=0)
    forecast_fold = nf_fold.predict()
    
    # Lấy predictions và ground truth từ val_fold
    pred_fold = forecast_fold[pred_col].values  # Predictions
    true_fold = val_fold['y'].values             # Ground truth
    
    # Thu thập cặp (prediction, ground_truth)
    X_post.extend(pred_fold.reshape(-1, 1))
    y_post.extend(true_fold)
```

**Kết quả**: Thu thập ~300 điểm (predictions, ground_truth) từ 3 folds

#### Bước 2: Train Post-processing Model

```python
# Train Linear Regression để học mapping: prediction → actual
best_post_model = LinearRegression()
best_post_model.fit(X_post, y_post)

# Model học được công thức:
# y_corrected = coef * y_predicted + intercept
# Ví dụ: y_corrected = 0.7267 * y_predicted + 9.3249
```

**Ý nghĩa**:
- Học cách điều chỉnh predictions từ PatchTST
- Tìm hệ số để map predictions → giá trị thực tế

#### Bước 3: Áp dụng Post-processing

```python
# Áp dụng công thức đã học để điều chỉnh predictions
pred_post = best_post_model.predict(pred_patchtst_baseline.reshape(-1, 1))

# Ví dụ:
# Baseline prediction: 120
# Post-processing: 0.7267 * 120 + 9.3249 = 96.53
# → Điều chỉnh từ 120 xuống 96.53 (gần với thực tế hơn)
```

### Ví dụ cụ thể

#### Trước Post-processing:
```
Baseline Prediction: 120
Ground Truth: 100
Error: +20 (dự đoán cao hơn)
```

#### Sau Post-processing:
```
Post-processing: 0.7267 * 120 + 9.3249 = 96.53
Ground Truth: 100
Error: -3.47 (chỉ sai 3.47 điểm, tốt hơn nhiều!)
```

### Tại sao Post-processing hiệu quả?

1. **Học từ dữ liệu thực tế**:
   - Không hardcode, học từ validation folds
   - Tự động phát hiện pattern sai số

2. **Điều chỉnh có hệ thống**:
   - Áp dụng công thức nhất quán cho tất cả predictions
   - Giảm bias hệ thống

3. **Cải thiện đáng kể**:
   - Baseline MSE: 639.02
   - Post-processing MSE: 71.22
   - Cải thiện: ~88.86%

---

## Cách Điều Chỉnh Prediction

### Bước 1: Học công thức điều chỉnh

```python
# Train Linear Regression để học mapping
best_post_model = LinearRegression()
best_post_model.fit(X_post, y_post)

# Model học được:
# coef = 0.7267
# intercept = 9.3249
# → Công thức: y_corrected = 0.7267 * y_predicted + 9.3249
```

**Ý nghĩa**:
- `coef` (0.7267): hệ số điều chỉnh tỷ lệ
- `intercept` (9.3249): hệ số điều chỉnh cộng thêm

### Bước 2: Áp dụng công thức

```python
# Input: predictions từ PatchTST baseline
pred_patchtst_baseline = [120, 115, 110, ...]  # 100 predictions

# Áp dụng công thức cho từng prediction
pred_post = best_post_model.predict(pred_patchtst_baseline.reshape(-1, 1))

# Tương đương với:
# pred_post[0] = 0.7267 * 120 + 9.3249 = 96.53
# pred_post[1] = 0.7267 * 115 + 9.3249 = 92.90
# pred_post[2] = 0.7267 * 110 + 9.3249 = 89.26
# ...
```

### Ví dụ minh họa

#### Trước điều chỉnh (Baseline):
```
Prediction từ PatchTST: 120
Ground Truth thực tế: 100
Error: +20 (dự đoán cao hơn 20 điểm)
```

#### Sau điều chỉnh (Post-processing):
```python
# Áp dụng công thức
pred_corrected = 0.7267 * 120 + 9.3249
                = 87.204 + 9.3249
                = 96.53

# So sánh
Ground Truth: 100
Error: -3.47 (chỉ sai 3.47 điểm, tốt hơn nhiều!)
```

### Giải thích công thức

#### Công thức: `y = coef * pred + intercept`

**1. Hệ số `coef` (0.7267)**:
- **Ý nghĩa**: PatchTST có xu hướng dự đoán cao hơn thực tế
- **Tác dụng**: Giảm predictions xuống ~72.67% giá trị gốc
- **Ví dụ**: 120 → 120 × 0.7267 = 87.2

**2. Hệ số `intercept` (9.3249)**:
- **Ý nghĩa**: Bù thêm một lượng cố định
- **Tác dụng**: Điều chỉnh offset sau khi nhân với coef
- **Ví dụ**: 87.2 + 9.3249 = 96.53

### Ví dụ với nhiều predictions

```python
# Predictions từ PatchTST baseline
baseline_predictions = [120, 115, 110, 105, 100]

# Áp dụng post-processing
corrected_predictions = []
for pred in baseline_predictions:
    corrected = 0.7267 * pred + 9.3249
    corrected_predictions.append(corrected)

# Kết quả:
# Baseline:  [120,  115,  110,  105,  100]
# Corrected: [96.5, 92.9, 89.3, 85.6, 82.0]
```

### So sánh trước và sau

#### Trường hợp 1: Prediction cao (120)
```
Baseline:     120
Post-process: 0.7267 * 120 + 9.3249 = 96.53
Giảm:         120 - 96.53 = 23.47 điểm
```

#### Trường hợp 2: Prediction trung bình (100)
```
Baseline:     100
Post-process: 0.7267 * 100 + 9.3249 = 82.00
Giảm:         100 - 82.00 = 18.00 điểm
```

#### Trường hợp 3: Prediction thấp (80)
```
Baseline:     80
Post-process: 0.7267 * 80 + 9.3249 = 67.46
Giảm:         80 - 67.46 = 12.54 điểm
```

### Tại sao công thức này hiệu quả?

1. **Học từ dữ liệu thực tế**:
   ```python
   # Model học từ 300 điểm (predictions, ground_truth)
   # Tự động phát hiện pattern:
   # - PatchTST thường dự đoán cao hơn
   # - Cần giảm xuống ~72.67% và cộng thêm 9.32
   ```

2. **Điều chỉnh nhất quán**:
   - Áp dụng cùng công thức cho tất cả predictions
   - Giảm bias hệ thống

3. **Cải thiện đáng kể**:
   ```
   Baseline MSE:  641.50
   Post-process:  48.62
   Cải thiện:     92.42%
   ```

---

## Cách Chia Dữ Liệu

### Trong `patchtst_best_method.ipynb`

#### 1. Chia dữ liệu training ban đầu (Cell 5):
```python
# Chia train/validation
train_size = int(T * 0.8)      # 80% đầu tiên
val_size = int(T * 0.1)        # 10% tiếp theo

train_data = close_values[:train_size]
val_data = close_values[train_size:train_size + val_size]
```

**Kết quả**:
- Train: 919 điểm (80%)
- Val: 114 điểm (10%)
- Còn lại: ~116 điểm (10%) không dùng cho train/val

#### 2. Chuẩn bị dữ liệu cho NeuralForecast (Cell 6):
```python
# Full train (train + val) để train final model
train_nf_full = pd.concat([train_nf, val_nf], ignore_index=True)
```

- `train_nf_full`: 1033 điểm (919 + 114) — dùng để train final model

#### 3. Chia dữ liệu cho Optuna optimization (Cell 8):
```python
# Chia dữ liệu cho Optuna optimization
# Dùng 90% đầu để train, 10% cuối để validation
optuna_train_size = int(len(train_nf_full) * 0.9)
train_nf_optuna = train_nf_full.iloc[:optuna_train_size].copy()
val_nf_optuna = train_nf_full.iloc[optuna_train_size:].copy()
```

**Kết quả**:
- Train cho Optuna: 929 điểm (90% của 1033)
- Val cho Optuna: 104 điểm (10% của 1033)

#### 4. Test data (Cell 6):
- Lấy từ file `FPT_test.csv` (tách biệt, không từ training data)
- Lọc theo `symbol='FPT'` và ngày > ngày cuối cùng của training
- Lấy 100 điểm đầu tiên làm ground truth (`y_true`)

#### 5. TimeSeriesSplit cho Post-processing (Cell 13):
- Dùng `TimeSeriesSplit(n_splits=3)` để tạo 3 folds cho cross-validation khi train post-processing model

### Tóm tắt:
- **Training data ban đầu**: 80% (919 điểm)
- **Validation data ban đầu**: 10% (114 điểm)
- **Full training (train+val)**: 1033 điểm — dùng để train final model
- **Optuna optimization**: 90/10 từ `train_nf_full` (929/104)
- **Test data**: Lấy từ file test riêng (100 điểm)
- **Post-processing**: TimeSeriesSplit với 3 folds

---

## TimeSeriesSplit - Cách Chia Validation

### Không phải random — Chia tuần tự theo thời gian

```python
tscv = TimeSeriesSplit(n_splits=3)
# Chia theo thứ tự thời gian, KHÔNG random
```

### Cách chia với 1033 điểm:

```
Full Data: [1, 2, 3, ..., 1033]
           ↑                    ↑
        Quá khứ              Tương lai

TimeSeriesSplit(n_splits=3) chia như sau:

┌─────────────────────────────────────────────────┐
│ Fold 1:                                          │
│   Train: [1-344]        (344 điểm đầu)          │
│   Val:   [345-688]      (344 điểm tiếp theo)     │
│                                 ↑                │
│                         Validation là "tương lai"│
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Fold 2:                                          │
│   Train: [1-688]        (gộp Fold 1 train+val)  │
│   Val:   [689-860]      (172 điểm tiếp theo)     │
│                                 ↑                │
│                         Validation là "tương lai"│
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Fold 3:                                          │
│   Train: [1-860]        (gộp Fold 1+2 train+val)│
│   Val:   [861-1033]     (173 điểm cuối)         │
│                                 ↑                │
│                         Validation là "tương lai"│
└─────────────────────────────────────────────────┘
```

### Đặc điểm của TimeSeriesSplit

#### 1. Chia tuần tự (không random):
- Giữ nguyên thứ tự thời gian
- Train luôn là quá khứ, Val luôn là tương lai
- Không xáo trộn dữ liệu

#### 2. Walk-forward validation:
```
Fold 1: Train[1-344]     → Val[345-688]
Fold 2: Train[1-688]     → Val[689-860]   (train mở rộng)
Fold 3: Train[1-860]     → Val[861-1033]  (train mở rộng)
```

#### 3. Validation luôn là "tương lai":
- Mỗi fold: Val là phần tiếp theo sau Train
- Mô phỏng dự đoán tương lai

### So sánh với các cách chia khác

#### Random Split (không dùng cho time series):
```python
# ❌ SAI cho time series
from sklearn.model_selection import train_test_split
train, val = train_test_split(data, test_size=0.2, random_state=42)
# → Xáo trộn thứ tự thời gian
# → Val có thể là quá khứ, Train có thể là tương lai
```

#### TimeSeriesSplit (đúng cho time series):
```python
# ✅ ĐÚNG cho time series
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
# → Giữ nguyên thứ tự thời gian
# → Val luôn là tương lai so với Train
```

#### Chia đơn giản (chỉ lấy cuối):
```python
# Chia đơn giản: 80% đầu train, 20% cuối val
train = data[:int(len(data) * 0.8)]
val = data[int(len(data) * 0.8):]
# → Chỉ có 1 fold
# → Ít data để học pattern bias
```

### Tại sao TimeSeriesSplit phù hợp?

1. **Mô phỏng thực tế**:
   ```
   Thực tế: Dự đoán tương lai dựa trên quá khứ
   TimeSeriesSplit: Train (quá khứ) → Val (tương lai)
   ```

2. **Thu thập nhiều data**:
   ```
   Chia đơn giản: 1 fold → ~200 điểm
   TimeSeriesSplit: 3 folds → ~300 điểm (nhiều hơn)
   ```

3. **Học pattern bias đa dạng**:
   ```
   Fold 1: Bias ở giai đoạn giữa
   Fold 2: Bias ở giai đoạn sau
   Fold 3: Bias ở giai đoạn cuối
   → Học được pattern bias qua nhiều giai đoạn
   ```

### Ví dụ minh họa

#### Với 1033 điểm:

```python
# TimeSeriesSplit(n_splits=3)
# 
# Fold 1:
#   Train indices: [0, 1, 2, ..., 343]      (344 điểm)
#   Val indices:   [344, 345, ..., 687]    (344 điểm)
#
# Fold 2:
#   Train indices: [0, 1, 2, ..., 687]      (688 điểm)
#   Val indices:   [688, 689, ..., 859]    (172 điểm)
#
# Fold 3:
#   Train indices: [0, 1, 2, ..., 859]      (860 điểm)
#   Val indices:   [860, 861, ..., 1032]   (173 điểm)
```

### Tại Sao Các Fold Không Bằng Nhau?

#### Thuật toán Expanding Window (Cửa sổ mở rộng)

TimeSeriesSplit sử dụng thuật toán **Expanding Window**, không phải chia đều. Đây là đặc điểm thiết kế, không phải lỗi.

#### Công thức chia:

Với `n_splits=3` và `n_samples=1033`:

```
test_size = n_samples / (n_splits + 1)
test_size = 1033 / 4 ≈ 258.25 điểm
```

**Cách chia:**

```
Fold 1:
  - Training size = 1 × test_size ≈ 258 điểm
  - Validation size = test_size ≈ 258 điểm

Fold 2:
  - Training size = 2 × test_size ≈ 516 điểm (mở rộng)
  - Validation size = test_size ≈ 258 điểm

Fold 3:
  - Training size = 3 × test_size ≈ 774 điểm (mở rộng tiếp)
  - Validation size = test_size ≈ 258 điểm
```

**Kết quả thực tế với 1033 điểm:**
- Fold 1: Train ~344, Val ~344
- Fold 2: Train ~688, Val ~172
- Fold 3: Train ~860, Val ~173

*Lưu ý: Sklearn có thể làm tròn một chút, dẫn đến số liệu hơi khác so với công thức lý thuyết.*

#### Tại sao không chia đều?

**1. Mô phỏng thực tế:**
```
Thực tế: Khi dự đoán tương lai, ta có:
- Ngày 1: Dự đoán dựa trên dữ liệu từ ngày 1-100
- Ngày 2: Dự đoán dựa trên dữ liệu từ ngày 1-101 (nhiều hơn)
- Ngày 3: Dự đoán dựa trên dữ liệu từ ngày 1-102 (nhiều hơn nữa)

TimeSeriesSplit mô phỏng điều này:
- Fold 1: Train nhỏ → Val
- Fold 2: Train lớn hơn → Val (mô phỏng có thêm dữ liệu)
- Fold 3: Train lớn nhất → Val (mô phỏng có nhiều dữ liệu nhất)
```

**2. Training set mở rộng dần:**
- Fold sau có nhiều dữ liệu training hơn fold trước
- Giúp model học tốt hơn với nhiều dữ liệu hơn
- Phản ánh thực tế: càng về sau càng có nhiều dữ liệu lịch sử

**3. Validation luôn là tương lai:**
- Đảm bảo không có data leakage
- Mỗi fold validation là "tương lai" so với training của fold đó

#### So sánh với chia đều (nếu có):

**Nếu chia đều (KHÔNG phù hợp time series):**
```
Fold 1: Train [0-257], Val [258-515]     (258 điểm mỗi phần)
Fold 2: Train [258-515], Val [516-773]   (258 điểm mỗi phần)
Fold 3: Train [516-773], Val [774-1032]  (258 điểm mỗi phần)
```

**Vấn đề:**
- Validation có thể là "quá khứ" so với một số training data
- Không mô phỏng thực tế (trong thực tế, ta không dùng tương lai để dự đoán quá khứ)
- Không phù hợp với time series

**TimeSeriesSplit (ĐÚNG):**
```
Fold 1: Train [0-343], Val [344-687]     (Train nhỏ, Val tiếp theo)
Fold 2: Train [0-687], Val [688-859]     (Train mở rộng, Val tiếp theo)
Fold 3: Train [0-859], Val [860-1032]    (Train mở rộng nhất, Val cuối)
```

**Ưu điểm:**
- Validation luôn là "tương lai" so với training
- Mô phỏng thực tế việc dự đoán
- Training set mở rộng dần (giống thực tế)

#### Lợi ích của cách chia này:

1. **Mô phỏng thực tế**: Mỗi fold giống một lần dự đoán tương lai trong thực tế
2. **Training set tăng dần**: Fold sau có nhiều dữ liệu hơn → model học tốt hơn
3. **Thu thập nhiều validation data**: ~300 điểm từ 3 folds (nhiều hơn 1 fold)
4. **Học pattern bias đa dạng**: Bias ở các giai đoạn khác nhau (giữa, sau, cuối)

#### Tóm tắt:

- ✅ **Các fold không bằng nhau là ĐÚNG** - do thuật toán Expanding Window
- ✅ **Training set mở rộng dần** - fold sau có nhiều dữ liệu hơn fold trước
- ✅ **Validation luôn là tương lai** - đảm bảo không có data leakage
- ✅ **Mô phỏng thực tế** - giống cách dự đoán trong thực tế

**Công thức tổng quát:**
```
Với n_splits = k và n_samples = n:
  test_size = n / (k + 1)
  
  Fold i (i = 1, 2, ..., k):
    - Training size = i × test_size
    - Validation size = test_size
```

---

## Tại Sao Phát Hiện Được Độ Lệch?

### 1. Validation folds là "unseen data" đối với model

```python
# TimeSeriesSplit chia dữ liệu theo thời gian
for train_idx, val_idx in tscv.split(full_data):
    train_fold = full_data.iloc[train_idx]  # Data từ quá khứ
    val_fold = full_data.iloc[val_idx]      # Data từ tương lai (chưa thấy)
    
    # Model CHỈ train trên train_fold
    model_fold = PatchTST(...)
    nf_fold.fit(df=train_fold, val_size=0)  # ← Chỉ học từ train_fold
    
    # Predict trên val_fold (chưa từng thấy)
    forecast_fold = nf_fold.predict()  # ← Predict trên data mới
```

**Điểm quan trọng**:
- Model chỉ train trên `train_fold`
- Predict trên `val_fold` (chưa thấy trong quá trình train)
- Validation folds là "tương lai" so với training folds

### 2. Ví dụ cụ thể với TimeSeriesSplit:

```
Full Data (1033 điểm): [1, 2, 3, ..., 1033]

Fold 1:
  Train: [1-344]     → Model học từ điểm 1-344
  Val:   [345-688]   → Predict trên điểm 345-688 (CHƯA THẤY)
  
Fold 2:
  Train: [1-688]     → Model học từ điểm 1-688
  Val:   [689-860]   → Predict trên điểm 689-860 (CHƯA THẤY)
  
Fold 3:
  Train: [1-860]     → Model học từ điểm 1-860
  Val:   [861-1033]  → Predict trên điểm 861-1033 (CHƯA THẤY)
```

### 3. Tại sao có độ lệch?

#### A. Model có bias hệ thống:

```python
# Model được train trên train_fold
# Nhưng khi predict trên val_fold (data mới), có thể có bias:

# Ví dụ:
# Train data: giá cổ phiếu 80-100
# Val data:   giá cổ phiếu 100-120 (cao hơn)

# Model học từ train data (80-100)
# → Khi predict trên val data (100-120), có thể dự đoán thấp hơn
# → Bias: predictions < ground truth
```

#### B. Distribution shift theo thời gian:

```python
# Time series có thể thay đổi theo thời gian
# Early data (train): pattern A
# Later data (val):   pattern B (khác pattern A)

# Model học pattern A
# → Predict pattern B → Có thể sai
```

#### C. Model chưa tối ưu:

```python
# Model được train với:
# - max_steps = 250-300 (có thể chưa đủ)
# - Learning rate có thể chưa tối ưu
# - Hyperparameters có thể chưa perfect

# → Model có thể có bias khi predict trên data mới
```

### 4. Ví dụ minh họa:

```python
# Fold 1:
train_fold = [80, 82, 85, 88, 90, ...]  # Giá tăng dần
val_fold   = [95, 98, 100, 102, ...]    # Giá cao hơn

# Model train trên train_fold
# → Học pattern: giá ~80-90
# → Predict trên val_fold: dự đoán ~85-95
# → Ground truth: 95-102
# → Bias: predictions thấp hơn ~5-10 điểm

# Thu thập:
X_post.append(85)  # Prediction
y_post.append(95)  # Ground truth
# → Độ lệch: 85 - 95 = -10
```

### 5. Tại sao độ lệch này có thể phát hiện?

#### A. Model có pattern sai số nhất quán:

```python
# Nếu model luôn dự đoán thấp hơn 10%:
# Prediction: 90 → Ground truth: 100 (thiếu 10%)
# Prediction: 95 → Ground truth: 105 (thiếu 10%)

# Post-processing học được:
# y = 1.1 * pred + 0  (tăng 10%)
```

#### B. Bias hệ thống (systematic bias):

```python
# Model có bias cố định:
# Prediction: 100 → Ground truth: 120 (thiếu 20)
# Prediction: 110 → Ground truth: 130 (thiếu 20)

# Post-processing học được:
# y = 1.0 * pred + 20  (cộng thêm 20)
```

#### C. Bias tỷ lệ (proportional bias):

```python
# Model có bias tỷ lệ:
# Prediction: 100 → Ground truth: 120 (thiếu 20%)
# Prediction: 200 → Ground truth: 240 (thiếu 20%)

# Post-processing học được:
# y = 1.2 * pred + 0  (tăng 20%)
```

### 6. Kết quả thực tế:

```python
# Từ notebook:
# Formula: y = 0.7267 * pred + 9.3249

# Điều này có nghĩa:
# - Model dự đoán cao hơn thực tế
# - Cần giảm xuống 72.67% và cộng thêm 9.32

# Ví dụ:
# Prediction: 120
# Post-process: 0.7267 * 120 + 9.3249 = 96.53
# → Giảm 23.47 điểm (model dự đoán cao)
```

### Tóm tắt: Tại sao phát hiện được độ lệch?

1. **Validation folds là "unseen data"**:
   - Model chỉ train trên `train_fold`
   - Predict trên `val_fold` (chưa thấy)
   - TimeSeriesSplit đảm bảo validation là "tương lai"

2. **Model có bias khi predict trên data mới**:
   - Distribution shift theo thời gian
   - Model chưa tối ưu hoàn toàn
   - Pattern khác nhau giữa train và val

3. **Bias có pattern nhất quán**:
   - Systematic bias (cố định)
   - Proportional bias (tỷ lệ)
   - Post-processing học được pattern này

4. **Thu thập từ nhiều folds**:
   - 3 folds → ~300 điểm
   - Đủ để học pattern bias
   - Pattern nhất quán qua các folds

---

## Normalize trong Post-Processing

### Kết luận: KHÔNG có normalize trong post-processing

#### 1. Trong `patchtst_best_method.ipynb`:

```python
# Thu thập dữ liệu - DÙNG TRỰC TIẾP (raw values)
X_post.extend(pred_fold.reshape(-1, 1))  # Predictions từ PatchTST
y_post.extend(true_fold)                 # Ground truth thực tế

# KHÔNG có normalize/scaling
X_post = np.array(X_post)
y_post = np.array(y_post)

# Train trực tiếp trên raw values
best_post_model = LinearRegression()
best_post_model.fit(X_post, y_post)  # ← Dùng raw values, không normalize
```

**Không có**:
- ❌ StandardScaler
- ❌ RobustScaler
- ❌ MinMaxScaler
- ❌ Bất kỳ normalization/scaling nào

#### 2. Trong `patchtst_advanced_preprocessing.ipynb`:

```python
# Chỉ có inverse transform nếu dùng log transformation
if selected_preprocessing in ['log_only', 'log_transformed']:
    # Inverse log transform (đưa về giá trị gốc)
    pred_fold = np.exp(pred_fold_raw) - log_epsilon
    true_fold = np.exp(true_fold) - log_epsilon
else:
    # Không cần inverse transform nếu dùng original data
    pred_fold = pred_fold_raw

# Vẫn KHÔNG có normalize/scaling cho post-processing
X_post.extend(pred_fold.reshape(-1, 1))
y_post.extend(true_fold)
```

**Chỉ có**: Inverse transform (nếu dùng log), không có normalize.

### Tại sao không cần normalize?

#### 1. Linear Regression không bắt buộc normalize:
- Linear Regression không yêu cầu normalize
- Có thể hoạt động tốt với raw values

#### 2. Dữ liệu đã ở cùng scale:
```python
# Predictions từ PatchTST: ~80-120 (giá cổ phiếu)
# Ground truth: ~80-120 (cùng scale)
# → Không cần normalize vì đã cùng scale
```

#### 3. Model học được mapping trực tiếp:
```python
# Công thức học được:
y = 0.7267 * pred + 9.3249

# Áp dụng trực tiếp trên raw values:
corrected = 0.7267 * 120 + 9.3249 = 96.53
```

### So sánh với các phương pháp khác:

#### Có normalize (ví dụ trong preprocessing):
```python
# Nếu có normalize:
scaler = StandardScaler()
X_post_scaled = scaler.fit_transform(X_post)
y_post_scaled = scaler.fit_transform(y_post.reshape(-1, 1))

# Train trên scaled data
model.fit(X_post_scaled, y_post_scaled)

# Predict cần inverse transform
pred_scaled = model.predict(X_test_scaled)
pred = scaler.inverse_transform(pred_scaled)
```

#### Không normalize (như trong notebook này):
```python
# Dùng trực tiếp raw values
model.fit(X_post, y_post)  # Raw values

# Predict trực tiếp
pred = model.predict(X_test)  # Raw values, không cần inverse
```

### Lưu ý về TemporalNorm trong PatchTST:

```python
# PatchTST có TemporalNorm BÊN TRONG model
model = PatchTST(
    revin=True,  # ← Có normalization bên trong model
    ...
)
```

- TemporalNorm là normalization **bên trong** PatchTST
- Không liên quan đến post-processing
- Post-processing nhận predictions đã được model xử lý (raw values)

### Tóm tắt:

- ✅ **KHÔNG có normalize** trong post-processing
- ✅ Dữ liệu được dùng **trực tiếp** (raw values)
- ✅ Linear Regression train trên raw predictions và raw ground truth
- ❌ Không cần StandardScaler, RobustScaler, hay MinMaxScaler
- ❌ Không cần inverse transform (trừ khi dùng log transformation trong preprocessing)

**Lý do**:
1. Predictions và ground truth đã cùng scale (~80-120)
2. Linear Regression không yêu cầu normalize
3. Đơn giản hóa pipeline, tránh lỗi transform/inverse transform

---

## Post-Processing Chia Dữ Liệu Train/Val

### Sử dụng TimeSeriesSplit với 3 folds:

```python
# Train linear regression để map predictions -> actual values
tscv = TimeSeriesSplit(n_splits=3)
X_post = []
y_post = []

full_data = train_nf_full.copy()  # 1033 điểm (train + val gộp lại)
for train_idx, val_idx in tscv.split(full_data):
    train_fold = full_data.iloc[train_idx]
    val_fold = full_data.iloc[val_idx]
    
    # Train PatchTST model trên train_fold
    model_fold = PatchTST(...)
    nf_fold = NeuralForecast(models=[model_fold], freq='D')
    nf_fold.fit(df=train_fold, val_size=0)
    forecast_fold = nf_fold.predict()
    
    # Lấy predictions từ val_fold
    pred_fold = forecast_fold[pred_col].values[:len(val_fold)]
    true_fold = val_fold['y'].values[:len(pred_fold)]
    
    # Thu thập data cho post-processing model
    X_post.extend(pred_fold.reshape(-1, 1))  # Predictions
    y_post.extend(true_fold)                 # Ground truth

# Train post-processing model trên tất cả data đã thu thập
X_post = np.array(X_post)
y_post = np.array(y_post)
best_post_model = LinearRegression()
best_post_model.fit(X_post, y_post)
```

### Cách hoạt động:

#### 1. TimeSeriesSplit(n_splits=3):
- Chia `train_nf_full` (1033 điểm) thành 3 folds theo thời gian
- Mỗi fold có train và validation riêng, không trùng lặp

#### 2. Với mỗi fold:
- **Train fold**: Dùng để train PatchTST model
- **Val fold**: Dùng để predict và thu thập cặp (prediction, ground_truth)

#### 3. Thu thập dữ liệu:
- Gộp tất cả predictions và ground truth từ 3 folds
- Kết quả: ~300 điểm (100 điểm/fold × 3 folds)

#### 4. Train Post-processing Model:
- Train Linear Regression trên toàn bộ dữ liệu đã thu thập
- Input: `X_post` (predictions từ PatchTST)
- Output: `y_post` (ground truth thực tế)

### Ví dụ cụ thể với 1033 điểm:

**TimeSeriesSplit(n_splits=3)** sẽ chia như sau:

- **Fold 1**:
  - Train: ~344 điểm đầu
  - Val: ~344 điểm tiếp theo
- **Fold 2**:
  - Train: ~688 điểm đầu (gộp fold 1 train + val)
  - Val: ~172 điểm tiếp theo
- **Fold 3**:
  - Train: ~860 điểm đầu (gộp tất cả trước đó)
  - Val: ~173 điểm cuối

### Tóm tắt:

- **Dữ liệu gốc**: `train_nf_full` (1033 điểm = train + val ban đầu)
- **Phương pháp**: TimeSeriesSplit với 3 folds
- **Mục đích**: Thu thập nhiều cặp (prediction, ground_truth) từ nhiều folds
- **Kết quả**: ~300 điểm để train post-processing model
- **Post-processing model**: Linear Regression được train trên tất cả data đã thu thập

**Điểm khác biệt**: Post Processing không chia train/val cố định mà dùng cross-validation (TimeSeriesSplit) để thu thập dữ liệu từ nhiều folds, giúp model học tốt hơn.

---

## Thuật Toán Học Bias Từ Đâu?

### ✅ Đúng: Học từ tập train+val (KHÔNG dùng test data)

#### Nguồn dữ liệu để học post-processing:

**1. Dữ liệu học (Training Data)**:

```python
# Dùng train_nf_full (train + val gộp lại)
full_data = train_nf_full.copy()  # 1033 điểm (919 train + 114 val)

# Chia thành folds bằng TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
for train_idx, val_idx in tscv.split(full_data):
    train_fold = full_data.iloc[train_idx]  # Train fold
    val_fold = full_data.iloc[val_idx]       # Val fold
    
    # Train PatchTST trên train_fold
    model_fold = PatchTST(...)
    nf_fold.fit(df=train_fold, val_size=0)
    
    # Predict trên val_fold
    forecast_fold = nf_fold.predict()
    pred_fold = forecast_fold[pred_col].values  # Predictions
    true_fold = val_fold['y'].values            # Ground truth từ VAL FOLD
    
    # Thu thập để train post-processing
    X_post.extend(pred_fold)  # Predictions từ val_fold
    y_post.extend(true_fold)  # Ground truth từ val_fold
```

**Kết quả**: Thu thập ~300 điểm từ validation folds của train_nf_full

**2. Test data KHÔNG được dùng để học**:

```python
# Test data CHỈ dùng để đánh giá cuối cùng
y_true = df_test.head(horizon)["close"].values  # Test data

# Train post-processing model (KHÔNG dùng y_true)
best_post_model.fit(X_post, y_post)  # ← Chỉ dùng data từ folds

# Đánh giá trên test data (SAU KHI đã train xong)
pred_post = best_post_model.predict(pred_patchtst_baseline.reshape(-1, 1))
mse_post = mean_squared_error(y_true, pred_post)  # ← Chỉ để đánh giá
```

### Quy trình học bias:

#### Bước 1: Thu thập từ validation folds
```
Train_nf_full (1033 điểm)
├── Fold 1: Train(344) → Predict → Val(344) → Thu thập 344 điểm
├── Fold 2: Train(688) → Predict → Val(172) → Thu thập 172 điểm  
└── Fold 3: Train(860) → Predict → Val(173) → Thu thập 173 điểm

Tổng: ~300 điểm (predictions, ground_truth) từ VALIDATION FOLDS
```

#### Bước 2: Học công thức
```python
# Học từ 300 điểm đã thu thập
best_post_model.fit(X_post, y_post)

# Học được: y = 0.7267 * pred + 9.3249
# Công thức này học từ độ lệch giữa predictions và ground truth
# trong VALIDATION FOLDS của training data
```

#### Bước 3: Áp dụng và đánh giá
```python
# Áp dụng cho test predictions
pred_post = best_post_model.predict(pred_patchtst_baseline)

# Đánh giá trên test data (chưa từng thấy)
mse_post = mean_squared_error(y_true, pred_post)
```

### Tại sao đúng?

1. **Không leak test data**:
   - Test data (`y_true`) chỉ dùng để đánh giá
   - Không được dùng để train post-processing model
   - Đảm bảo đánh giá công bằng

2. **Học từ validation folds**:
   - Dùng TimeSeriesSplit trên `train_nf_full`
   - Mỗi fold: train trên train_fold, predict trên val_fold
   - Thu thập độ lệch từ validation folds

3. **Giả định bias tương tự**:
   - Giả định bias trên validation folds tương tự trên test data
   - Nếu bias khác nhiều, post-processing có thể kém hiệu quả

### So sánh với các phương pháp:

#### Nếu dùng test data (sai):
```python
# ❌ SAI - Leak test data
X_post.extend(pred_test)  # Predictions trên test
y_post.extend(y_true)     # Ground truth từ test ← LEAK!
model.fit(X_post, y_post)  # Học từ test data
```

#### Cách làm đúng (như trong notebook):
```python
# ✅ ĐÚNG - Chỉ dùng train+val
X_post.extend(pred_val_fold)  # Predictions trên val fold
y_post.extend(true_val_fold)  # Ground truth từ val fold
model.fit(X_post, y_post)     # Học từ validation folds

# Test data chỉ để đánh giá
mse = mean_squared_error(y_true, model.predict(pred_test))
```

### Tóm tắt:

- ✅ **Đúng**: Post-processing học cơ chế bias chỉ từ tập train+val
- ✅ **Cụ thể**:
  - Dùng `train_nf_full` (train + val gộp lại)
  - Chia bằng TimeSeriesSplit thành 3 folds
  - Thu thập độ lệch từ validation folds
  - Học công thức: `y = coef * pred + intercept`
- ✅ **Test data**:
  - Không được dùng để train
  - Chỉ dùng để đánh giá cuối cùng
- ✅ **Giả định**:
  - Bias trên validation folds tương tự trên test data
  - Nếu khác nhiều, hiệu quả có thể giảm

Đây là cách làm **đúng**, tránh data leakage và đảm bảo đánh giá công bằng.

---

## Kết Luận

Post-processing là một kỹ thuật hiệu quả để cải thiện predictions của time series forecasting models:

1. **Học từ validation folds**: Thu thập độ lệch từ nhiều folds để học pattern bias
2. **Không dùng test data**: Đảm bảo đánh giá công bằng, tránh data leakage
3. **Đơn giản nhưng hiệu quả**: Chỉ cần Linear Regression, không cần normalize
4. **Cải thiện đáng kể**: Giảm MSE từ 639 → 71 (cải thiện ~88.86%)

Cách chia dữ liệu sử dụng TimeSeriesSplit đảm bảo:
- Validation luôn là "tương lai" so với training
- Mô phỏng thực tế việc dự đoán tương lai
- Thu thập nhiều data để học pattern bias đa dạng

