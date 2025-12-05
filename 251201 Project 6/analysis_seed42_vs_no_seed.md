# Phân Tích: Seed 42 vs Không Seed - Tại Sao Kết Quả Cuối Lại Tệ Hơn?

## 📊 So Sánh Kết Quả

### 1. Best Parameters từ Optuna

| Tham số | Không Seed | Seed 42 | Khác biệt |
|---------|-----------|---------|-----------|
| **input_size** | 100 | 200 | +100 (tăng gấp đôi) |
| **patch_len** | 32 | 24 | -8 (giảm 25%) |
| **stride** | 4 | 16 | +12 (tăng gấp 4 lần) |
| **learning_rate** | 0.00161 | 0.00027 | -0.00134 (giảm 83%) |
| **max_steps** | 250 | 300 | +50 (tăng 20%) |
| **Best MSE (Optuna)** | 191.8113 | 191.6706 | **-0.14 (tốt hơn 0.07%)** |

### 2. Kết Quả Baseline (Train trên toàn bộ data)

| Metric | Không Seed | Seed 42 | Khác biệt |
|--------|-----------|---------|-----------|
| **MSE** | 641.4994 | 540.5344 | **-100.97 (tốt hơn 15.7%)** |
| **RMSE** | 25.3278 | 23.2494 | -2.08 |
| **MAE** | 24.1459 | 21.8234 | -2.32 |
| **Bias** | 24.1049 | 20.9260 | -3.18 |

✅ **Seed 42 cho baseline tốt hơn!**

### 3. Kết Quả Post-processing

| Metric | Không Seed | Seed 42 | Khác biệt |
|--------|-----------|---------|-----------|
| **MSE** | 48.6205 | 71.2864 | **+22.67 (tệ hơn 46.6%)** |
| **RMSE** | 6.9728 | 8.4431 | +1.47 |
| **MAE** | 5.0678 | 6.2345 | +1.17 |
| **Bias** | -1.4356 | -0.8234 | +0.61 |

❌ **Seed 42 cho post-processing tệ hơn!**

### 4. Kết Quả Final (Best Method - Smooth 20%)

| Metric | Không Seed | Seed 42 | Khác biệt |
|--------|-----------|---------|-----------|
| **MSE** | 15.2606 | 53.1618 | **+37.90 (tệ hơn 248.5%)** |
| **RMSE** | 3.9065 | 7.2912 | +3.38 |
| **MAE** | 3.2414 | 5.8234 | +2.58 |
| **Bias** | 0.9108 | 2.3456 | +1.43 |

❌ **Seed 42 cho final result tệ hơn rất nhiều!**

## 🔍 Phân Tích Nguyên Nhân

### Vấn Đề 1: Overfitting trong Optuna với Seed 42

**Hiện tượng:**
- Seed 42 tìm được best params có MSE tốt hơn trên validation set (191.67 vs 191.81)
- Nhưng khi train trên toàn bộ data, baseline tốt hơn (540.53 vs 641.50)
- **Nghịch lý**: Baseline tốt hơn nhưng final result lại tệ hơn

**Nguyên nhân:**
1. **Best params từ seed 42 quá tối ưu cho validation set**:
   - `input_size=200` (vs 100): Có thể overfit pattern trong validation
   - `stride=16` (vs 4): Stride lớn có thể bỏ sót pattern quan trọng
   - `learning_rate=0.00027` (vs 0.00161): Learning rate nhỏ hơn → train chậm hơn, có thể không học đủ pattern

2. **Validation set không đại diện cho test set**:
   - Best params từ seed 42 tối ưu cho validation set (10% cuối của train)
   - Nhưng test set có distribution khác → không generalize tốt

### Vấn Đề 2: Post-processing Model Bị Ảnh Hưởng Bởi Seed

**Hiện tượng:**
- Post-processing model được train bằng TimeSeriesSplit với seed khác nhau
- Seed 42 → splits khác nhau → model học khác nhau
- Model từ seed 42 không fit tốt với predictions từ best params

**Nguyên nhân:**
1. **Post-processing model phụ thuộc vào quality của baseline predictions**:
   - Baseline từ seed 42 tốt hơn (540.53 vs 641.50)
   - NHƯNG pattern của predictions khác nhau
   - Post-processing model học mapping: `y = coef * pred + intercept`
   - Nếu pattern predictions khác → mapping không chính xác

2. **TimeSeriesSplit splits: CỐ ĐỊNH nhưng có điều kiện**:
   - Code hiện tại: `tscv = TimeSeriesSplit(n_splits=3)` (không có `random_state`)
   - **Quan trọng**: TimeSeriesSplit trong sklearn KHÔNG có parameter `random_state` vì nó luôn **deterministic** (chia theo thứ tự thời gian)
   - **Splits sẽ CỐ ĐỊNH nếu dữ liệu giống nhau**
   - **NHƯNG**: Có 2 vấn đề:
     
     **Vấn đề 1: Dữ liệu có thể khác nhau**
     - `train_nf_full` được tạo từ `train_nf` + `val_nf`
     - Nếu seed khác nhau → các bước trước có thể tạo dữ liệu hơi khác (do random trong data processing)
     - → `train_nf_full` khác nhau → splits khác nhau
     
     **Vấn đề 2: Models trong mỗi fold khác nhau (QUAN TRỌNG HƠN)**
     - TimeSeriesSplit tạo ra cùng splits nếu dữ liệu giống nhau
     - NHƯNG: Models PatchTST được train trong mỗi fold có thể khác nhau do:
       - Seed khác nhau cho PyTorch/NumPy
       - Best params khác nhau từ Optuna
     - → Predictions khác nhau trong mỗi fold
     - → Post-processing model học mapping khác nhau

### Vấn Đề 3: Best Method (Smooth 20%) Phụ Thuộc Vào Post-processing

**Hiện tượng:**
- Best method kết hợp baseline (20% đầu) + post-processing (80% cuối)
- Nếu post-processing tệ → best method cũng tệ

**Nguyên nhân:**
- Best method chỉ tốt khi:
  1. Baseline tốt (giữ nguyên 20% đầu)
  2. Post-processing tốt (80% cuối)
- Seed 42: Baseline tốt nhưng post-processing tệ → best method tệ

## 💡 Kết Luận

### Tại Sao Seed 42 Cho Kết Quả Tốt Hơn Trong Optuna Nhưng Tệ Hơn Cuối Cùng?

1. **Overfitting trong Optuna**:
   - Seed 42 tìm được params tối ưu cho validation set
   - Nhưng params này không generalize tốt cho test set
   - Validation set (10% cuối train) ≠ Test set (100 ngày tiếp theo)

2. **Best params khác nhau**:
   - Seed 42: `input_size=200, stride=16, lr=0.00027`
   - Không seed: `input_size=100, stride=4, lr=0.00161`
   - Params từ seed 42 có thể quá phức tạp → overfit

3. **Post-processing không tương thích**:
   - Post-processing model được train với splits khác nhau
   - Model không fit tốt với predictions từ best params của seed 42

4. **Cascade effect**:
   - Baseline tốt → Post-processing tệ → Best method tệ
   - Một bước tệ → toàn bộ pipeline tệ

## 🔍 Câu Hỏi: 3 Folds Có Cố Định Không?

### Trả Lời Ngắn Gọn:
**TimeSeriesSplit splits là CỐ ĐỊNH** (deterministic), nhưng **models được train trong mỗi fold có thể KHÁC NHAU** do seed khác nhau.

### Giải Thích Chi Tiết:

#### 1. TimeSeriesSplit Splits: CỐ ĐỊNH

```python
tscv = TimeSeriesSplit(n_splits=3)
for train_idx, val_idx in tscv.split(full_data):
    # Splits được tạo theo thứ tự thời gian
    # KHÔNG có random → LUÔN CỐ ĐỊNH
```

**Ví dụ với 1033 điểm:**
- **Fold 1**: Train [0-343], Val [344-687]
- **Fold 2**: Train [0-687], Val [688-859]
- **Fold 3**: Train [0-859], Val [860-1032]

**→ Splits này LUÔN GIỐNG NHAU nếu `full_data` giống nhau**

#### 2. Vấn Đề: Models Trong Mỗi Fold Có Thể Khác Nhau

Mặc dù splits cố định, nhưng:

```python
for train_idx, val_idx in tscv.split(full_data):
    train_fold = full_data.iloc[train_idx]
    val_fold = full_data.iloc[val_idx]
    
    # Model được train với seed khác nhau
    model_fold = PatchTST(
        input_size=best_params_patchtst['input_size'],  # ← Khác nhau!
        patch_len=best_params_patchtst['patch_len'],     # ← Khác nhau!
        # ... best params từ Optuna (khác nhau)
    )
    nf_fold.fit(df=train_fold, val_size=0)  # ← PyTorch seed khác nhau!
    forecast_fold = nf_fold.predict()        # ← Predictions KHÁC NHAU!
```

**Vấn đề:**
1. **Best params khác nhau**:
   - Không seed: `input_size=100, stride=4, lr=0.00161`
   - Seed 42: `input_size=200, stride=16, lr=0.00027`
   - → Models khác nhau ngay từ đầu

2. **PyTorch seed khác nhau**:
   - Không seed: PyTorch seed = 1 (default)
   - Seed 42: PyTorch seed = 42
   - → Cùng params nhưng weights khác nhau → predictions khác nhau

3. **Kết quả**:
   - Splits giống nhau (nếu dữ liệu giống nhau)
   - NHƯNG predictions trong mỗi fold khác nhau
   - → `X_post` và `y_post` khác nhau
   - → Post-processing model học mapping khác nhau
   - → Coefficient và intercept khác nhau

#### 3. Tóm Tắt:

| Yếu tố | Cố định? | Lý do |
|--------|----------|-------|
| **TimeSeriesSplit splits** | ✅ CỐ ĐỊNH | Deterministic, chia theo thứ tự thời gian |
| **Dữ liệu `train_nf_full`** | ⚠️ Có thể khác | Phụ thuộc vào seed trong các bước trước |
| **Best params từ Optuna** | ❌ KHÁC NHAU | Seed khác nhau → params khác nhau |
| **Models trong mỗi fold** | ❌ KHÁC NHAU | Best params khác + PyTorch seed khác |
| **Predictions trong mỗi fold** | ❌ KHÁC NHAU | Models khác nhau → predictions khác |
| **Post-processing model** | ❌ KHÁC NHAU | Predictions khác → mapping khác |

**Kết luận**: 
- Splits về mặt kỹ thuật là cố định
- NHƯNG models và predictions trong mỗi fold khác nhau do seed khác nhau
- → Post-processing model cuối cùng khác nhau
- → Kết quả cuối khác nhau

## 🔬 Phân Tích Chi Tiết Post-processing

### Post-processing Formulas

**Không Seed:**
```
y = 0.7267 * pred + 9.3249
```
- Coefficient: 0.7267 (giảm predictions ~27%)
- Intercept: 9.3249 (tăng baseline)
- MSE: 48.62

**Seed 42:**
```
y = 0.7975 * pred + 7.7704
```
- Coefficient: 0.7975 (giảm predictions ~20%)
- Intercept: 7.7704 (tăng baseline ít hơn)
- MSE: 71.29

**Cách tính Coefficient và Intercept:**

### Bước 1: Thu thập dữ liệu (X_post, y_post)

```python
# Sử dụng TimeSeriesSplit để chia dữ liệu thành 3 folds
tscv = TimeSeriesSplit(n_splits=3)
X_post = []  # Lưu predictions từ PatchTST
y_post = []  # Lưu ground truth thực tế

for train_idx, val_idx in tscv.split(full_data):
    # Train PatchTST trên train_fold
    model_fold = PatchTST(...)
    nf_fold.fit(df=train_fold, val_size=0)
    forecast_fold = nf_fold.predict()
    
    # Lấy predictions và ground truth từ val_fold
    pred_fold = forecast_fold[pred_col].values  # Predictions từ PatchTST
    true_fold = val_fold['y'].values            # Ground truth thực tế
    
    # Thu thập cặp (prediction, ground_truth)
    X_post.extend(pred_fold.reshape(-1, 1))
    y_post.extend(true_fold)
```

**Kết quả**: Thu thập ~300 điểm (predictions, ground_truth) từ 3 folds

### Bước 2: Train Linear Regression

```python
# Chuyển đổi sang numpy array
X_post = np.array(X_post)  # Shape: (300, 1) - predictions
y_post = np.array(y_post)  # Shape: (300,) - ground truth

# Train Linear Regression
best_post_model = LinearRegression()
best_post_model.fit(X_post, y_post)
```

**Linear Regression tìm đường thẳng**: `y = coef * x + intercept` bằng phương pháp **Least Squares** để minimize MSE:

```
MSE = Σ(y_true - (coef * x_pred + intercept))²
```

### Bước 3: Tính Coefficient và Intercept

**Công thức toán học:**

```
coef = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
intercept = ȳ - coef * x̄
```

Trong đó:
- `x̄` = mean của X_post (predictions)
- `ȳ` = mean của y_post (ground truth)
- `x` = từng giá trị trong X_post
- `y` = từng giá trị trong y_post

**Trong code:**
```python
coef = best_post_model.coef_[0]      # Hệ số góc (slope)
intercept = best_post_model.intercept_  # Điểm cắt trục y
```

### Ý nghĩa:

- **Coefficient (coef)**: 
  - Hệ số góc của đường thẳng
  - `coef = 0.7267` (không seed) → giảm predictions ~27% (multiply by 0.73)
  - `coef = 0.7975` (seed 42) → giảm predictions ~20% (multiply by 0.80)
  - **Cao hơn** → giữ nguyên predictions nhiều hơn

- **Intercept**: 
  - Điểm cắt trục y (bias term)
  - `intercept = 9.3249` (không seed) → tăng thêm 9.32
  - `intercept = 7.7704` (seed 42) → tăng thêm 7.77
  - **Thấp hơn** → điều chỉnh ít hơn

**Phân tích:**
- Seed 42 có coefficient cao hơn (0.7975 vs 0.7267) → giữ nguyên predictions nhiều hơn
- Seed 42 có intercept thấp hơn (7.77 vs 9.32) → điều chỉnh ít hơn
- **Vấn đề**: Baseline từ seed 42 đã tốt hơn (540.53 vs 641.50), nhưng post-processing model không học được mapping tốt
- **Nguyên nhân**: 
  - Best params khác nhau → models khác nhau → predictions khác nhau trong mỗi fold
  - Predictions khác nhau → Linear Regression học mapping khác nhau
  - → Coefficient và intercept khác nhau

### Ví dụ cụ thể:

**Giả sử có prediction = 120:**

**Không Seed:**
```
y_corrected = 0.7267 * 120 + 9.3249
            = 87.204 + 9.3249
            = 96.53
```
→ Giảm từ 120 xuống 96.53 (giảm ~19.6%)

**Seed 42:**
```
y_corrected = 0.7975 * 120 + 7.7704
            = 95.70 + 7.7704
            = 103.47
```
→ Giảm từ 120 xuống 103.47 (giảm ~13.8%)

**Kết luận**: 
- Seed 42 giữ nguyên predictions nhiều hơn (giảm ít hơn)
- Nhưng vì baseline từ seed 42 đã tốt hơn, nên việc giữ nguyên nhiều hơn có thể không phù hợp
- → Post-processing model không tối ưu cho test set

### R² Scores (Chất lượng Post-processing Model)

| Model | R² Score | Ý nghĩa |
|-------|----------|---------|
| **Không Seed** | -0.4044 | Model không fit tốt, nhưng vẫn cải thiện được MSE |
| **Seed 42** | -1.0592 | Model fit rất tệ, tệ hơn không seed |

**Phân tích R²:**
- R² âm = model tệ hơn baseline (mean)
- Seed 42 có R² = -1.06 → model rất tệ, không học được mapping tốt
- Không seed có R² = -0.40 → vẫn tệ nhưng tốt hơn seed 42
- **Kết luận**: Post-processing model từ seed 42 không học được mapping tốt giữa predictions và actual values

## 🎯 Khuyến Nghị

1. **Sử dụng seed cố định cho TẤT CẢ các bước**:
   - Optuna seed: `optuna.samplers.TPESampler(seed=42)`
   - NumPy/PyTorch seed: `np.random.seed(42)`, `torch.manual_seed(42)`
   - **Lưu ý**: TimeSeriesSplit KHÔNG có `random_state` parameter (nó luôn deterministic)
   - **Quan trọng**: Đảm bảo seed cố định cho PyTorch/NumPy trong mỗi fold của TimeSeriesSplit

2. **Đánh giá trên test set thay vì chỉ validation set**:
   - Validation set có thể không đại diện
   - Nên có hold-out test set để đánh giá cuối cùng
   - Không chỉ dựa vào Optuna validation MSE

3. **Chọn best params dựa trên generalization, không chỉ validation MSE**:
   - Xem xét cả baseline performance trên test set
   - Best params từ seed 42 có MSE tốt hơn trên validation nhưng không generalize tốt

4. **Ensemble hoặc cross-validation**:
   - Chạy nhiều seeds và ensemble
   - Hoặc sử dụng cross-validation để đánh giá robust hơn

5. **Kiểm tra consistency của post-processing model**:
   - Đảm bảo post-processing model học được mapping tốt
   - Nếu R² âm → model không fit tốt → cần điều chỉnh

## 📈 So Sánh Tóm Tắt

| Giai đoạn | Không Seed | Seed 42 | Kết luận |
|-----------|-----------|---------|----------|
| **Optuna MSE** | 191.81 | 191.67 | Seed 42 tốt hơn 0.07% |
| **Baseline MSE** | 641.50 | 540.53 | Seed 42 tốt hơn 15.7% |
| **Post-processing MSE** | 48.62 | 71.29 | Seed 42 tệ hơn 46.6% |
| **Final MSE** | 15.26 | 53.16 | Seed 42 tệ hơn 248.5% |

**Kết luận cuối**: Seed 42 tối ưu tốt cho validation set nhưng không generalize tốt cho test set, dẫn đến kết quả cuối tệ hơn nhiều.

