# 🚀 Quick Start Guide - Advanced LightGBM Optimization

## ⚡ Cài đặt nhanh (5 phút)

### 1. Cài đặt và kiểm tra
```bash
# Chạy script cài đặt tự động
python install_and_test.py
```

### 2. Chạy demo nhanh
```bash
# Demo tất cả tính năng
python demo.py

# Hoặc chạy optimization nhanh
python run_optimization.py --quick
```

### 3. Chạy pipeline đầy đủ
```bash
# Pipeline hoàn chỉnh
python main.py

# Hoặc với tham số tùy chỉnh
python run_optimization.py --dataset fe --trials 100 --gpu
```

## 🎯 Các lệnh chính

### Chạy optimization
```bash
# Quick mode (khuyến nghị cho lần đầu)
python run_optimization.py --quick

# Full mode với GPU
python run_optimization.py --mode full --gpu --trials 200

# Demo mode (ít tham số)
python run_optimization.py --mode demo
```

### Chạy với dataset khác
```bash
# Sử dụng dataset khác
python run_optimization.py --dataset raw --quick
python run_optimization.py --dataset dt --quick
python run_optimization.py --dataset fe_dt --quick
```

### Tùy chỉnh output
```bash
# Lưu kết quả vào thư mục khác
python run_optimization.py --output-dir my_results --quick

# Bỏ qua ensemble methods
python run_optimization.py --no-ensemble --quick
```

## 📊 Kết quả mong đợi

### Performance cải thiện
- **Accuracy**: 85-90% (vs baseline 83.87%)
- **F1-Score**: 84-89% (vs baseline 82.76%)
- **AUC-ROC**: 93-96% (vs baseline 92.02%)

### Files được tạo
```
results/
├── advanced_lightgbm_model.txt     # Model đã train
├── ensemble_models/                # Các ensemble models
├── evaluation_report.txt           # Báo cáo đánh giá
├── plots/                         # Các biểu đồ
└── results_summary.json           # Tóm tắt kết quả
```

## 🔧 Troubleshooting

### Lỗi thường gặp

1. **GPU không hoạt động**
   ```bash
   # Cài đặt LightGBM với GPU support
   pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
   ```

2. **Thiếu dependencies**
   ```bash
   # Cài đặt lại requirements
   pip install -r requirements.txt
   ```

3. **Memory không đủ**
   ```bash
   # Chạy với ít trials hơn
   python run_optimization.py --trials 20 --quick
   ```

### Kiểm tra hệ thống
```bash
# Kiểm tra Python version
python --version  # Cần >= 3.8

# Kiểm tra GPU
nvidia-smi  # Nếu có GPU

# Kiểm tra dependencies
python install_and_test.py
```

## 📈 So sánh với baseline

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | 83.87% | 85-90% | +1-6% |
| F1-Score | 82.76% | 84-89% | +1-6% |
| AUC-ROC | 92.02% | 93-96% | +1-4% |

## 🎭 Các tính năng chính

### 1. Hyperparameter Optimization
- Optuna với TPE sampler
- Bayesian optimization
- Multi-objective optimization
- Advanced pruning

### 2. Feature Engineering
- Polynomial features
- Statistical features
- Target encoding
- Feature selection

### 3. Ensemble Methods
- Voting Classifier
- Stacking Classifier
- Blending Ensemble
- Weighted Ensemble

### 4. Model Interpretability
- SHAP analysis
- Feature importance
- Waterfall plots
- Summary plots

## 🚀 Tips để có kết quả tốt nhất

1. **Sử dụng GPU** nếu có
2. **Tăng số trials** cho optimization tốt hơn
3. **Chạy full pipeline** thay vì quick mode
4. **Kiểm tra feature importance** để hiểu model
5. **So sánh ensemble methods** để chọn tốt nhất

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Chạy `python install_and_test.py` để kiểm tra
2. Xem log trong `results/logs/`
3. Kiểm tra `results/evaluation_report.txt`
4. Thử chạy `python demo.py` trước

---

**Chúc bạn thành công! 🎉**
