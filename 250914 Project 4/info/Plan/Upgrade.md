1. Random Forest Baseline & Data Optimization
Mục tiêu: Xây dựng baseline mạnh với RF, tối đa hóa Accuracy & Macro-F1, và làm sạch dữ liệu.
Nội dung:
Tiền xử lý nâng cao: lemmatization, rare words filter, stopwords ngữ cảnh, phrase detection.
Remove duplicates/near-duplicates, stratified split.
Vector hóa: BoW/TF-IDF (RAPIDS cuML GPU) + Embeddings (sentence-transformers CUDA).
Train RF (cuML GPU): n_estimators, max_depth, max_features, class_weight.
Threshold tuning để tối ưu Macro-F1.
Đầu ra: 3 model RF + confusion matrix + metrics.json (Accuracy, Macro-F1, CM).

2. AdaBoost Rescue & Alternatives
Mục tiêu: Cải thiện AdaBoost vốn yếu, đảm bảo Macro-F1 cao hơn baseline.
Nội dung:
Grid search n_estimators, learning_rate, base learner (stump vs depth=2).
So sánh SAMME vs SAMME.R.
Data optimization: cleaning, n-gram (1–2), min_df/max_df tuning.
Threshold tuning để tăng Macro-F1.
Nếu cần GPU: so sánh AdaBoost CPU vs thay thế bằng XGBoost/LightGBM GPU.
Đầu ra: Báo cáo cấu hình tốt nhất, Macro-F1 ≥ +5pp so baseline; metrics.json, CM.

3. Gradient Boosting (GBDT) + Feature Fusion
Mục tiêu: Nâng hiệu năng GBDT, tận dụng Fusion để tăng Macro-F1.
Nội dung:
Thay sklearn GBDT bằng XGBoost/LightGBM GPU.
Feature fusion: concat TF-IDF + Embeddings (chuẩn hóa).
Thêm feature phụ: text length, digit ratio, uppercase ratio.
Tuning: learning_rate, n_estimators, subsample, max_depth.
Early stopping trên dev; threshold tuning cho Macro-F1.
Đầu ra: GBDT tốt nhất (Accuracy/Macro-F1 ≥ baseline), SHAP summary, metrics.json, CM.

4. XGBoost Optimization & Interpretability
Mục tiêu: Tối ưu XGBoost trên GPU, giải thích mô hình rõ ràng.
Nội dung:
GPU flags: tree_method="gpu_hist", predictor="gpu_predictor".
Hyperparam tune: eta, max_depth, subsample, colsample_bytree, lambda, alpha.
Early stopping, threshold tuning.
Data optimization: fusion, feature phụ (length, digit_ratio).
Interpretability: SHAP global/local, PDP/ICE.
Đầu ra: XGB Embeddings (GPU) với Accuracy ≥ 87.5%, Macro-F1 tối đa; SHAP/PDP plots; metrics.json, CM.

5. LightGBM Enhancement & Streamlit Integration
Mục tiêu: Giữ LightGBM là top model, ổn định xác suất, và tích hợp toàn bộ vào Streamlit UI.
Nội dung:
GPU build (device_type="gpu", boosting_type="goss"/"gbdt").
Tuning: num_leaves, learning_rate, feature_fraction, bagging_fraction.
Calibration (Platt/Isotonic).
Threshold tuning để tối ưu Macro-F1.
UI Integration:
Tabs: Predict/Evaluate/Explain/Models/Settings.
Chọn model/vectorizer, batch eval, confusion matrix, SHAP/PDP.
Caching model/vectorizer (st.cache_resource).
Dockerfile + README.
Đầu ra: LGBM Embeddings (GPU) Accuracy ≥ 88%, Macro-F1 cao; calibrated model + Streamlit app full feature.