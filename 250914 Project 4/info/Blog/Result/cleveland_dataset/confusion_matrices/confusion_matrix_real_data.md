# Confusion Matrix Thuc Te - Cleveland Heart Disease Dataset

## Tong Quan
- **Dataset**: Cleveland Heart Disease Dataset
- **Binary Classification**: 0 = No Heart Disease, 1 = Heart Disease
- **Models**: 12 models với 3 scalers mỗi model (36 combinations)
- **Scalers**: MinMaxScaler, RobustScaler, StandardScaler

## Adaboost (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      14       3
Actual      1       3      11
```

**Metrics:**
- **True Negatives (TN)**: 14
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 3
- **True Positives (TP)**: 11
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7857
- **Recall**: 0.7857
- **F1-Score**: 0.7857

## Adaboost (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      14       3
Actual      1       3      11
```

**Metrics:**
- **True Negatives (TN)**: 14
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 3
- **True Positives (TP)**: 11
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7857
- **Recall**: 0.7857
- **F1-Score**: 0.7857

## Adaboost (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      14       3
Actual      1       3      11
```

**Metrics:**
- **True Negatives (TN)**: 14
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 3
- **True Positives (TP)**: 11
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7857
- **Recall**: 0.7857
- **F1-Score**: 0.7857

## Catboost (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      16       1
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 16
- **False Positives (FP)**: 1
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.9355 (93.55%)
- **Precision**: 0.9286
- **Recall**: 0.9286
- **F1-Score**: 0.9286

## Catboost (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      16       1
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 16
- **False Positives (FP)**: 1
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.9355 (93.55%)
- **Precision**: 0.9286
- **Recall**: 0.9286
- **F1-Score**: 0.9286

## Catboost (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      16       1
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 16
- **False Positives (FP)**: 1
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.9355 (93.55%)
- **Precision**: 0.9286
- **Recall**: 0.9286
- **F1-Score**: 0.9286

## Decision Tree (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      11       6
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 11
- **False Positives (FP)**: 6
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.7419 (74.19%)
- **Precision**: 0.6667
- **Recall**: 0.8571
- **F1-Score**: 0.7500

## Decision Tree (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      11       6
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 11
- **False Positives (FP)**: 6
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.7419 (74.19%)
- **Precision**: 0.6667
- **Recall**: 0.8571
- **F1-Score**: 0.7500

## Decision Tree (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      11       6
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 11
- **False Positives (FP)**: 6
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.7419 (74.19%)
- **Precision**: 0.6667
- **Recall**: 0.8571
- **F1-Score**: 0.7500

## Gradient Boosting (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      14       3
Actual      1       3      11
```

**Metrics:**
- **True Negatives (TN)**: 14
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 3
- **True Positives (TP)**: 11
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7857
- **Recall**: 0.7857
- **F1-Score**: 0.7857

## Gradient Boosting (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.7647
- **Recall**: 0.9286
- **F1-Score**: 0.8387

## Gradient Boosting (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      12       5
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 12
- **False Positives (FP)**: 5
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7222
- **Recall**: 0.9286
- **F1-Score**: 0.8125

## Knn (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       4      10
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 4
- **True Positives (TP)**: 10
- **Accuracy**: 0.7419 (74.19%)
- **Precision**: 0.7143
- **Recall**: 0.7143
- **F1-Score**: 0.7143

## Knn (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7500
- **Recall**: 0.8571
- **F1-Score**: 0.8000

## Knn (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       0      14
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 0
- **True Positives (TP)**: 14
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.7778
- **Recall**: 1.0000
- **F1-Score**: 0.8750

## Lightgbm (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       4      10
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 4
- **True Positives (TP)**: 10
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.8333
- **Recall**: 0.7143
- **F1-Score**: 0.7692

## Lightgbm (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       4      10
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 4
- **True Positives (TP)**: 10
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.8333
- **Recall**: 0.7143
- **F1-Score**: 0.7692

## Lightgbm (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       4      10
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 4
- **True Positives (TP)**: 10
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.8333
- **Recall**: 0.7143
- **F1-Score**: 0.7692

## Logistic Regression (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7500
- **Recall**: 0.8571
- **F1-Score**: 0.8000

## Logistic Regression (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7500
- **Recall**: 0.8571
- **F1-Score**: 0.8000

## Logistic Regression (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8065 (80.65%)
- **Precision**: 0.7500
- **Recall**: 0.8571
- **F1-Score**: 0.8000

## Naive Bayes (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.7647
- **Recall**: 0.9286
- **F1-Score**: 0.8387

## Naive Bayes (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.7647
- **Recall**: 0.9286
- **F1-Score**: 0.8387

## Naive Bayes (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      13       4
Actual      1       1      13
```

**Metrics:**
- **True Negatives (TN)**: 13
- **False Positives (FP)**: 4
- **False Negatives (FN)**: 1
- **True Positives (TP)**: 13
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.7647
- **Recall**: 0.9286
- **F1-Score**: 0.8387

## Random Forest (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Random Forest (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Random Forest (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Stacking Ensemble Logistic Regression (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       3      11
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 3
- **True Positives (TP)**: 11
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.8462
- **Recall**: 0.7857
- **F1-Score**: 0.8148

## Stacking Ensemble Logistic Regression (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Stacking Ensemble Logistic Regression (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Svm (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      17       0
Actual      1      14       0
```

**Metrics:**
- **True Negatives (TN)**: 17
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 14
- **True Positives (TP)**: 0
- **Accuracy**: 0.5484 (54.84%)
- **Precision**: 0.0000
- **Recall**: 0.0000
- **F1-Score**: 0.0000

## Svm (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      16       1
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 16
- **False Positives (FP)**: 1
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.9032 (90.32%)
- **Precision**: 0.9231
- **Recall**: 0.8571
- **F1-Score**: 0.8889

## Svm (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      14       3
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 14
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.8000
- **Recall**: 0.8571
- **F1-Score**: 0.8276

## Voting Ensemble Hard (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       3      11
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 3
- **True Positives (TP)**: 11
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.8462
- **Recall**: 0.7857
- **F1-Score**: 0.8148

## Voting Ensemble Hard (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Voting Ensemble Hard (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      14       3
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 14
- **False Positives (FP)**: 3
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8387 (83.87%)
- **Precision**: 0.8000
- **Recall**: 0.8571
- **F1-Score**: 0.8276

## Xgboost (MinMaxScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Xgboost (RobustScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

## Xgboost (StandardScaler)

**Confusion Matrix:**
```
                Predicted
                     0       1
Actual      0      15       2
Actual      1       2      12
```

**Metrics:**
- **True Negatives (TN)**: 15
- **False Positives (FP)**: 2
- **False Negatives (FN)**: 2
- **True Positives (TP)**: 12
- **Accuracy**: 0.8710 (87.10%)
- **Precision**: 0.8571
- **Recall**: 0.8571
- **F1-Score**: 0.8571

---

## Tong Ket va Phan Tich

### Bang Tong Ket Theo Models

| Model | MinMaxScaler | RobustScaler | StandardScaler | Best Scaler |
|-------|--------------|--------------|----------------|-------------|
| Adaboost | 0.806 | 0.806 | 0.806 | MinMaxScaler |
| Catboost | 0.935 | 0.935 | 0.935 | MinMaxScaler |
| Decision Tree | 0.742 | 0.742 | 0.742 | MinMaxScaler |
| Gradient Boosting | 0.806 | 0.839 | 0.806 | RobustScaler |
| Knn | 0.742 | 0.806 | 0.871 | StandardScaler |
| Lightgbm | 0.806 | 0.806 | 0.806 | MinMaxScaler |
| Logistic Regression | 0.806 | 0.806 | 0.806 | MinMaxScaler |
| Naive Bayes | 0.839 | 0.839 | 0.839 | MinMaxScaler |
| Random Forest | 0.871 | 0.871 | 0.871 | MinMaxScaler |
| Stacking Ensemble Logistic Regression | 0.839 | 0.871 | 0.871 | RobustScaler |
| Svm | 0.548 | 0.903 | 0.839 | RobustScaler |
| Voting Ensemble Hard | 0.839 | 0.871 | 0.839 | RobustScaler |
| Xgboost | 0.871 | 0.871 | 0.871 | MinMaxScaler |

### Top Performers

| Rank | Model | Scaler | Accuracy |
|------|-------|--------|----------|
| 1 | Catboost | StandardScaler | 0.9355 (93.55%) |
| 2 | Catboost | RobustScaler | 0.9355 (93.55%) |
| 3 | Catboost | MinMaxScaler | 0.9355 (93.55%) |
| 4 | Svm | RobustScaler | 0.9032 (90.32%) |
| 5 | Xgboost | StandardScaler | 0.8710 (87.10%) |
| 6 | Xgboost | RobustScaler | 0.8710 (87.10%) |
| 7 | Xgboost | MinMaxScaler | 0.8710 (87.10%) |
| 8 | Voting Ensemble Hard | RobustScaler | 0.8710 (87.10%) |
| 9 | Stacking Ensemble Logistic Regression | StandardScaler | 0.8710 (87.10%) |
| 10 | Stacking Ensemble Logistic Regression | RobustScaler | 0.8710 (87.10%) |
