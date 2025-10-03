# Detailed SHAP Values Analysis - Heart Disease Dataset

## Executive Summary
- **Models Analyzed**: 43
- **Total Samples**: 870
- **Unique Configurations**: 43
- **Features**: 13

## Global Feature Importance Ranking

| Rank | Feature | Avg Importance | Std Dev | Models Count |
|------|---------|----------------|---------|--------------|
| 1 | **feature_22** | 0.6141 | 0.7482 | 15 |
| 2 | **feature_29** | 0.5253 | 0.6822 | 15 |
| 3 | **feature_16** | 0.5117 | 0.4293 | 15 |
| 4 | **ca** | 0.5105 | 1.1355 | 43 |
| 5 | **feature_26** | 0.4756 | 0.5284 | 15 |
| 6 | **feature_15** | 0.4370 | 0.4612 | 15 |
| 7 | **feature_25** | 0.4340 | 0.4579 | 15 |
| 8 | **thal** | 0.4225 | 0.9751 | 43 |
| 9 | **cp** | 0.4024 | 0.9777 | 43 |
| 10 | **feature_19** | 0.3711 | 0.2980 | 15 |
| 11 | **feature_24** | 0.3578 | 0.3604 | 15 |
| 12 | **feature_14** | 0.3577 | 0.3962 | 15 |
| 13 | **feature_23** | 0.3483 | 0.3646 | 15 |
| 14 | **feature_20** | 0.3337 | 0.2685 | 15 |
| 15 | **oldpeak** | 0.3292 | 0.6635 | 43 |

## Individual Model SHAP Analysis

### 1. xgboost_RobustScaler (02d0f3aeaed2da6453a6fec8d74e9165.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.xgboost_model.XGBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 02d0f3aeaed2da6453a6fec8d74e9165.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0004
- **Standard Deviation**: 0.0619
- **Range**: -0.1722 to 0.2158
- **Median**: 0.0012
- **Mean Absolute Value**: 0.0454

**Top 5 Most Important Features:**
1. **ca**: 0.1277 ↓ (decrease)
   - Mean SHAP: -0.0080, Std: 0.1307
2. **thal**: 0.0816 ↓ (decrease)
   - Mean SHAP: -0.0046, Std: 0.0920
3. **cp**: 0.0792 ↑ (increase)
   - Mean SHAP: 0.0214, Std: 0.0876
4. **sex**: 0.0622 ↓ (decrease)
   - Mean SHAP: -0.0013, Std: 0.0699
5. **slope**: 0.0467 ↓ (decrease)
   - Mean SHAP: -0.0103, Std: 0.0494

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5054
- **Max Influence**: 0.1178
- **Feature Complexity**: 0.0392
**Top Influential Features:**
- age: 0.0385 🔼
- sex: 0.0399 🔼
- ca: 0.0859 🔼
- cp: 0.1166 🔼
- thal: 0.1178 🔼

#### Sample 1:
- **Prediction Sum**: -0.4388
- **Max Influence**: 0.1511
- **Feature Complexity**: 0.0394
**Top Influential Features:**
- thalach: -0.0394 🔽
- slope: -0.0536 🔽
- cp: -0.0669 🔽
- thal: -0.0692 🔽
- ca: -0.1511 🔽

#### Sample 2:
- **Prediction Sum**: 0.4710
- **Max Influence**: 0.1193
- **Feature Complexity**: 0.0389
**Top Influential Features:**
- chol: 0.0566 🔼
- slope: 0.0605 🔼
- sex: 0.0613 🔼
- ca: 0.1181 🔼
- thal: 0.1193 🔼

---

### 2. knn_StandardScaler (234c375a095d70543556e6f1cc39d741.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.knn_model.KNNModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 234c375a095d70543556e6f1cc39d741.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0005
- **Standard Deviation**: 0.0704
- **Range**: -0.1900 to 0.2925
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0518

**Top 5 Most Important Features:**
1. **ca**: 0.1070 ↑ (increase)
   - Mean SHAP: 0.0063, Std: 0.1259
2. **sex**: 0.0923 ↑ (increase)
   - Mean SHAP: 0.0174, Std: 0.0984
3. **thal**: 0.0906 ↓ (decrease)
   - Mean SHAP: -0.0018, Std: 0.0996
4. **cp**: 0.0628 ↑ (increase)
   - Mean SHAP: 0.0254, Std: 0.0804
5. **oldpeak**: 0.0619 ↓ (decrease)
   - Mean SHAP: -0.0271, Std: 0.0658

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4600
- **Max Influence**: 0.1700
- **Feature Complexity**: 0.0493
**Top Influential Features:**
- thalach: 0.0300 🔼
- sex: 0.0650 🔼
- cp: 0.0825 🔼
- thal: 0.1075 🔼
- ca: 0.1700 🔼

#### Sample 1:
- **Prediction Sum**: -0.5400
- **Max Influence**: 0.1900
- **Feature Complexity**: 0.0541
**Top Influential Features:**
- thal: -0.0675 🔽
- age: -0.0700 🔽
- slope: -0.0800 🔽
- ca: -0.1225 🔽
- cp: -0.1900 🔽

#### Sample 2:
- **Prediction Sum**: 0.4600
- **Max Influence**: 0.2300
- **Feature Complexity**: 0.0622
**Top Influential Features:**
- cp: 0.0300 🔼
- restecg: -0.0900 🔽
- sex: 0.0925 🔼
- thal: 0.1125 🔼
- ca: 0.2300 🔼

---

### 3. lightgbm_RobustScaler (2d732449090971a49e6125301271454f.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.lightgbm_model.LightGBMModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 2d732449090971a49e6125301271454f.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0000
- **Standard Deviation**: 0.4195
- **Range**: -0.6802 to 0.6802
- **Median**: 0.0000
- **Mean Absolute Value**: 0.4084

**Top 5 Most Important Features:**
1. **feature_16**: 0.6802 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.6802
2. **chol**: 0.5836 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5836
3. **feature_19**: 0.5353 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5353
4. **feature_25**: 0.5235 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5235
5. **feature_20**: 0.5217 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5217

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4064
- **Max Influence**: 0.6802
- **Feature Complexity**: 0.0957
**Top Influential Features:**
- feature_20: -0.5217 🔽
- feature_25: 0.5235 🔼
- feature_19: -0.5353 🔽
- chol: -0.5836 🔽
- feature_16: -0.6802 🔽

#### Sample 1:
- **Prediction Sum**: -0.4064
- **Max Influence**: 0.6802
- **Feature Complexity**: 0.0957
**Top Influential Features:**
- feature_20: 0.5217 🔼
- feature_25: -0.5235 🔽
- feature_19: 0.5353 🔼
- chol: 0.5836 🔼
- feature_16: 0.6802 🔼

---

### 4. xgboost_MinMaxScaler (2db4b0d0dbe854b39046abc9b0f75143.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.xgboost_model.XGBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 2db4b0d0dbe854b39046abc9b0f75143.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0004
- **Standard Deviation**: 0.0616
- **Range**: -0.1763 to 0.2013
- **Median**: 0.0013
- **Mean Absolute Value**: 0.0453

**Top 5 Most Important Features:**
1. **ca**: 0.1292 ↓ (decrease)
   - Mean SHAP: -0.0047, Std: 0.1327
2. **cp**: 0.0805 ↑ (increase)
   - Mean SHAP: 0.0193, Std: 0.0880
3. **thal**: 0.0792 ↓ (decrease)
   - Mean SHAP: -0.0040, Std: 0.0893
4. **sex**: 0.0615 ↑ (increase)
   - Mean SHAP: 0.0006, Std: 0.0691
5. **slope**: 0.0451 ↓ (decrease)
   - Mean SHAP: -0.0104, Std: 0.0478

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5054
- **Max Influence**: 0.1167
- **Feature Complexity**: 0.0378
**Top Influential Features:**
- sex: 0.0454 🔼
- oldpeak: 0.0465 🔼
- thal: 0.0890 🔼
- cp: 0.1066 🔼
- ca: 0.1167 🔼

#### Sample 1:
- **Prediction Sum**: -0.4388
- **Max Influence**: 0.1575
- **Feature Complexity**: 0.0394
**Top Influential Features:**
- slope: -0.0428 🔽
- thalach: -0.0490 🔽
- thal: -0.0542 🔽
- cp: -0.0627 🔽
- ca: -0.1575 🔽

#### Sample 2:
- **Prediction Sum**: 0.4710
- **Max Influence**: 0.1410
- **Feature Complexity**: 0.0414
**Top Influential Features:**
- sex: 0.0472 🔼
- slope: 0.0515 🔼
- chol: 0.0545 🔼
- ca: 0.1143 🔼
- thal: 0.1410 🔼

---

### 5. lightgbm_MinMaxScaler (361ef88296d27af48d2d47b06162118d.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.lightgbm_model.LightGBMModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 361ef88296d27af48d2d47b06162118d.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 1.1093
- **Range**: -2.0811 to 2.0811
- **Median**: 0.0000
- **Mean Absolute Value**: 1.0368

**Top 5 Most Important Features:**
1. **feature_22**: 2.0811 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 2.0811
2. **feature_29**: 1.9261 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.9261
3. **age**: 1.6733 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.6733
4. **feature_26**: 1.4829 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.4829
5. **feature_15**: 1.3256 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.3256

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -6.3268
- **Max Influence**: 2.0811
- **Feature Complexity**: 0.3946
**Top Influential Features:**
- feature_15: -1.3256 🔽
- feature_26: -1.4829 🔽
- age: 1.6733 🔼
- feature_29: 1.9261 🔼
- feature_22: 2.0811 🔼

#### Sample 1:
- **Prediction Sum**: 6.3268
- **Max Influence**: 2.0811
- **Feature Complexity**: 0.3946
**Top Influential Features:**
- feature_15: 1.3256 🔼
- feature_26: 1.4829 🔼
- age: -1.6733 🔽
- feature_29: -1.9261 🔽
- feature_22: -2.0811 🔽

---

### 6. catboost_MinMaxScaler (482bcacc2e1e2d3b64b8aafc9adf8052.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.catboost_model.CatBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 482bcacc2e1e2d3b64b8aafc9adf8052.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0183
- **Standard Deviation**: 0.0765
- **Range**: -0.3793 to 0.1196
- **Median**: -0.0013
- **Mean Absolute Value**: 0.0434

**Top 5 Most Important Features:**
1. **ca**: 0.1240 ↓ (decrease)
   - Mean SHAP: -0.0456, Std: 0.1528
2. **cp**: 0.1189 ↓ (decrease)
   - Mean SHAP: -0.0818, Std: 0.1455
3. **thal**: 0.0951 ↓ (decrease)
   - Mean SHAP: -0.0153, Std: 0.1153
4. **oldpeak**: 0.0402 ↓ (decrease)
   - Mean SHAP: -0.0290, Std: 0.0576
5. **age**: 0.0341 ↓ (decrease)
   - Mean SHAP: -0.0181, Std: 0.0396

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.0977
- **Max Influence**: 0.1138
- **Feature Complexity**: 0.0342
**Top Influential Features:**
- thalach: -0.0247 🔽
- restecg: 0.0354 🔼
- cp: 0.0608 🔼
- ca: 0.0951 🔼
- thal: -0.1138 🔽

#### Sample 1:
- **Prediction Sum**: 0.1036
- **Max Influence**: 0.0377
- **Feature Complexity**: 0.0123
**Top Influential Features:**
- cp: 0.0216 🔼
- thalach: 0.0243 🔼
- restecg: -0.0263 🔽
- thal: 0.0365 🔼
- ca: 0.0377 🔼

#### Sample 2:
- **Prediction Sum**: 0.0965
- **Max Influence**: 0.0786
- **Feature Complexity**: 0.0243
**Top Influential Features:**
- cp: 0.0302 🔼
- sex: -0.0313 🔽
- oldpeak: -0.0386 🔽
- thal: 0.0754 🔼
- ca: 0.0786 🔼

---

### 7. xgboost_StandardScaler (4909b94c48a292cf0fe8b8c02fb18c29.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.xgboost_model.XGBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 4909b94c48a292cf0fe8b8c02fb18c29.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0004
- **Standard Deviation**: 0.0626
- **Range**: -0.1796 to 0.2079
- **Median**: 0.0015
- **Mean Absolute Value**: 0.0455

**Top 5 Most Important Features:**
1. **ca**: 0.1309 ↓ (decrease)
   - Mean SHAP: -0.0052, Std: 0.1348
2. **thal**: 0.0826 ↓ (decrease)
   - Mean SHAP: -0.0043, Std: 0.0932
3. **cp**: 0.0821 ↑ (increase)
   - Mean SHAP: 0.0217, Std: 0.0897
4. **sex**: 0.0609 ↓ (decrease)
   - Mean SHAP: -0.0019, Std: 0.0691
5. **slope**: 0.0438 ↓ (decrease)
   - Mean SHAP: -0.0100, Std: 0.0465

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5054
- **Max Influence**: 0.1327
- **Feature Complexity**: 0.0427
**Top Influential Features:**
- oldpeak: 0.0330 🔼
- sex: 0.0407 🔼
- ca: 0.0820 🔼
- cp: 0.1248 🔼
- thal: 0.1327 🔼

#### Sample 1:
- **Prediction Sum**: -0.4388
- **Max Influence**: 0.1723
- **Feature Complexity**: 0.0431
**Top Influential Features:**
- thalach: -0.0372 🔽
- slope: -0.0451 🔽
- cp: -0.0512 🔽
- thal: -0.0665 🔽
- ca: -0.1723 🔽

#### Sample 2:
- **Prediction Sum**: 0.4710
- **Max Influence**: 0.1251
- **Feature Complexity**: 0.0412
**Top Influential Features:**
- chol: 0.0396 🔼
- oldpeak: 0.0532 🔼
- sex: 0.0756 🔼
- thal: 0.1249 🔼
- ca: 0.1251 🔼

---

### 8. naive_bayes_RobustScaler (4bb53c146651511ba253d11098218eda.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.naive_bayes_model.NaiveBayesModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 4bb53c146651511ba253d11098218eda.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0014
- **Standard Deviation**: 0.0817
- **Range**: -0.2052 to 0.5351
- **Median**: -0.0010
- **Mean Absolute Value**: 0.0518

**Top 5 Most Important Features:**
1. **ca**: 0.1532 ↑ (increase)
   - Mean SHAP: 0.0101, Std: 0.1863
2. **oldpeak**: 0.1323 ↓ (decrease)
   - Mean SHAP: -0.0346, Std: 0.1463
3. **thal**: 0.0642 ↑ (increase)
   - Mean SHAP: 0.0082, Std: 0.0839
4. **exang**: 0.0625 ↑ (increase)
   - Mean SHAP: 0.0153, Std: 0.0771
5. **sex**: 0.0553 ↑ (increase)
   - Mean SHAP: 0.0016, Std: 0.0649

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4606
- **Max Influence**: 0.1940
- **Feature Complexity**: 0.0496
**Top Influential Features:**
- oldpeak: 0.0387 🔼
- cp: 0.0397 🔼
- thal: 0.0558 🔼
- exang: 0.0583 🔼
- ca: 0.1940 🔼

#### Sample 1:
- **Prediction Sum**: -0.5394
- **Max Influence**: 0.1821
- **Feature Complexity**: 0.0581
**Top Influential Features:**
- age: -0.0263 🔽
- thalach: -0.0336 🔽
- cp: -0.0801 🔽
- oldpeak: -0.1610 🔽
- ca: -0.1821 🔽

#### Sample 2:
- **Prediction Sum**: 0.4606
- **Max Influence**: 0.3160
- **Feature Complexity**: 0.0813
**Top Influential Features:**
- sex: 0.0225 🔼
- age: 0.0338 🔼
- oldpeak: 0.0341 🔼
- thal: 0.0667 🔼
- ca: 0.3160 🔼

---

### 9. random_forest_MinMaxScaler (5f2f1d1aa6fcf380d252942157173988.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.random_forest_model.RandomForestModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 5f2f1d1aa6fcf380d252942157173988.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0000
- **Standard Deviation**: 0.0837
- **Range**: -0.1280 to 0.1280
- **Median**: -0.0000
- **Mean Absolute Value**: 0.0809

**Top 5 Most Important Features:**
1. **chol**: 0.1280 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1280
2. **feature_16**: 0.1164 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1164
3. **cp**: 0.1146 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1146
4. **feature_23**: 0.1057 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1057
5. **feature_27**: 0.1049 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1049

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.2172
- **Max Influence**: 0.1280
- **Feature Complexity**: 0.0214
**Top Influential Features:**
- feature_27: -0.1049 🔽
- feature_23: -0.1057 🔽
- cp: -0.1146 🔽
- feature_16: -0.1164 🔽
- chol: -0.1280 🔽

#### Sample 1:
- **Prediction Sum**: -0.2172
- **Max Influence**: 0.1280
- **Feature Complexity**: 0.0214
**Top Influential Features:**
- feature_27: 0.1049 🔼
- feature_23: 0.1057 🔼
- cp: 0.1146 🔼
- feature_16: 0.1164 🔼
- chol: 0.1280 🔼

---

### 10. adaboost_RobustScaler (612b3c45c7184ed12fcb5b6d449ce6fa.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.adaboost_model.AdaBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 612b3c45c7184ed12fcb5b6d449ce6fa.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0013
- **Standard Deviation**: 0.0128
- **Range**: -0.0363 to 0.0281
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0091

**Top 5 Most Important Features:**
1. **ca**: 0.0170 ↓ (decrease)
   - Mean SHAP: -0.0041, Std: 0.0192
2. **cp**: 0.0169 ↓ (decrease)
   - Mean SHAP: -0.0121, Std: 0.0197
3. **sex**: 0.0148 ↑ (increase)
   - Mean SHAP: 0.0042, Std: 0.0155
4. **slope**: 0.0137 ↓ (decrease)
   - Mean SHAP: -0.0018, Std: 0.0135
5. **oldpeak**: 0.0121 ↓ (decrease)
   - Mean SHAP: -0.0016, Std: 0.0153

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -0.0015
- **Max Influence**: 0.0193
- **Feature Complexity**: 0.0062
**Top Influential Features:**
- sex: -0.0094 🔽
- ca: 0.0102 🔼
- slope: -0.0136 🔽
- thal: -0.0156 🔽
- oldpeak: 0.0193 🔼

#### Sample 1:
- **Prediction Sum**: 0.0380
- **Max Influence**: 0.0239
- **Feature Complexity**: 0.0077
**Top Influential Features:**
- thal: 0.0067 🔼
- ca: 0.0102 🔼
- slope: 0.0136 🔼
- sex: 0.0219 🔼
- chol: -0.0239 🔽

#### Sample 2:
- **Prediction Sum**: -0.0036
- **Max Influence**: 0.0137
- **Feature Complexity**: 0.0042
**Top Influential Features:**
- oldpeak: -0.0092 🔽
- sex: -0.0094 🔽
- trestbps: 0.0102 🔼
- ca: 0.0103 🔼
- slope: -0.0137 🔽

---

### 11. decision_tree_StandardScaler (62f44e5f90f5ccec12546e56883d209a.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.decision_tree_model.DecisionTreeModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 62f44e5f90f5ccec12546e56883d209a.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 0.2375
- **Range**: -0.4126 to 0.4126
- **Median**: -0.0000
- **Mean Absolute Value**: 0.2148

**Top 5 Most Important Features:**
1. **feature_16**: 0.4126 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.4126
2. **chol**: 0.4048 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.4048
3. **feature_22**: 0.3562 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.3562
4. **feature_27**: 0.3504 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.3504
5. **feature_19**: 0.3412 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.3412

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.1847
- **Max Influence**: 0.4126
- **Feature Complexity**: 0.1013
**Top Influential Features:**
- feature_19: -0.3412 🔽
- feature_27: -0.3504 🔽
- feature_22: -0.3562 🔽
- chol: -0.4048 🔽
- feature_16: -0.4126 🔽

#### Sample 1:
- **Prediction Sum**: -0.1847
- **Max Influence**: 0.4126
- **Feature Complexity**: 0.1013
**Top Influential Features:**
- feature_19: 0.3412 🔼
- feature_27: 0.3504 🔼
- feature_22: 0.3562 🔼
- chol: 0.4048 🔼
- feature_16: 0.4126 🔼

---

### 12. knn_RobustScaler (673b3f175fc87117b95c6d5d29f51360.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.knn_model.KNNModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 673b3f175fc87117b95c6d5d29f51360.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0021
- **Standard Deviation**: 0.0713
- **Range**: -0.2250 to 0.3625
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0495

**Top 5 Most Important Features:**
1. **ca**: 0.1531 ↑ (increase)
   - Mean SHAP: 0.0259, Std: 0.1695
2. **oldpeak**: 0.0826 ↓ (decrease)
   - Mean SHAP: -0.0388, Std: 0.0855
3. **thal**: 0.0654 ↑ (increase)
   - Mean SHAP: 0.0079, Std: 0.0753
4. **thalach**: 0.0622 ↑ (increase)
   - Mean SHAP: 0.0105, Std: 0.0775
5. **sex**: 0.0491 ↑ (increase)
   - Mean SHAP: 0.0002, Std: 0.0552

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5200
- **Max Influence**: 0.2250
- **Feature Complexity**: 0.0565
**Top Influential Features:**
- thalach: 0.0375 🔼
- exang: 0.0400 🔼
- cp: 0.0475 🔼
- thal: 0.0675 🔼
- ca: 0.2250 🔼

#### Sample 1:
- **Prediction Sum**: -0.4800
- **Max Influence**: 0.1475
- **Feature Complexity**: 0.0524
**Top Influential Features:**
- thalach: -0.0275 🔽
- slope: -0.0625 🔽
- oldpeak: -0.1200 🔽
- cp: -0.1350 🔽
- ca: -0.1475 🔽

#### Sample 2:
- **Prediction Sum**: 0.5200
- **Max Influence**: 0.2300
- **Feature Complexity**: 0.0570
**Top Influential Features:**
- chol: 0.0350 🔼
- age: 0.0425 🔼
- sex: 0.0575 🔼
- thal: 0.0925 🔼
- ca: 0.2300 🔼

---

### 13. random_forest_MinMaxScaler (6c644fef3d5551b471daad24d9e163db.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.random_forest_model.RandomForestModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 6c644fef3d5551b471daad24d9e163db.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 0.0925
- **Range**: -0.1527 to 0.1527
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0894

**Top 5 Most Important Features:**
1. **feature_29**: 0.1527 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1527
2. **feature_22**: 0.1502 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1502
3. **feature_17**: 0.1171 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1171
4. **feature_23**: 0.1152 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1152
5. **feature_26**: 0.1104 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1104

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -0.6671
- **Max Influence**: 0.1527
- **Feature Complexity**: 0.0237
**Top Influential Features:**
- feature_26: -0.1104 🔽
- feature_23: -0.1152 🔽
- feature_17: -0.1171 🔽
- feature_22: 0.1502 🔼
- feature_29: 0.1527 🔼

#### Sample 1:
- **Prediction Sum**: 0.6671
- **Max Influence**: 0.1527
- **Feature Complexity**: 0.0237
**Top Influential Features:**
- feature_26: 0.1104 🔼
- feature_23: 0.1152 🔼
- feature_17: 0.1171 🔼
- feature_22: -0.1502 🔽
- feature_29: -0.1527 🔽

---

### 14. random_forest_StandardScaler (707a003f3ac9c12cc12b0c9d44e85207.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.random_forest_model.RandomForestModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 707a003f3ac9c12cc12b0c9d44e85207.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 0.0924
- **Range**: -0.1532 to 0.1532
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0893

**Top 5 Most Important Features:**
1. **feature_29**: 0.1532 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1532
2. **feature_22**: 0.1502 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1502
3. **feature_17**: 0.1168 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1168
4. **feature_23**: 0.1149 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1149
5. **feature_26**: 0.1102 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1102

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -0.6763
- **Max Influence**: 0.1532
- **Feature Complexity**: 0.0237
**Top Influential Features:**
- feature_26: -0.1102 🔽
- feature_23: -0.1149 🔽
- feature_17: -0.1168 🔽
- feature_22: 0.1502 🔼
- feature_29: 0.1532 🔼

#### Sample 1:
- **Prediction Sum**: 0.6763
- **Max Influence**: 0.1532
- **Feature Complexity**: 0.0237
**Top Influential Features:**
- feature_26: 0.1102 🔼
- feature_23: 0.1149 🔼
- feature_17: 0.1168 🔼
- feature_22: -0.1502 🔽
- feature_29: -0.1532 🔽

---

### 15. catboost_RobustScaler (76708fdb3526d173316c51935488784f.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.catboost_model.CatBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 76708fdb3526d173316c51935488784f.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0015
- **Standard Deviation**: 0.0644
- **Range**: -0.1906 to 0.2152
- **Median**: -0.0005
- **Mean Absolute Value**: 0.0448

**Top 5 Most Important Features:**
1. **ca**: 0.1517 ↓ (decrease)
   - Mean SHAP: -0.0004, Std: 0.1559
2. **thal**: 0.0859 ↓ (decrease)
   - Mean SHAP: -0.0002, Std: 0.0963
3. **cp**: 0.0768 ↑ (increase)
   - Mean SHAP: 0.0204, Std: 0.0847
4. **sex**: 0.0499 ↑ (increase)
   - Mean SHAP: 0.0016, Std: 0.0560
5. **oldpeak**: 0.0394 ↓ (decrease)
   - Mean SHAP: -0.0128, Std: 0.0433

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5333
- **Max Influence**: 0.1220
- **Feature Complexity**: 0.0426
**Top Influential Features:**
- thalach: 0.0341 🔼
- sex: 0.0553 🔼
- cp: 0.1017 🔼
- thal: 0.1182 🔼
- ca: 0.1220 🔼

#### Sample 1:
- **Prediction Sum**: -0.4350
- **Max Influence**: 0.1751
- **Feature Complexity**: 0.0445
**Top Influential Features:**
- thalach: -0.0368 🔽
- slope: -0.0390 🔽
- thal: -0.0446 🔽
- cp: -0.0764 🔽
- ca: -0.1751 🔽

#### Sample 2:
- **Prediction Sum**: 0.4892
- **Max Influence**: 0.1841
- **Feature Complexity**: 0.0533
**Top Influential Features:**
- oldpeak: 0.0356 🔼
- slope: 0.0384 🔼
- chol: 0.0492 🔼
- thal: 0.1417 🔼
- ca: 0.1841 🔼

---

### 16. random_forest_StandardScaler (79d3c3a3d9fb6adb0c7726f5d7e5f2f2.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.random_forest_model.RandomForestModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 79d3c3a3d9fb6adb0c7726f5d7e5f2f2.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0000
- **Standard Deviation**: 0.0840
- **Range**: -0.1280 to 0.1280
- **Median**: -0.0000
- **Mean Absolute Value**: 0.0815

**Top 5 Most Important Features:**
1. **chol**: 0.1280 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1280
2. **feature_16**: 0.1164 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1164
3. **cp**: 0.1147 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1147
4. **feature_23**: 0.1057 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1057
5. **feature_27**: 0.1049 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1049

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.1973
- **Max Influence**: 0.1280
- **Feature Complexity**: 0.0204
**Top Influential Features:**
- feature_27: -0.1049 🔽
- feature_23: -0.1057 🔽
- cp: -0.1147 🔽
- feature_16: -0.1164 🔽
- chol: -0.1280 🔽

#### Sample 1:
- **Prediction Sum**: -0.1973
- **Max Influence**: 0.1280
- **Feature Complexity**: 0.0204
**Top Influential Features:**
- feature_27: 0.1049 🔼
- feature_23: 0.1057 🔼
- cp: 0.1147 🔼
- feature_16: 0.1164 🔼
- chol: 0.1280 🔼

---

### 17. logistic_regression_MinMaxScaler (7d5254abb5ba57cb8b7c261fbeaee1ff.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.logistic_regression_model.LogisticRegressionModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 7d5254abb5ba57cb8b7c261fbeaee1ff.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0002
- **Standard Deviation**: 0.0679
- **Range**: -0.1824 to 0.3492
- **Median**: 0.0015
- **Mean Absolute Value**: 0.0445

**Top 5 Most Important Features:**
1. **ca**: 0.1637 ↓ (decrease)
   - Mean SHAP: -0.0016, Std: 0.1792
2. **thal**: 0.0852 ↑ (increase)
   - Mean SHAP: 0.0076, Std: 0.0949
3. **sex**: 0.0720 ↓ (decrease)
   - Mean SHAP: -0.0006, Std: 0.0784
4. **exang**: 0.0504 ↑ (increase)
   - Mean SHAP: 0.0072, Std: 0.0556
5. **cp**: 0.0488 ↑ (increase)
   - Mean SHAP: 0.0147, Std: 0.0588

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4865
- **Max Influence**: 0.2032
- **Feature Complexity**: 0.0541
**Top Influential Features:**
- sex: 0.0420 🔼
- exang: 0.0511 🔼
- cp: 0.0557 🔼
- thal: 0.0843 🔼
- ca: 0.2032 🔼

#### Sample 1:
- **Prediction Sum**: -0.4553
- **Max Influence**: 0.1636
- **Feature Complexity**: 0.0439
**Top Influential Features:**
- exang: -0.0383 🔽
- slope: -0.0534 🔽
- thal: -0.0581 🔽
- cp: -0.1080 🔽
- ca: -0.1636 🔽

#### Sample 2:
- **Prediction Sum**: 0.4538
- **Max Influence**: 0.2895
- **Feature Complexity**: 0.0786
**Top Influential Features:**
- exang: -0.0223 🔽
- restecg: -0.0318 🔽
- sex: 0.0533 🔼
- thal: 0.1272 🔼
- ca: 0.2895 🔼

---

### 18. gradient_boosting_MinMaxScaler (83770695de3d61591cfe88f0b1008025.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.gradient_boosting_model.GradientBoostingModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 83770695de3d61591cfe88f0b1008025.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0007
- **Standard Deviation**: 0.2445
- **Range**: -0.7107 to 1.1076
- **Median**: 0.0000
- **Mean Absolute Value**: 0.1509

**Top 5 Most Important Features:**
1. **thal**: 0.5473 ↓ (decrease)
   - Mean SHAP: -0.0556, Std: 0.5700
2. **ca**: 0.4164 ↑ (increase)
   - Mean SHAP: 0.1046, Std: 0.4445
3. **cp**: 0.3756 ↓ (decrease)
   - Mean SHAP: -0.0780, Std: 0.3756
4. **oldpeak**: 0.1967 ↓ (decrease)
   - Mean SHAP: -0.0004, Std: 0.2251
5. **sex**: 0.1035 ↑ (increase)
   - Mean SHAP: 0.0077, Std: 0.1208

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 2.0197
- **Max Influence**: 0.5963
- **Feature Complexity**: 0.2019
**Top Influential Features:**
- age: 0.0733 🔼
- oldpeak: 0.2086 🔼
- cp: 0.4133 🔼
- ca: 0.5152 🔼
- thal: 0.5963 🔼

#### Sample 1:
- **Prediction Sum**: -1.5781
- **Max Influence**: 0.4508
- **Feature Complexity**: 0.1440
**Top Influential Features:**
- oldpeak: -0.1159 🔽
- age: -0.1419 🔽
- ca: -0.2973 🔽
- cp: -0.3821 🔽
- thal: -0.4508 🔽

#### Sample 2:
- **Prediction Sum**: 1.2371
- **Max Influence**: 0.6247
- **Feature Complexity**: 0.2090
**Top Influential Features:**
- oldpeak: 0.1086 🔼
- chol: 0.1152 🔼
- cp: -0.4011 🔽
- ca: 0.5561 🔼
- thal: 0.6247 🔼

---

### 19. lightgbm_MinMaxScaler (8737848a4ea978fe0a10a0719666a281.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.lightgbm_model.LightGBMModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: 8737848a4ea978fe0a10a0719666a281.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0000
- **Standard Deviation**: 0.4195
- **Range**: -0.6802 to 0.6802
- **Median**: 0.0000
- **Mean Absolute Value**: 0.4084

**Top 5 Most Important Features:**
1. **feature_16**: 0.6802 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.6802
2. **chol**: 0.5836 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5836
3. **feature_19**: 0.5353 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5353
4. **feature_25**: 0.5235 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5235
5. **feature_20**: 0.5217 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5217

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4064
- **Max Influence**: 0.6802
- **Feature Complexity**: 0.0957
**Top Influential Features:**
- feature_20: -0.5217 🔽
- feature_25: 0.5235 🔼
- feature_19: -0.5353 🔽
- chol: -0.5836 🔽
- feature_16: -0.6802 🔽

#### Sample 1:
- **Prediction Sum**: -0.4064
- **Max Influence**: 0.6802
- **Feature Complexity**: 0.0957
**Top Influential Features:**
- feature_20: 0.5217 🔼
- feature_25: -0.5235 🔽
- feature_19: 0.5353 🔼
- chol: 0.5836 🔼
- feature_16: 0.6802 🔼

---

### 20. catboost_MinMaxScaler (8b33cd07361ff0cd887f31bc734d1420.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.catboost_model.CatBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 8b33cd07361ff0cd887f31bc734d1420.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0015
- **Standard Deviation**: 0.0639
- **Range**: -0.1841 to 0.2147
- **Median**: -0.0006
- **Mean Absolute Value**: 0.0444

**Top 5 Most Important Features:**
1. **ca**: 0.1490 ↑ (increase)
   - Mean SHAP: 0.0012, Std: 0.1535
2. **thal**: 0.0888 ↓ (decrease)
   - Mean SHAP: -0.0002, Std: 0.1005
3. **cp**: 0.0756 ↑ (increase)
   - Mean SHAP: 0.0184, Std: 0.0827
4. **sex**: 0.0489 ↑ (increase)
   - Mean SHAP: 0.0020, Std: 0.0551
5. **oldpeak**: 0.0393 ↓ (decrease)
   - Mean SHAP: -0.0110, Std: 0.0430

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5333
- **Max Influence**: 0.1281
- **Feature Complexity**: 0.0412
**Top Influential Features:**
- slope: 0.0354 🔼
- oldpeak: 0.0401 🔼
- cp: 0.0973 🔼
- thal: 0.1093 🔼
- ca: 0.1281 🔼

#### Sample 1:
- **Prediction Sum**: -0.4350
- **Max Influence**: 0.1281
- **Feature Complexity**: 0.0373
**Top Influential Features:**
- chol: -0.0280 🔽
- thalach: -0.0667 🔽
- thal: -0.0681 🔽
- cp: -0.0865 🔽
- ca: -0.1281 🔽

#### Sample 2:
- **Prediction Sum**: 0.4892
- **Max Influence**: 0.1633
- **Feature Complexity**: 0.0497
**Top Influential Features:**
- oldpeak: 0.0375 🔼
- sex: 0.0411 🔼
- chol: 0.0597 🔼
- thal: 0.1453 🔼
- ca: 0.1633 🔼

---

### 21. naive_bayes_MinMaxScaler (9571d82a80db9b4a4e08f015c173ccfd.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.naive_bayes_model.NaiveBayesModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 9571d82a80db9b4a4e08f015c173ccfd.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0014
- **Standard Deviation**: 0.0796
- **Range**: -0.2018 to 0.5132
- **Median**: -0.0007
- **Mean Absolute Value**: 0.0513

**Top 5 Most Important Features:**
1. **ca**: 0.1495 ↑ (increase)
   - Mean SHAP: 0.0062, Std: 0.1798
2. **oldpeak**: 0.1280 ↓ (decrease)
   - Mean SHAP: -0.0344, Std: 0.1423
3. **thal**: 0.0696 ↑ (increase)
   - Mean SHAP: 0.0094, Std: 0.0878
4. **exang**: 0.0620 ↑ (increase)
   - Mean SHAP: 0.0172, Std: 0.0736
5. **sex**: 0.0527 ↑ (increase)
   - Mean SHAP: 0.0077, Std: 0.0582

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4606
- **Max Influence**: 0.1714
- **Feature Complexity**: 0.0445
**Top Influential Features:**
- sex: 0.0416 🔼
- cp: 0.0482 🔼
- exang: 0.0552 🔼
- thal: 0.0654 🔼
- ca: 0.1714 🔼

#### Sample 1:
- **Prediction Sum**: -0.5394
- **Max Influence**: 0.1542
- **Feature Complexity**: 0.0479
**Top Influential Features:**
- thalach: -0.0502 🔽
- cp: -0.0530 🔽
- age: -0.0575 🔽
- oldpeak: -0.1392 🔽
- ca: -0.1542 🔽

#### Sample 2:
- **Prediction Sum**: 0.4606
- **Max Influence**: 0.2873
- **Feature Complexity**: 0.0737
**Top Influential Features:**
- age: 0.0246 🔼
- oldpeak: 0.0261 🔼
- sex: 0.0398 🔼
- thal: 0.0767 🔼
- ca: 0.2873 🔼

---

### 22. logistic_regression_StandardScaler (9cae7645a3969147c5db51416dda9b8b.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.logistic_regression_model.LogisticRegressionModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 9cae7645a3969147c5db51416dda9b8b.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0020
- **Standard Deviation**: 0.0760
- **Range**: -0.2099 to 0.4316
- **Median**: 0.0036
- **Mean Absolute Value**: 0.0499

**Top 5 Most Important Features:**
1. **ca**: 0.1912 ↑ (increase)
   - Mean SHAP: 0.0014, Std: 0.2077
2. **sex**: 0.0798 ↓ (decrease)
   - Mean SHAP: -0.0023, Std: 0.0869
3. **thal**: 0.0729 ↑ (increase)
   - Mean SHAP: 0.0096, Std: 0.0839
4. **cp**: 0.0506 ↑ (increase)
   - Mean SHAP: 0.0147, Std: 0.0618
5. **slope**: 0.0440 ↓ (decrease)
   - Mean SHAP: -0.0147, Std: 0.0491

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5058
- **Max Influence**: 0.2339
- **Feature Complexity**: 0.0609
**Top Influential Features:**
- exang: 0.0365 🔼
- sex: 0.0453 🔼
- cp: 0.0464 🔼
- thal: 0.0859 🔼
- ca: 0.2339 🔼

#### Sample 1:
- **Prediction Sum**: -0.4577
- **Max Influence**: 0.1996
- **Feature Complexity**: 0.0522
**Top Influential Features:**
- sex: 0.0331 🔼
- thal: -0.0511 🔽
- slope: -0.0618 🔽
- cp: -0.1179 🔽
- ca: -0.1996 🔽

#### Sample 2:
- **Prediction Sum**: 0.4922
- **Max Influence**: 0.3222
- **Feature Complexity**: 0.0843
**Top Influential Features:**
- slope: 0.0224 🔼
- restecg: -0.0274 🔽
- sex: 0.0551 🔼
- thal: 0.1036 🔼
- ca: 0.3222 🔼

---

### 23. adaboost_StandardScaler (9dd3e048bc14008aad9f4ad650994275.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.adaboost_model.AdaBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 9dd3e048bc14008aad9f4ad650994275.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0013
- **Standard Deviation**: 0.0128
- **Range**: -0.0363 to 0.0282
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0092

**Top 5 Most Important Features:**
1. **ca**: 0.0170 ↓ (decrease)
   - Mean SHAP: -0.0041, Std: 0.0192
2. **cp**: 0.0169 ↓ (decrease)
   - Mean SHAP: -0.0120, Std: 0.0197
3. **sex**: 0.0149 ↑ (increase)
   - Mean SHAP: 0.0042, Std: 0.0156
4. **slope**: 0.0137 ↓ (decrease)
   - Mean SHAP: -0.0018, Std: 0.0135
5. **oldpeak**: 0.0121 ↓ (decrease)
   - Mean SHAP: -0.0017, Std: 0.0153

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -0.0015
- **Max Influence**: 0.0193
- **Feature Complexity**: 0.0062
**Top Influential Features:**
- sex: -0.0094 🔽
- ca: 0.0102 🔼
- slope: -0.0137 🔽
- thal: -0.0156 🔽
- oldpeak: 0.0193 🔼

#### Sample 1:
- **Prediction Sum**: 0.0380
- **Max Influence**: 0.0240
- **Feature Complexity**: 0.0077
**Top Influential Features:**
- thal: 0.0066 🔼
- ca: 0.0102 🔼
- slope: 0.0136 🔼
- sex: 0.0219 🔼
- chol: -0.0240 🔽

#### Sample 2:
- **Prediction Sum**: -0.0036
- **Max Influence**: 0.0137
- **Feature Complexity**: 0.0042
**Top Influential Features:**
- oldpeak: -0.0092 🔽
- sex: -0.0094 🔽
- trestbps: 0.0102 🔼
- ca: 0.0103 🔼
- slope: -0.0137 🔽

---

### 24. adaboost_MinMaxScaler (9f796792e058b513f1f3336d50b135e8.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.adaboost_model.AdaBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: 9f796792e058b513f1f3336d50b135e8.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0002
- **Standard Deviation**: 0.0452
- **Range**: -0.1003 to 0.1284
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0290

**Top 5 Most Important Features:**
1. **ca**: 0.0948 ↓ (decrease)
   - Mean SHAP: -0.0065, Std: 0.0947
2. **thal**: 0.0865 ↑ (increase)
   - Mean SHAP: 0.0040, Std: 0.0895
3. **cp**: 0.0709 ↑ (increase)
   - Mean SHAP: 0.0156, Std: 0.0757
4. **oldpeak**: 0.0347 ↓ (decrease)
   - Mean SHAP: -0.0132, Std: 0.0353
5. **sex**: 0.0263 ↓ (decrease)
   - Mean SHAP: -0.0024, Std: 0.0288

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.3976
- **Max Influence**: 0.1149
- **Feature Complexity**: 0.0394
**Top Influential Features:**
- exang: 0.0279 🔼
- oldpeak: 0.0300 🔼
- ca: 0.0902 🔼
- cp: 0.0985 🔼
- thal: 0.1149 🔼

#### Sample 1:
- **Prediction Sum**: -0.3122
- **Max Influence**: 0.0978
- **Feature Complexity**: 0.0286
**Top Influential Features:**
- slope: -0.0286 🔽
- oldpeak: -0.0397 🔽
- cp: -0.0460 🔽
- thal: -0.0650 🔽
- ca: -0.0978 🔽

#### Sample 2:
- **Prediction Sum**: 0.2666
- **Max Influence**: 0.1278
- **Feature Complexity**: 0.0391
**Top Influential Features:**
- sex: 0.0187 🔼
- oldpeak: 0.0311 🔼
- cp: -0.0382 🔽
- ca: 0.1021 🔼
- thal: 0.1278 🔼

---

### 25. gradient_boosting_StandardScaler (ab26ebd9718345a8f59556422ad063c5.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.gradient_boosting_model.GradientBoostingModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: ab26ebd9718345a8f59556422ad063c5.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0976
- **Standard Deviation**: 2.5447
- **Range**: -5.3054 to 10.6581
- **Median**: -0.0935
- **Mean Absolute Value**: 1.8948

**Top 5 Most Important Features:**
1. **ca**: 4.6082 ↑ (increase)
   - Mean SHAP: 1.2943, Std: 4.9523
2. **thal**: 3.9658 ↓ (decrease)
   - Mean SHAP: -0.7742, Std: 4.2825
3. **cp**: 3.1663 ↓ (decrease)
   - Mean SHAP: -0.4579, Std: 3.2376
4. **oldpeak**: 2.2837 ↑ (increase)
   - Mean SHAP: 0.0351, Std: 2.7169
5. **thalach**: 1.8801 ↑ (increase)
   - Mean SHAP: 0.3322, Std: 2.1921

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 23.4434
- **Max Influence**: 6.4878
- **Feature Complexity**: 1.7689
**Top Influential Features:**
- thalach: 2.0936 🔼
- oldpeak: 2.2736 🔼
- cp: 2.9070 🔼
- thal: 4.7215 🔼
- ca: 6.4878 🔼

#### Sample 1:
- **Prediction Sum**: -18.4750
- **Max Influence**: 4.2288
- **Feature Complexity**: 1.2286
**Top Influential Features:**
- thalach: -1.6488 🔽
- chol: -1.8296 🔽
- thal: -3.4224 🔽
- ca: -3.7177 🔽
- cp: -4.2288 🔽

#### Sample 2:
- **Prediction Sum**: 22.4009
- **Max Influence**: 8.3010
- **Feature Complexity**: 2.1908
**Top Influential Features:**
- oldpeak: 1.8973 🔼
- chol: 2.4306 🔼
- slope: 2.6720 🔼
- thal: 4.9307 🔼
- ca: 8.3010 🔼

---

### 26. naive_bayes_StandardScaler (ad8feed0890cf0e839f1c5a2ebb2e3dc.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.naive_bayes_model.NaiveBayesModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: ad8feed0890cf0e839f1c5a2ebb2e3dc.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0014
- **Standard Deviation**: 0.0819
- **Range**: -0.1973 to 0.5384
- **Median**: -0.0015
- **Mean Absolute Value**: 0.0520

**Top 5 Most Important Features:**
1. **ca**: 0.1520 ↑ (increase)
   - Mean SHAP: 0.0123, Std: 0.1871
2. **oldpeak**: 0.1314 ↓ (decrease)
   - Mean SHAP: -0.0358, Std: 0.1449
3. **thal**: 0.0687 ↑ (increase)
   - Mean SHAP: 0.0075, Std: 0.0853
4. **exang**: 0.0643 ↑ (increase)
   - Mean SHAP: 0.0159, Std: 0.0785
5. **sex**: 0.0557 ↑ (increase)
   - Mean SHAP: 0.0045, Std: 0.0630

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4606
- **Max Influence**: 0.1681
- **Feature Complexity**: 0.0450
**Top Influential Features:**
- oldpeak: 0.0445 🔼
- cp: 0.0455 🔼
- exang: 0.0632 🔼
- thal: 0.0739 🔼
- ca: 0.1681 🔼

#### Sample 1:
- **Prediction Sum**: -0.5394
- **Max Influence**: 0.1488
- **Feature Complexity**: 0.0470
**Top Influential Features:**
- age: -0.0521 🔽
- cp: -0.0522 🔽
- thal: -0.0764 🔽
- ca: -0.1393 🔽
- oldpeak: -0.1488 🔽

#### Sample 2:
- **Prediction Sum**: 0.4606
- **Max Influence**: 0.3667
- **Feature Complexity**: 0.0952
**Top Influential Features:**
- exang: -0.0169 🔽
- age: 0.0260 🔼
- sex: 0.0343 🔼
- thal: 0.0571 🔼
- ca: 0.3667 🔼

---

### 27. decision_tree_RobustScaler (b71a0603d0b97512a9772255f25790c3.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.decision_tree_model.DecisionTreeModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: b71a0603d0b97512a9772255f25790c3.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 0.2378
- **Range**: -0.4126 to 0.4126
- **Median**: -0.0000
- **Mean Absolute Value**: 0.2153

**Top 5 Most Important Features:**
1. **feature_16**: 0.4126 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.4126
2. **chol**: 0.4048 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.4048
3. **feature_22**: 0.3562 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.3562
4. **feature_27**: 0.3504 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.3504
5. **feature_19**: 0.3412 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.3412

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.1668
- **Max Influence**: 0.4126
- **Feature Complexity**: 0.1009
**Top Influential Features:**
- feature_19: -0.3412 🔽
- feature_27: -0.3504 🔽
- feature_22: -0.3562 🔽
- chol: -0.4048 🔽
- feature_16: -0.4126 🔽

#### Sample 1:
- **Prediction Sum**: -0.1668
- **Max Influence**: 0.4126
- **Feature Complexity**: 0.1009
**Top Influential Features:**
- feature_19: 0.3412 🔼
- feature_27: 0.3504 🔼
- feature_22: 0.3562 🔼
- chol: 0.4048 🔼
- feature_16: 0.4126 🔼

---

### 28. adaboost_MinMaxScaler (c4c091394afb500d051d52dcad952c9b.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.adaboost_model.AdaBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: c4c091394afb500d051d52dcad952c9b.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0013
- **Standard Deviation**: 0.0128
- **Range**: -0.0363 to 0.0281
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0091

**Top 5 Most Important Features:**
1. **ca**: 0.0170 ↓ (decrease)
   - Mean SHAP: -0.0041, Std: 0.0192
2. **cp**: 0.0169 ↓ (decrease)
   - Mean SHAP: -0.0121, Std: 0.0197
3. **sex**: 0.0149 ↑ (increase)
   - Mean SHAP: 0.0042, Std: 0.0155
4. **slope**: 0.0137 ↓ (decrease)
   - Mean SHAP: -0.0018, Std: 0.0135
5. **oldpeak**: 0.0121 ↓ (decrease)
   - Mean SHAP: -0.0017, Std: 0.0153

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -0.0015
- **Max Influence**: 0.0193
- **Feature Complexity**: 0.0062
**Top Influential Features:**
- sex: -0.0094 🔽
- ca: 0.0102 🔼
- slope: -0.0137 🔽
- thal: -0.0156 🔽
- oldpeak: 0.0193 🔼

#### Sample 1:
- **Prediction Sum**: 0.0380
- **Max Influence**: 0.0239
- **Feature Complexity**: 0.0077
**Top Influential Features:**
- thal: 0.0067 🔼
- ca: 0.0102 🔼
- slope: 0.0136 🔼
- sex: 0.0219 🔼
- chol: -0.0239 🔽

#### Sample 2:
- **Prediction Sum**: -0.0036
- **Max Influence**: 0.0137
- **Feature Complexity**: 0.0042
**Top Influential Features:**
- oldpeak: -0.0092 🔽
- sex: -0.0094 🔽
- trestbps: 0.0102 🔼
- ca: 0.0103 🔼
- slope: -0.0137 🔽

---

### 29. random_forest_RobustScaler (c9cb430e72cf910a5996c53aab2ea0ce.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.random_forest_model.RandomForestModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: c9cb430e72cf910a5996c53aab2ea0ce.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0000
- **Standard Deviation**: 0.0836
- **Range**: -0.1280 to 0.1280
- **Median**: -0.0000
- **Mean Absolute Value**: 0.0808

**Top 5 Most Important Features:**
1. **chol**: 0.1280 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1280
2. **feature_16**: 0.1164 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1164
3. **cp**: 0.1146 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1146
4. **feature_23**: 0.1057 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1057
5. **feature_27**: 0.1049 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1049

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.2169
- **Max Influence**: 0.1280
- **Feature Complexity**: 0.0214
**Top Influential Features:**
- feature_27: -0.1049 🔽
- feature_23: -0.1057 🔽
- cp: -0.1146 🔽
- feature_16: -0.1164 🔽
- chol: -0.1280 🔽

#### Sample 1:
- **Prediction Sum**: -0.2169
- **Max Influence**: 0.1280
- **Feature Complexity**: 0.0214
**Top Influential Features:**
- feature_27: 0.1049 🔼
- feature_23: 0.1057 🔼
- cp: 0.1146 🔼
- feature_16: 0.1164 🔼
- chol: 0.1280 🔼

---

### 30. gradient_boosting_RobustScaler (cb64048086ab50dd25410c92a230a187.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.gradient_boosting_model.GradientBoostingModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: cb64048086ab50dd25410c92a230a187.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0519
- **Standard Deviation**: 2.5281
- **Range**: -5.9219 to 9.3074
- **Median**: -0.1081
- **Mean Absolute Value**: 1.8968

**Top 5 Most Important Features:**
1. **ca**: 4.3637 ↑ (increase)
   - Mean SHAP: 1.1001, Std: 4.6174
2. **thal**: 3.8552 ↓ (decrease)
   - Mean SHAP: -0.7981, Std: 4.3044
3. **cp**: 3.3813 ↓ (decrease)
   - Mean SHAP: -0.4626, Std: 3.4634
4. **oldpeak**: 2.5683 ↑ (increase)
   - Mean SHAP: 0.0730, Std: 3.0110
5. **thalach**: 1.7981 ↑ (increase)
   - Mean SHAP: 0.5273, Std: 2.1138

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 22.5496
- **Max Influence**: 5.5158
- **Feature Complexity**: 1.5357
**Top Influential Features:**
- thalach: 2.0563 🔼
- oldpeak: 2.2486 🔼
- cp: 2.8418 🔼
- thal: 4.4728 🔼
- ca: 5.5158 🔼

#### Sample 1:
- **Prediction Sum**: -17.7008
- **Max Influence**: 4.3303
- **Feature Complexity**: 1.3505
**Top Influential Features:**
- thalach: -1.6816 🔽
- age: -2.1730 🔽
- thal: -3.4578 🔽
- ca: -3.7112 🔽
- cp: -4.3303 🔽

#### Sample 2:
- **Prediction Sum**: 21.1843
- **Max Influence**: 6.9013
- **Feature Complexity**: 1.9638
**Top Influential Features:**
- oldpeak: 2.1217 🔼
- slope: 2.4482 🔼
- chol: 2.7545 🔼
- thal: 5.3154 🔼
- ca: 6.9013 🔼

---

### 31. catboost_StandardScaler (ce2c307b980c938d0ca56931ff361c7d.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.catboost_model.CatBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: ce2c307b980c938d0ca56931ff361c7d.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0015
- **Standard Deviation**: 0.0643
- **Range**: -0.1804 to 0.2110
- **Median**: -0.0006
- **Mean Absolute Value**: 0.0447

**Top 5 Most Important Features:**
1. **ca**: 0.1515 ↑ (increase)
   - Mean SHAP: 0.0040, Std: 0.1559
2. **thal**: 0.0895 ↓ (decrease)
   - Mean SHAP: -0.0026, Std: 0.0997
3. **cp**: 0.0720 ↑ (increase)
   - Mean SHAP: 0.0203, Std: 0.0797
4. **sex**: 0.0515 ↑ (increase)
   - Mean SHAP: 0.0010, Std: 0.0572
5. **oldpeak**: 0.0388 ↓ (decrease)
   - Mean SHAP: -0.0122, Std: 0.0421

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5333
- **Max Influence**: 0.2110
- **Feature Complexity**: 0.0538
**Top Influential Features:**
- thalach: 0.0361 🔼
- sex: 0.0450 🔼
- cp: 0.0629 🔼
- thal: 0.0765 🔼
- ca: 0.2110 🔼

#### Sample 1:
- **Prediction Sum**: -0.4350
- **Max Influence**: 0.1515
- **Feature Complexity**: 0.0406
**Top Influential Features:**
- thalach: -0.0346 🔽
- slope: -0.0417 🔽
- thal: -0.0652 🔽
- cp: -0.0812 🔽
- ca: -0.1515 🔽

#### Sample 2:
- **Prediction Sum**: 0.4892
- **Max Influence**: 0.1624
- **Feature Complexity**: 0.0513
**Top Influential Features:**
- sex: 0.0398 🔼
- chol: 0.0411 🔼
- oldpeak: 0.0487 🔼
- thal: 0.1565 🔼
- ca: 0.1624 🔼

---

### 32. lightgbm_StandardScaler (cf1f1b18e0f15f76052a057f307a2d23.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.lightgbm_model.LightGBMModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: cf1f1b18e0f15f76052a057f307a2d23.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 0.4195
- **Range**: -0.6802 to 0.6802
- **Median**: 0.0000
- **Mean Absolute Value**: 0.4084

**Top 5 Most Important Features:**
1. **feature_16**: 0.6802 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.6802
2. **chol**: 0.5836 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5836
3. **feature_19**: 0.5353 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5353
4. **feature_25**: 0.5235 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5235
5. **feature_20**: 0.5217 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 0.5217

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.4064
- **Max Influence**: 0.6802
- **Feature Complexity**: 0.0957
**Top Influential Features:**
- feature_20: -0.5217 🔽
- feature_25: 0.5235 🔼
- feature_19: -0.5353 🔽
- chol: -0.5836 🔽
- feature_16: -0.6802 🔽

#### Sample 1:
- **Prediction Sum**: -0.4064
- **Max Influence**: 0.6802
- **Feature Complexity**: 0.0957
**Top Influential Features:**
- feature_20: 0.5217 🔼
- feature_25: -0.5235 🔽
- feature_19: 0.5353 🔼
- chol: 0.5836 🔼
- feature_16: 0.6802 🔼

---

### 33. lightgbm_RobustScaler (d10cd535613fc42738da94feb7e2177e.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.lightgbm_model.LightGBMModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: d10cd535613fc42738da94feb7e2177e.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0000
- **Standard Deviation**: 1.0973
- **Range**: -2.0922 to 2.0922
- **Median**: 0.0000
- **Mean Absolute Value**: 1.0230

**Top 5 Most Important Features:**
1. **feature_22**: 2.0922 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 2.0922
2. **feature_29**: 1.7911 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.7911
3. **age**: 1.7113 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.7113
4. **feature_26**: 1.5501 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.5501
5. **exang**: 1.3247 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.3247

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -6.1961
- **Max Influence**: 2.0922
- **Feature Complexity**: 0.3971
**Top Influential Features:**
- exang: 1.3247 🔼
- feature_26: -1.5501 🔽
- age: 1.7113 🔼
- feature_29: 1.7911 🔼
- feature_22: 2.0922 🔼

#### Sample 1:
- **Prediction Sum**: 6.1961
- **Max Influence**: 2.0922
- **Feature Complexity**: 0.3971
**Top Influential Features:**
- exang: -1.3247 🔽
- feature_26: 1.5501 🔼
- age: -1.7113 🔽
- feature_29: -1.7911 🔽
- feature_22: -2.0922 🔽

---

### 34. random_forest_RobustScaler (d3a52e81867f4ae3ba5a396e88a3ea53.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.random_forest_model.RandomForestModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: d3a52e81867f4ae3ba5a396e88a3ea53.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 0.0923
- **Range**: -0.1532 to 0.1532
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0892

**Top 5 Most Important Features:**
1. **feature_29**: 0.1532 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1532
2. **feature_22**: 0.1503 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1503
3. **feature_17**: 0.1170 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.1170
4. **feature_23**: 0.1160 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1160
5. **feature_26**: 0.1104 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.1104

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -0.6764
- **Max Influence**: 0.1532
- **Feature Complexity**: 0.0237
**Top Influential Features:**
- feature_26: -0.1104 🔽
- feature_23: -0.1160 🔽
- feature_17: -0.1170 🔽
- feature_22: 0.1503 🔼
- feature_29: 0.1532 🔼

#### Sample 1:
- **Prediction Sum**: 0.6764
- **Max Influence**: 0.1532
- **Feature Complexity**: 0.0237
**Top Influential Features:**
- feature_26: 0.1104 🔼
- feature_23: 0.1160 🔼
- feature_17: 0.1170 🔼
- feature_22: -0.1503 🔽
- feature_29: -0.1532 🔽

---

### 35. knn_MinMaxScaler (d8a51a0ba86a0bd5f53177df83f1e1c9.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.knn_model.KNNModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: d8a51a0ba86a0bd5f53177df83f1e1c9.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0031
- **Standard Deviation**: 0.0728
- **Range**: -0.2125 to 0.3050
- **Median**: 0.0013
- **Mean Absolute Value**: 0.0479

**Top 5 Most Important Features:**
1. **thal**: 0.1257 ↑ (increase)
   - Mean SHAP: 0.0038, Std: 0.1383
2. **ca**: 0.0956 ↓ (decrease)
   - Mean SHAP: -0.0004, Std: 0.1125
3. **sex**: 0.0907 ↑ (increase)
   - Mean SHAP: 0.0010, Std: 0.1055
4. **exang**: 0.0877 ↑ (increase)
   - Mean SHAP: 0.0260, Std: 0.1083
5. **cp**: 0.0563 ↑ (increase)
   - Mean SHAP: 0.0244, Std: 0.0659

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5600
- **Max Influence**: 0.1400
- **Feature Complexity**: 0.0516
**Top Influential Features:**
- exang: 0.0600 🔼
- sex: 0.0800 🔼
- cp: 0.1150 🔼
- thal: 0.1250 🔼
- ca: 0.1400 🔼

#### Sample 1:
- **Prediction Sum**: -0.4400
- **Max Influence**: 0.1150
- **Feature Complexity**: 0.0380
**Top Influential Features:**
- thalach: -0.0450 🔽
- thal: -0.0800 🔽
- cp: -0.0850 🔽
- age: -0.0950 🔽
- ca: -0.1150 🔽

#### Sample 2:
- **Prediction Sum**: 0.3600
- **Max Influence**: 0.1950
- **Feature Complexity**: 0.0671
**Top Influential Features:**
- exang: -0.0450 🔽
- sex: 0.0875 🔼
- restecg: -0.0975 🔽
- ca: 0.1850 🔼
- thal: 0.1950 🔼

---

### 36. decision_tree_MinMaxScaler (d9fde9d57a4c60c3e5ca693ddf2cdf42.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.decision_tree_model.DecisionTreeModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: d9fde9d57a4c60c3e5ca693ddf2cdf42.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0000
- **Standard Deviation**: 0.2372
- **Range**: -0.4126 to 0.4126
- **Median**: -0.0000
- **Mean Absolute Value**: 0.2151

**Top 5 Most Important Features:**
1. **feature_16**: 0.4126 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.4126
2. **chol**: 0.4024 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.4024
3. **feature_22**: 0.3562 ↑ (increase)
   - Mean SHAP: 0.0000, Std: 0.3562
4. **feature_27**: 0.3504 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.3504
5. **feature_19**: 0.3412 ↓ (decrease)
   - Mean SHAP: -0.0000, Std: 0.3412

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.1653
- **Max Influence**: 0.4126
- **Feature Complexity**: 0.0999
**Top Influential Features:**
- feature_19: -0.3412 🔽
- feature_27: -0.3504 🔽
- feature_22: -0.3562 🔽
- chol: -0.4024 🔽
- feature_16: -0.4126 🔽

#### Sample 1:
- **Prediction Sum**: -0.1653
- **Max Influence**: 0.4126
- **Feature Complexity**: 0.0999
**Top Influential Features:**
- feature_19: 0.3412 🔼
- feature_27: 0.3504 🔼
- feature_22: 0.3562 🔼
- chol: 0.4024 🔼
- feature_16: 0.4126 🔼

---

### 37. gradient_boosting_MinMaxScaler (daafe7cf36cf7fe76e20487da8068c17.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.gradient_boosting_model.GradientBoostingModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: daafe7cf36cf7fe76e20487da8068c17.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.4028
- **Standard Deviation**: 2.8457
- **Range**: -12.6453 to 9.4051
- **Median**: 0.1671
- **Mean Absolute Value**: 2.0097

**Top 5 Most Important Features:**
1. **cp**: 4.9850 ↑ (increase)
   - Mean SHAP: 0.9320, Std: 5.2066
2. **ca**: 4.7901 ↑ (increase)
   - Mean SHAP: 1.1076, Std: 4.9138
3. **thal**: 3.8058 ↑ (increase)
   - Mean SHAP: 1.1153, Std: 3.8510
4. **oldpeak**: 2.6693 ↑ (increase)
   - Mean SHAP: 0.0586, Std: 3.5999
5. **thalach**: 1.9004 ↓ (decrease)
   - Mean SHAP: -0.0856, Std: 2.5697

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 17.5513
- **Max Influence**: 6.6301
- **Feature Complexity**: 2.2297
**Top Influential Features:**
- thal: -3.5899 🔽
- ca: 3.8725 🔼
- oldpeak: 4.8673 🔼
- age: 5.5669 🔼
- cp: 6.6301 🔼

#### Sample 1:
- **Prediction Sum**: 19.8462
- **Max Influence**: 4.1430
- **Feature Complexity**: 1.2150
**Top Influential Features:**
- thalach: 2.0759 🔼
- sex: 2.4023 🔼
- ca: 2.7533 🔼
- cp: 2.8942 🔼
- thal: 4.1430 🔼

#### Sample 2:
- **Prediction Sum**: 18.4137
- **Max Influence**: 6.6328
- **Feature Complexity**: 1.9858
**Top Influential Features:**
- sex: -1.2909 🔽
- thal: 2.0455 🔼
- chol: 3.7112 🔼
- cp: 4.7467 🔼
- ca: 6.6328 🔼

---

### 38. lightgbm_StandardScaler (ddeb252661b7d97dbd6bcb05bd8fdecd.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.lightgbm_model.LightGBMModel'>
- **Samples Analyzed**: 2
- **Features**: 13
- **Cache File**: ddeb252661b7d97dbd6bcb05bd8fdecd.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0000
- **Standard Deviation**: 1.1105
- **Range**: -2.0803 to 2.0803
- **Median**: 0.0000
- **Mean Absolute Value**: 1.0380

**Top 5 Most Important Features:**
1. **feature_22**: 2.0803 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 2.0803
2. **feature_29**: 1.9261 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.9261
3. **age**: 1.6731 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.6731
4. **feature_26**: 1.4868 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.4868
5. **feature_15**: 1.3284 ↓ (decrease)
   - Mean SHAP: 0.0000, Std: 1.3284

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: -6.3459
- **Max Influence**: 2.0803
- **Feature Complexity**: 0.3946
**Top Influential Features:**
- feature_15: -1.3284 🔽
- feature_26: -1.4868 🔽
- age: 1.6731 🔼
- feature_29: 1.9261 🔼
- feature_22: 2.0803 🔼

#### Sample 1:
- **Prediction Sum**: 6.3459
- **Max Influence**: 2.0803
- **Feature Complexity**: 0.3946
**Top Influential Features:**
- feature_15: 1.3284 🔼
- feature_26: 1.4868 🔼
- age: -1.6731 🔽
- feature_29: -1.9261 🔽
- feature_22: -2.0803 🔽

---

### 39. logistic_regression_RobustScaler (de881a5bf44b3f37495e0e511a31ddba.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.logistic_regression_model.LogisticRegressionModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: de881a5bf44b3f37495e0e511a31ddba.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: 0.0019
- **Standard Deviation**: 0.0751
- **Range**: -0.2195 to 0.4220
- **Median**: 0.0040
- **Mean Absolute Value**: 0.0492

**Top 5 Most Important Features:**
1. **ca**: 0.1921 ↑ (increase)
   - Mean SHAP: 0.0018, Std: 0.2087
2. **sex**: 0.0743 ↓ (decrease)
   - Mean SHAP: -0.0012, Std: 0.0808
3. **thal**: 0.0725 ↑ (increase)
   - Mean SHAP: 0.0104, Std: 0.0822
4. **cp**: 0.0503 ↑ (increase)
   - Mean SHAP: 0.0139, Std: 0.0614
5. **thalach**: 0.0434 ↓ (decrease)
   - Mean SHAP: -0.0022, Std: 0.0513

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.5017
- **Max Influence**: 0.2451
- **Feature Complexity**: 0.0632
**Top Influential Features:**
- sex: 0.0399 🔼
- exang: 0.0427 🔼
- cp: 0.0506 🔼
- thal: 0.0723 🔼
- ca: 0.2451 🔼

#### Sample 1:
- **Prediction Sum**: -0.4582
- **Max Influence**: 0.2095
- **Feature Complexity**: 0.0537
**Top Influential Features:**
- thalach: -0.0351 🔽
- slope: -0.0380 🔽
- thal: -0.0426 🔽
- cp: -0.1134 🔽
- ca: -0.2095 🔽

#### Sample 2:
- **Prediction Sum**: 0.4861
- **Max Influence**: 0.3410
- **Feature Complexity**: 0.0893
**Top Influential Features:**
- slope: 0.0145 🔼
- restecg: -0.0230 🔽
- sex: 0.0520 🔼
- thal: 0.0943 🔼
- ca: 0.3410 🔼

---

### 40. catboost_StandardScaler (e8903d854418cd5f1674af547a8db2a8.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.catboost_model.CatBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: e8903d854418cd5f1674af547a8db2a8.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0183
- **Standard Deviation**: 0.0755
- **Range**: -0.4027 to 0.1344
- **Median**: -0.0013
- **Mean Absolute Value**: 0.0437

**Top 5 Most Important Features:**
1. **cp**: 0.1166 ↓ (decrease)
   - Mean SHAP: -0.0812, Std: 0.1469
2. **ca**: 0.1162 ↓ (decrease)
   - Mean SHAP: -0.0400, Std: 0.1439
3. **thal**: 0.0939 ↓ (decrease)
   - Mean SHAP: -0.0129, Std: 0.1116
4. **oldpeak**: 0.0408 ↓ (decrease)
   - Mean SHAP: -0.0278, Std: 0.0615
5. **age**: 0.0378 ↓ (decrease)
   - Mean SHAP: -0.0198, Std: 0.0459

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.0977
- **Max Influence**: 0.1412
- **Feature Complexity**: 0.0402
**Top Influential Features:**
- age: 0.0293 🔼
- restecg: 0.0333 🔼
- cp: 0.0636 🔼
- ca: 0.0973 🔼
- thal: -0.1412 🔽

#### Sample 1:
- **Prediction Sum**: 0.1036
- **Max Influence**: 0.0302
- **Feature Complexity**: 0.0095
**Top Influential Features:**
- thalach: 0.0180 🔼
- chol: -0.0219 🔽
- cp: 0.0237 🔼
- thal: 0.0287 🔼
- ca: 0.0302 🔼

#### Sample 2:
- **Prediction Sum**: 0.0965
- **Max Influence**: 0.0684
- **Feature Complexity**: 0.0212
**Top Influential Features:**
- cp: 0.0279 🔼
- sex: -0.0316 🔽
- age: -0.0420 🔽
- ca: 0.0604 🔼
- thal: 0.0684 🔼

---

### 41. adaboost_StandardScaler (eb5de39e24a41e80acf7be80cf05ec09.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.adaboost_model.AdaBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: eb5de39e24a41e80acf7be80cf05ec09.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0003
- **Standard Deviation**: 0.0451
- **Range**: -0.1004 to 0.1307
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0289

**Top 5 Most Important Features:**
1. **ca**: 0.0938 ↓ (decrease)
   - Mean SHAP: -0.0068, Std: 0.0938
2. **thal**: 0.0860 ↑ (increase)
   - Mean SHAP: 0.0031, Std: 0.0889
3. **cp**: 0.0718 ↑ (increase)
   - Mean SHAP: 0.0168, Std: 0.0768
4. **oldpeak**: 0.0345 ↓ (decrease)
   - Mean SHAP: -0.0131, Std: 0.0352
5. **sex**: 0.0265 ↓ (decrease)
   - Mean SHAP: -0.0021, Std: 0.0290

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.3976
- **Max Influence**: 0.1118
- **Feature Complexity**: 0.0392
**Top Influential Features:**
- exang: 0.0268 🔼
- oldpeak: 0.0277 🔼
- ca: 0.0825 🔼
- thal: 0.1075 🔼
- cp: 0.1118 🔼

#### Sample 1:
- **Prediction Sum**: -0.3122
- **Max Influence**: 0.0956
- **Feature Complexity**: 0.0286
**Top Influential Features:**
- thalach: -0.0260 🔽
- oldpeak: -0.0435 🔽
- cp: -0.0449 🔽
- thal: -0.0682 🔽
- ca: -0.0956 🔽

#### Sample 2:
- **Prediction Sum**: 0.2666
- **Max Influence**: 0.1307
- **Feature Complexity**: 0.0393
**Top Influential Features:**
- sex: 0.0197 🔼
- oldpeak: 0.0305 🔼
- cp: -0.0380 🔽
- ca: 0.0997 🔼
- thal: 0.1307 🔼

---

### 42. catboost_RobustScaler (ebf62cec281df62d14805d972a3001aa.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.catboost_model.CatBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: ebf62cec281df62d14805d972a3001aa.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0183
- **Standard Deviation**: 0.0747
- **Range**: -0.4067 to 0.1264
- **Median**: -0.0004
- **Mean Absolute Value**: 0.0430

**Top 5 Most Important Features:**
1. **ca**: 0.1149 ↓ (decrease)
   - Mean SHAP: -0.0446, Std: 0.1428
2. **cp**: 0.1134 ↓ (decrease)
   - Mean SHAP: -0.0781, Std: 0.1393
3. **thal**: 0.1030 ↓ (decrease)
   - Mean SHAP: -0.0158, Std: 0.1240
4. **age**: 0.0368 ↓ (decrease)
   - Mean SHAP: -0.0179, Std: 0.0411
5. **oldpeak**: 0.0347 ↓ (decrease)
   - Mean SHAP: -0.0242, Std: 0.0519

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.0977
- **Max Influence**: 0.1206
- **Feature Complexity**: 0.0361
**Top Influential Features:**
- thalach: -0.0334 🔽
- restecg: 0.0337 🔼
- cp: 0.0606 🔼
- ca: 0.0967 🔼
- thal: -0.1206 🔽

#### Sample 1:
- **Prediction Sum**: 0.1036
- **Max Influence**: 0.0548
- **Feature Complexity**: 0.0165
**Top Influential Features:**
- ca: 0.0238 🔼
- chol: -0.0239 🔽
- thalach: 0.0277 🔼
- restecg: -0.0451 🔽
- thal: 0.0548 🔼

#### Sample 2:
- **Prediction Sum**: 0.0965
- **Max Influence**: 0.0742
- **Feature Complexity**: 0.0222
**Top Influential Features:**
- sex: -0.0182 🔽
- slope: -0.0237 🔽
- age: -0.0411 🔽
- thal: 0.0600 🔼
- ca: 0.0742 🔼

---

### 43. adaboost_RobustScaler (ef2937f94f8727c4cc9268844672d24c.pkl)

**Model Information:**
- **Model Type**: <class 'models.classification.adaboost_model.AdaBoostModel'>
- **Samples Analyzed**: 30
- **Features**: 13
- **Cache File**: ef2937f94f8727c4cc9268844672d24c.pkl

**SHAP Value Statistics:**
- **Mean SHAP Value**: -0.0003
- **Standard Deviation**: 0.0452
- **Range**: -0.0998 to 0.1307
- **Median**: 0.0000
- **Mean Absolute Value**: 0.0290

**Top 5 Most Important Features:**
1. **ca**: 0.0933 ↓ (decrease)
   - Mean SHAP: -0.0057, Std: 0.0933
2. **thal**: 0.0878 ↑ (increase)
   - Mean SHAP: 0.0033, Std: 0.0907
3. **cp**: 0.0709 ↑ (increase)
   - Mean SHAP: 0.0160, Std: 0.0758
4. **oldpeak**: 0.0353 ↓ (decrease)
   - Mean SHAP: -0.0135, Std: 0.0359
5. **sex**: 0.0260 ↓ (decrease)
   - Mean SHAP: -0.0020, Std: 0.0284

**Sample Analysis (First Few Samples):**
#### Sample 0:
- **Prediction Sum**: 0.3976
- **Max Influence**: 0.1203
- **Feature Complexity**: 0.0400
**Top Influential Features:**
- exang: 0.0255 🔼
- oldpeak: 0.0282 🔼
- ca: 0.0805 🔼
- cp: 0.1040 🔼
- thal: 0.1203 🔼

#### Sample 1:
- **Prediction Sum**: -0.3122
- **Max Influence**: 0.0934
- **Feature Complexity**: 0.0287
**Top Influential Features:**
- thalach: -0.0263 🔽
- oldpeak: -0.0399 🔽
- cp: -0.0496 🔽
- thal: -0.0715 🔽
- ca: -0.0934 🔽

#### Sample 2:
- **Prediction Sum**: 0.2666
- **Max Influence**: 0.1307
- **Feature Complexity**: 0.0388
**Top Influential Features:**
- sex: 0.0186 🔼
- oldpeak: 0.0346 🔼
- cp: -0.0381 🔽
- ca: 0.0956 🔼
- thal: 0.1307 🔼

---

## Feature Interaction Analysis


## Clinical Interpretation of SHAP Features

### Heart Disease Risk Indicators (High SHAP Values)
- **thal**: Thallium stress test - Key diagnostic test
- **cp**: Chest pain type - Primary symptom
- **ca**: Major vessels colored - Blood flow indicator
- **oldpeak**: ST depression - Heart stress indicator
- **exang**: Exercise angina - Functional limitation

### Protective Factors (Negative SHAP Values)
- **thalach**: High max heart rate - Better cardiac function
- Normal resting ECG results
- Absence of exercise-induced angina

## Model-Specific Insights

### Feature Importance Consistency
- **Consistent Top Features**: thal, cp, ca appear in most models
- **Model-Specific Features**: Some models emphasize different features
- **Feature Stability**: Low std in global importance indicates stability

**Summary**: Phan tich SHAP chi tiet cho 43 models voi 870 samples tong cong.
Ket qua cho thay thal, cp, ca la cac features quan trong nhat trong viec du doan benh tim mach.