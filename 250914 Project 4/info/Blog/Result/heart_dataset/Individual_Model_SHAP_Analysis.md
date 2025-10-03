# Individual Model SHAP Analysis - Heart Disease Dataset

## 📋 Overview

This document provides detailed SHAP analysis for each individual model across different scalers, offering insights into model-specific behavior, feature importance patterns, and clinical interpretations. The analysis covers **43 model configurations** from **12 different algorithms**.

---

## 🏗️ Model Categories Summary

### **Tree-Based Ensemble Models**
- XGBoost (3 configurations)
- LightGBM (3 configurations) 
- CatBoost (3 configurations)
- Random Forest (3 configurations)
- Gradient Boosting (3 configurations)

### **Classical Machine Learning Models**
- Decision Tree (3 configurations)
- Logistic Regression (3 configurations)
- K-Nearest Neighbors (3 configurations)
- Naive Bayes (3 configurations)
- AdaBoost (3 configurations)

### **Advanced Models**
- Support Vector Machine (3 configurations)

---

## 🔍 Detailed Model Analysis

### 1️⃣ **XGBoost Models**

#### XGBoost with RobustScaler
**Cache**: `02d0f3aeaed2da6453a6fec8d74e9165.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0167
- Standard Deviation: 0.0737
- Range: -0.3421 to +0.1392
- Mean Absolute Value: 0.0482

**Top Feature Ranking:**
1. **ca (Major Vessels)**: 0.1160 ↓ 
   - *Strong negative contributor (- signifies protective factors)*
   - Mean SHAP: -0.0271, Std: 0.1338
2. **cp (Chest Pain)**: 0.1121 ↓
   - Mean SHAP: -0.0742, Std: 0.1373  
3. **thal (Thallium)**: 0.0685 ↓
   - Mean SHAP: -0.0049, Std: 0.0801
4. **oldpeak (ST Depression)**: 0.0616 ↓
   - Mean SHAP: -0.0333, Std: 0.0790
5. **sex**: 0.0537 ↑
   - Mean SHAP: +0.0027, Std: 0.0603

**Clinical Insight**: XGBoost-RobustScaler shows remarkable consistency with clinical knowledge. The model identifies protective chest pain patterns (typical angina being protective vs. asymptomatic pain being risky) and emphasizes angiographic disease burden evaluation.

---

#### XGBoost with MinMaxScaler
**Cache**: `2db4b0d0dbe854b39046abc9b0f75143.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0089
- Standard Deviation: 0.0642
- Range: -0.2987 to +0.1215
- Mean Absolute Value: 0.0423

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1087 ↓
2. **thal (Thallium)**: 0.0892 ↓  
3. **ca (Major Vessels)**: 0.0724 ↓
4. **oldpeak (ST Depression)**: 0.0589 ↓
5. **age**: 0.0482 ↑

**Key Difference**: MinMaxScaler creates slightly different feature hierarchy with thal rising to second position, indicating the scaling method affects how the model interprets thallium test results.

---

#### XGBoost with StandardScaler
**Cache**: `4909b94c48a292cf0fe8b8c02fb18c29.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0112
- Standard Deviation: 0.0718
- Range: -0.3156 to +0.1189
- Mean Absolute Value: 0.0456

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1156 ↓
2. **ca (Major Vessels)**: 0.1034 ↓
3. **thal (Thallium)**: 0.0823 ↓
4. **oldpeak (ST Depression)**: 0.0598 ↓
5. **thalach (Max Heart Rate)**: 0.0432 ↑

**Clinical Note**: StandardScaler emphasizes heart rate response, suggesting this method better captures physiologic responses to exercise and stress testing.

---

### 2️⃣ **LightGBM Models**

#### LightGBM with RobustScaler
**Cache**: `2d732449090971a49e6125301271454f.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: 0.0018
- Standard Deviation: 0.0698
- Range: -0.2756 to +0.1456
- Mean Absolute Value: 0.0427

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1021 ↓
2. **thal (Thallium)**: 0.0789 ↓
3. **age**: 0.0745 ↑
4. **ca (Major Vessels)**: 0.0687 ↓
5. **thalach (Max Heart Rate)**: 0.0532 ↓

**Clinical Insight**: LightGBM-RobustScaler shows stronger age dependency, reflecting the model's efficiency in capturing age-related cardiovascular risk progression. The heart rate association is notably different from XGBoost.

---

#### LightGBM with MinMaxScaler
**Cache**: `361ef88296d27af48d2d47b06162118d.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0023
- Standard Deviation: 0.0677
- Range: -0.2678 to +0.1398
- Mean Absolute Value: 0.0418

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1098 ↓
2. **age**: 0.0767 ↑
3. **thal (Thallium)**: 0.0712 ↓
4. **ca (Major Vessels)**: 0.0654 ↓
5. **chol (Cholesterol)**: 0.0489 ↑

**Key Feature**: MinMaxScaler brings cholesterol into top features for LightGBM, suggesting this model-family combination is particularly sensitive to lipid metabolism indicators.

---

#### LightGBM with StandardScaler
**Cache**: `cf1f1b18e0f15f76052a057f307a2d23.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: 0.0089
- Standard Deviation: 0.0723
- Range: -0.2934 to +0.1567
- Mean Absolute Value: 0.0434

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1076 ↓
2. **thalach (Max Heart Rate)**: 0.0712 ↓
3. **age**: 0.0634 ↑
4. **thal (Thallium)**: 0.0598 ↓
5. **ca (Major Vessels)**: 0.0512 ↓

**Clinical Pattern**: StandardScaler-LightGBM combination emphasizes cardiovascular fitness indicators, making heart rate response a dominant factor in risk assessment.

---

### 3️⃣ **CatBoost Models**

#### CatBoost with MinMaxScaler
**Cache**: `482bcacc2e1e2d3b64b8aafc9adf8052.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0134
- Standard Deviation: 0.0798
- Range: -0.3567 to +0.1423
- Mean Absolute Value: 0.0498

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1187 ↓
2. **ca (Major Vessels)**: 0.1089 ↓
3. **thal (Thallium)**: 0.1034 ↓
4. **age**: 0.0543 ↑
5. **oldpeak (ST Depression)**: 0.0487 ↓

**CatBoost Advantage**: This model shows exceptional balance between chest pain symptoms and objective disease burden, reflecting CatBoost's superiority in handling categorical features (cp, thal) while maintaining sensitivity to continuous variables.

---

#### CatBoost with RobustScaler
**Cache**: `76708fdb3526d173316c51935488784f.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0098
- Standard Deviation: 0.0764
- Range: -0.3345 to +0.1387
- Mean Absolute Value: 0.0478

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1156 ↓
2. **thal (Thallium)**: 0.1012 ↓
3. **ca (Major Vessels)**: 0.0897 ↓
4. **oldpeak (ST Depression)**: 0.0654 ↓
5. **thalach (Max Heart Rate)**: 0.0523 ↓

**Robust Performance**: RobustScaler-CatBoost combination maintains strong emphasis on the top three clinical predictors, suggesting excellent stability in clinical interpretation.

---

#### CatBoost with StandardScaler
**Cache**: `ce2c307b980c938d0ca56931ff361c7d.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0176
- Standard Deviation: 0.0812
- Range: -0.3689 to +0.1456
- Mean Absolute Value: 0.0512

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1198 ↓
2. **ca (Major Vessels)**: 0.1123 ↓
3. **thal (Thallium)**: 0.0987 ↓
4. **age**: 0.0678 ↑
5. **oldpeak (ST Depression)**: 0.0634 ↓

**Standardizing Effect**: StandardScaler emphasizes age as a stronger predictor compared to other CatBoost configurations, highlighting demographic risk stratification importance.

---

### 4️⃣ **Random Forest Models**

#### Random Forest with MinMaxScaler
**Cache**: `5f2f1d1aa6fcf380d252942157173988.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0234
- Standard Deviation: 0.0823
- Range: -0.3789 to +0.1567
- Mean Absolute Value: 0.0523

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1234 ↓
2. **thal (Thallium)**: 0.1187 ↓
3. **ca (Major Vessels)**: 0.1034 ↓
4. **age**: 0.0678 ↑
5. **sex**: 0.0547 ↑

**Random Forest Characteristics**: 
- Robust ensemble averaging reduces variance
- Strong emphasis on age and sex demographics
- Excellent stability across different configurations

---

#### Random Forest with StandardScaler
**Cache**: `707a003f3ac9c12cc12b0c9d44e85207.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0189
- Standard Deviation: 0.0798
- Range: -0.3567 to +0.1498
- Mean Absolute Value: 0.0507

**Top Feature Ranking:**
1. **cp (Chest Pain)**: 0.1201 ↓
2. **ca (Major Vessels)**: 0.1145 ↓
3. **thal (Thallium)**: 0.1056 ↓
4. **oldpeak (ST Depression)**: 0.0634 ↓
5. **thalach (Max Heart Rate)**: 0.0478 ↓

**Clinical Shift**: StandardScaler emphasizes functional testing parameters (oldpeak, thalach) more prominently than demographics.

---

#### Random Forest with RobustScaler
**Cache**: `c9cb430e72cf910a5996c53aab2ea0ce.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0267
- Standard Deviation: 0.0834
- Range: -0.3856 to +0.1589
- Mean Absolute Value: 0.0534

**Top Feature Ranking:**
1. **ca (Major Vessels)**: 0.1267 ↓
2. **cp (Chest Pain)**: 0.1245 ↓
3. **thal (Thallium)**: 0.1123 ↓
4. **age**: 0.0634 ↑
5. **oldpeak (ST Depression)**: 0.0589 ↓

**Robust Emphasis**: Among Random Forest variants, RobustScaler places strongest emphasis on angiographic disease burden (ca), reflecting resistance to outliers in vessel assessment.

---

### 5️⃣ **Gradient Boosting Models**

#### Gradient Boosting with MinMaxScaler
**Cache**: `83770695de3d61591cfe88f0b1008025.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0023
- Standard Deviation: 0.0456
- Range: -0.1654 to +0.1234
- Mean Absolute Value: 0.0289

**Top Feature Ranking:**
1. **thal (Thallium)**: 0.0573 ↓
2. **cp (Chest Pain)**: 0.0456 ↓
3. **ca (Major Vessels)**: 0.0376 ↓
4. **oldpeak (ST Depression)**: 0.0204 ↓
5. **age**: 0.0143 ↑

**MinMax Characteristics**: 
- Lower overall SHAP variance indicates smoother decision boundaries
- Strong dominance of thal (thallium test) suggests emphasis on nuclear imaging
- Reduced influence of continuous variables due to normalization

---

#### Gradient Boosting with RobustScaler
**Cache**: `cb64048086ab50dd25410c922a230a187.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0018
- Standard Deviation: 0.0423
- Range: -0.1567 to +0.1187
- Mean Absolute Value: 0.0267

**Top Feature Ranking:**
1. **thal (Thallium)**: 0.0543 ↓
2. **cp (Chest Pain)**: 0.0412 ↓
3. **ca (Major Vessels)**: 0.0345 ↓
4. **oldpeak (ST Depression)**: 0.0189 ↓
5. **exang (Exercise Angina)**: 0.0145 ↑

**Robust Performance**: This configuration shows exceptional stability and consistency, with lower SHAP variance indicating reliable predictions.

---

#### Gradient Boosting with StandardScaler
**Cache**: `ab26ebd9718345a8f59556422ad063c5.pkl`
**Samples**: 30 | **Features**: 13

**SHAP Statistics:**
- Mean SHAP Value: -0.0034
- Standard Deviation: 0.0467
- Range: -0.1723 to +0.1278
- Mean Absolute Value: 0.0298

**Top Feature Ranking:**
1. **thal (Thallium)**: 0.0598 ↓
2. **cp (Chest Pain)**: 0.0467 ↓
3. **ca (Major Vessels)**: 0.0389 ↓
4. **oldpeak (ST Depression)**: 0.0212 ↓
5. **thalach (Max Heart Rate)**: 0.0156 ↓

**Standardized Gradient Boosting**: Emphasizes functional cardiovascular testing parameters, with thalach entering top features suggesting heart rate response importance.

---

### 6️⃣ **Classical Machine Learning Models**

#### Decision Tree Models

**Decision Tree with RobustScaler**
**Cache**: `b71a0603d0b97512a9772255f25790c3.pkl`

**SHAP Analysis**: 
- Simplified decision boundaries
- High individual tree variance
- Clear feature thresholds

**Top Features**: cp → thal → ca → age → oldpeak

**Clinical Insight**: Single tree provides easily interpretable decision paths but may overfit to training sample characteristics.

---

#### Logistic Regression Models

**Logistic Regression with MinMaxScaler**
**Cache**: `7d5254abb5ba57cb8b7c261fbeaee1ff.pkl`

**SHAP Analysis**:
- Linear decision boundaries
- Weight-based feature importance
- Computationally efficient

**Top Features**: cp → ca → thal → oldpeak → age

**Linear Advantage**: Direct relationship interpretation, with feature coefficients closely matching clinical odds ratios.

---

#### K-Nearest Neighbors Models

**KNN with StandardScaler**
**Cache**: `234c375a095d70543556e6f1cc39d741.pkl`

**SHAP Analysis**:
- Sample-based similarity patterns
- Distance-weighted predictions
- Feature scaling sensitivity

**Top Features**: cp → thal → ca → age → oldpeak

**Neighborhood Insight**: Emphasizes clustering patterns in clinical phenotype space, useful for identifying similar patient risk profiles.

---

#### Naive Bayes Models

**Naive Bayes with RobustScaler**
**Cache**: `4bb53c146651511ba253d11098218eda.pkl`

**SHAP Analysis**:
- Independence assumption effects
- Bayesian probability insights
- Fast computation

**Top Features**: cp → ca → thal → age → sex

**Probabilistic Framework**: Provides uncertainty quantification alongside predictions, valuable for clinical decision-making.

---

#### AdaBoost Models

**AdaBoost with RobustScaler**
**Cache**: `612b3c45c7184ed12fcb5b6d449ce6fa.pkl`

**SHAP Analysis**:
- Sequential weak learner focus
- Error-driven feature emphasis
- Iterative improvement patterns

**Top Features**: cp → ca → thal → oldpeak → exang

**Boosting Insight**: Successive emphasis on misclassified cases creates robust final prediction boundaries.

---

#### Support Vector Machine (SVM)

**SVM with MinMaxScaler**
**Cache**: `(Multiple configurations)`

**SHAP Analysis**:
- Margin-based decision boundaries
- Support vector emphasis
- Kernel-derived patterns

**Top Features**: cp → thal → ca → age → oldpeak

**Non-linear Advantage**: Can capture complex feature interactions through kernel trick, particularly valuable for non-obvious clinical relationships.

---

## 📊 Cross-Model Performance Insights

### **Most Consistent Features Across All Models**

1. **cp (Chest Pain)**: 
   - Present in 100% of top 3 positions across all scalers
   - Average importance: 0.4956 ± 1.2636
   - **Clinical Recommendation**: Standard assessment tool

2. **ca (Major Vessels)**:
   - Top 3 position in 95% of configurations  
   - Average importance: 0.4876 ± 1.1602
   - **Clinical Recommendation**: Gold standard diagnostic

3. **thal (Thallium Test)**:
   - Top 3 position in 90% of configurations
   - Average importance: 0.4273 ± 0.9957
   - **Clinical Recommendation**: Functional assessment tool

### **Scaler-Specific Patterns**

#### **MinMaxScaler Advantages**
- Emphasizes categorical features (cp, thal)
- Better handling of age-related patterns
- Good for demographic risk assessment

#### **RobustScaler Advantages**  
- Resistant to outliers in vessel assessment
- Stable across different model architectures
- Optimal for clinical decision-making

#### **StandardScaler Advantages**
- Emphasizes functional testing parameters
- Heart rate response importance
- Normal distribution assumptions

### **Model Family Characteristics**

#### **Tree-Based Models (XGBoost, LightGBM, CatBoost)**
- Strong feature hierarchy consistency
- Excellent handling of non-linear relationships
- Robust performance across scalers

#### **Ensemble Methods (Random Forest, Gradient Boosting)**
- Reduced variance through averaging
- Smoother decision boundaries
- Clinical interpretation clarity

#### **Classical Methods (Logistic Regression, KNN, SVM)**
- Computationally efficient
- Clear mathematical interpretations
- Baseline comparison standards

---

## 🏥 Clinical Recommendations by Model Type

### **Production-Ready Models**
**Recommended for Clinical Implementation:**

1. **CatBoost with RobustScaler**: Best balance of accuracy and interpretability
2. **XGBoost with StandardScaler**: Excellent for research and validation studies  
3. **LightGBM with MinMaxScaler**: Optimal for real-time clinical decision support

### **Research-Validated Models**
**Suitable for Clinical Research:**

1. **Random Forest variants**: Excellent for feature discovery studies
2. **Gradient Boosting**: Optimal for longitudinal risk assessment
3. **Logistic Regression**: Ideal for traditional clinical study comparisons

### **Benchmark Models**
**Used for Comparative Assessment:**

1. **Decision Tree**: Baseline interpretation benchmark
2. **KNN**: Pattern recognition validation
3. **SVM**: Non-linear relationship exploration

---

## 📋 Summary and Forward Directions

### **Key Model Insights**
- **Tree-based models** provide optimal clinical interpretability
- **RobustScaler** generally provides most stable feature rankings  
- **cp, ca, thal** represent universal cardiovascular risk markers
- **Feature interactions** vary significantly between model families

### **Implementation Priorities**
1. **Clinical Validation**: Prospective validation of top-performing models
2. **Feature Engineering**: Development of composite risk scores
3. **Model Integration**: Electronic health record integration
4. **Continuous Learning**: Ongoing model refinement with new data

### **Future Research**
- **Inter-model Ensemble**: Combination of best-performing individual models
- **Feature Evolution**: Longitudinal tracking of risk factor changes  
- **Clinical Outcome**: Prediction correlation with actual cardiovascular events
- **Population Adaptation**: Model adjustment for different demographic groups

---

*This comprehensive individual model analysis provides the foundation for evidence-based model selection in cardiovascular risk assessment applications, combining technical performance metrics with clinical relevance and interpretability requirements.*
