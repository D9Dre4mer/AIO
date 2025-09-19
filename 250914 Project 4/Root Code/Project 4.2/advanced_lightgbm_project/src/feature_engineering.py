"""
Advanced Feature Engineering Module

This module provides comprehensive feature engineering capabilities including:
- Polynomial features
- Statistical features
- Target encoding
- Feature selection
- Interaction features
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering with multiple techniques
    
    Features:
    - Polynomial feature generation
    - Statistical feature creation
    - Target encoding for categorical variables
    - Feature selection with multiple methods
    - Interaction feature generation
    - Dimensionality reduction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AdvancedFeatureEngineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_config = config.get('feature_engineering', {})
        self.scalers = {}
        self.encoders = {}
        self.selectors = {}
        self.feature_names = {}
        
    def create_polynomial_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                 X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create polynomial features with feature selection
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            y_train: Training target
            
        Returns:
            Tuple of polynomial features for train/val/test
        """
        print("🔧 Creating polynomial features...")
        
        # Get numeric columns only
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("⚠️  No numeric columns found for polynomial features")
            return X_train, X_val, X_test
        
        # Create polynomial features
        degree = self.feature_config.get('polynomial_degree', 2)
        interaction_only = self.feature_config.get('interaction_only', False)
        include_bias = self.feature_config.get('include_bias', False)
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        # Fit on training data
        X_train_poly = poly.fit_transform(X_train[numeric_cols])
        X_val_poly = poly.transform(X_val[numeric_cols])
        X_test_poly = poly.transform(X_test[numeric_cols])
        
        # Get feature names
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Feature selection to avoid curse of dimensionality
        max_features = self.feature_config.get('max_features', 50)
        if X_train_poly.shape[1] > max_features:
            selector = SelectKBest(f_classif, k=max_features)
            X_train_poly = selector.fit_transform(X_train_poly, y_train)
            X_val_poly = selector.transform(X_val_poly)
            X_test_poly = selector.transform(X_test_poly)
            
            # Update feature names
            selected_features = selector.get_support()
            poly_feature_names = poly_feature_names[selected_features]
        
        # Create dataframes
        X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)
        X_val_poly_df = pd.DataFrame(X_val_poly, columns=poly_feature_names, index=X_val.index)
        X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test.index)
        
        # Combine with original categorical features
        categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
        if len(categorical_cols) > 0:
            X_train_poly_df = pd.concat([X_train_poly_df, X_train[categorical_cols]], axis=1)
            X_val_poly_df = pd.concat([X_val_poly_df, X_val[categorical_cols]], axis=1)
            X_test_poly_df = pd.concat([X_test_poly_df, X_test[categorical_cols]], axis=1)
        
        print(f"✅ Polynomial features created: {X_train_poly_df.shape[1]} features")
        
        return X_train_poly_df, X_val_poly_df, X_test_poly_df
    
    def create_statistical_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create statistical features
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            
        Returns:
            Tuple of statistical features for train/val/test
        """
        print("🔧 Creating statistical features...")
        
        # Get numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("⚠️  No numeric columns found for statistical features")
            return X_train, X_val, X_test
        
        def add_statistical_features(df, numeric_cols):
            df_stats = df.copy()
            
            for col in numeric_cols:
                # Percentile features
                df_stats[f'{col}_percentile_25'] = df[col].rank(pct=True)
                df_stats[f'{col}_percentile_75'] = df[col].rank(pct=True)
                
                # Z-score normalization
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_stats[f'{col}_zscore'] = (df[col] - mean_val) / std_val
                
                # Log transformation (if all values are positive)
                if df[col].min() > 0:
                    df_stats[f'{col}_log'] = np.log1p(df[col])
                
                # Square root transformation
                if df[col].min() >= 0:
                    df_stats[f'{col}_sqrt'] = np.sqrt(df[col])
                
                # Box-Cox transformation (simplified)
                if df[col].min() > 0:
                    df_stats[f'{col}_boxcox'] = np.log1p(df[col])
            
            return df_stats
        
        # Apply to all datasets
        X_train_stats = add_statistical_features(X_train, numeric_cols)
        X_val_stats = add_statistical_features(X_val, numeric_cols)
        X_test_stats = add_statistical_features(X_test, numeric_cols)
        
        print(f"✅ Statistical features created: {X_train_stats.shape[1]} features")
        
        return X_train_stats, X_val_stats, X_test_stats
    
    def create_interaction_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create interaction features between numeric columns
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            
        Returns:
            Tuple of interaction features for train/val/test
        """
        print("🔧 Creating interaction features...")
        
        # Get numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("⚠️  Not enough numeric columns for interaction features")
            return X_train, X_val, X_test
        
        def add_interaction_features(df, numeric_cols):
            df_inter = df.copy()
            
            # Limit to top 10 numeric columns to avoid explosion
            top_cols = numeric_cols[:10]
            
            for i, col1 in enumerate(top_cols):
                for col2 in top_cols[i+1:]:
                    # Multiplication
                    df_inter[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    
                    # Division (with small constant to avoid division by zero)
                    df_inter[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    
                    # Addition
                    df_inter[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                    
                    # Subtraction
                    df_inter[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
            
            return df_inter
        
        # Apply to all datasets
        X_train_inter = add_interaction_features(X_train, numeric_cols)
        X_val_inter = add_interaction_features(X_val, numeric_cols)
        X_test_inter = add_interaction_features(X_test, numeric_cols)
        
        print(f"✅ Interaction features created: {X_train_inter.shape[1]} features")
        
        return X_train_inter, X_val_inter, X_test_inter
    
    def apply_target_encoding(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, X_test: pd.DataFrame,
                            categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply target encoding to categorical variables
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            y_train: Training target
            categorical_cols: List of categorical column names
            
        Returns:
            Tuple of target encoded features for train/val/test
        """
        print("🔧 Applying target encoding...")
        
        if not categorical_cols:
            print("⚠️  No categorical columns specified for target encoding")
            return X_train, X_val, X_test
        
        # Filter to existing categorical columns
        existing_cat_cols = [col for col in categorical_cols if col in X_train.columns]
        
        if not existing_cat_cols:
            print("⚠️  No existing categorical columns found")
            return X_train, X_val, X_test
        
        X_train_encoded = X_train.copy()
        X_val_encoded = X_val.copy()
        X_test_encoded = X_test.copy()
        
        for col in existing_cat_cols:
            # Target encoding
            encoder = TargetEncoder(cols=[col], smoothing=1.0)
            X_train_encoded[col] = encoder.fit_transform(X_train[col], y_train)
            X_val_encoded[col] = encoder.transform(X_val[col])
            X_test_encoded[col] = encoder.transform(X_test[col])
            
            # Store encoder for later use
            self.encoders[f'target_{col}'] = encoder
        
        print(f"✅ Target encoding applied to {len(existing_cat_cols)} categorical columns")
        
        return X_train_encoded, X_val_encoded, X_test_encoded
    
    def apply_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, X_test: pd.DataFrame,
                              method: str = 'mutual_info') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply feature selection
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            y_train: Training target
            method: Feature selection method ('mutual_info', 'f_classif', 'rfe')
            
        Returns:
            Tuple of selected features for train/val/test
        """
        print(f"🔧 Applying feature selection with {method}...")
        
        max_features = self.feature_config.get('max_features', 50)
        
        if X_train.shape[1] <= max_features:
            print(f"⚠️  Number of features ({X_train.shape[1]}) <= max_features ({max_features})")
            return X_train, X_val, X_test
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=max_features)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=max_features)
        elif method == 'rfe':
            # Use Random Forest as base estimator for RFE
            base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(base_estimator, n_features_to_select=max_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit selector
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            selected_features = X_train.columns[selector.get_support()].tolist()
        else:
            selected_features = X_train.columns.tolist()
        
        # Create dataframes
        X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_val_selected_df = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val.index)
        X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        # Store selector
        self.selectors[method] = selector
        
        print(f"✅ Feature selection completed: {X_train_selected_df.shape[1]} features selected")
        
        return X_train_selected_df, X_val_selected_df, X_test_selected_df
    
    def apply_scaling(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                     X_test: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply feature scaling
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            method: Scaling method ('standard', 'robust')
            
        Returns:
            Tuple of scaled features for train/val/test
        """
        print(f"🔧 Applying {method} scaling...")
        
        # Get numeric columns only
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("⚠️  No numeric columns found for scaling")
            return X_train, X_val, X_test
        
        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit scaler on training data
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        # Store scaler
        self.scalers[method] = scaler
        
        print(f"✅ {method} scaling applied to {len(numeric_cols)} numeric columns")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_comprehensive_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create comprehensive features using all available techniques
        
        Args:
            X_train, X_val, X_test: Feature dataframes
            y_train: Training target
            
        Returns:
            Tuple of comprehensive features for train/val/test
        """
        print("🚀 Creating comprehensive features...")
        
        # Start with original data
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy()
        X_test_processed = X_test.copy()
        
        # Identify categorical columns
        categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # 1. Statistical features
        if self.feature_config.get('statistical_features', True):
            X_train_processed, X_val_processed, X_test_processed = self.create_statistical_features(
                X_train_processed, X_val_processed, X_test_processed
            )
        
        # 2. Interaction features
        X_train_processed, X_val_processed, X_test_processed = self.create_interaction_features(
            X_train_processed, X_val_processed, X_test_processed
        )
        
        # 3. Target encoding
        if self.feature_config.get('target_encoding', True) and categorical_cols:
            X_train_processed, X_val_processed, X_test_processed = self.apply_target_encoding(
                X_train_processed, y_train, X_val_processed, X_test_processed, categorical_cols
            )
        
        # 4. Polynomial features
        X_train_processed, X_val_processed, X_test_processed = self.create_polynomial_features(
            X_train_processed, X_val_processed, X_test_processed, y_train
        )
        
        # 5. Feature selection
        if self.feature_config.get('feature_selection', True):
            X_train_processed, X_val_processed, X_test_processed = self.apply_feature_selection(
                X_train_processed, y_train, X_val_processed, X_test_processed, method='mutual_info'
            )
        
        # 6. Scaling
        X_train_processed, X_val_processed, X_test_processed = self.apply_scaling(
            X_train_processed, X_val_processed, X_test_processed, method='standard'
        )
        
        print(f"✅ Comprehensive feature engineering completed!")
        print(f"   Original features: {X_train.shape[1]}")
        print(f"   Processed features: {X_train_processed.shape[1]}")
        print(f"   Feature expansion: {X_train_processed.shape[1] / X_train.shape[1]:.2f}x")
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'feature_importance'):
            importance = model.feature_importance(importance_type='gain')
        else:
            print("⚠️  Model does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
