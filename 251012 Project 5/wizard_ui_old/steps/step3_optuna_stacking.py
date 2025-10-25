"""
Step 3: Optuna Optimization & Stacking Configuration

Created: 2025-01-27
"""

import streamlit as st
import logging
from typing import Dict, List

from ..session_manager import SessionManager
from config import (
    OPTUNA_ENABLE, OPTUNA_TRIALS, OPTUNA_TIMEOUT, OPTUNA_DIRECTION,
    STACKING_ENABLE, STACKING_REQUIRE_MIN_BASE_MODELS, STACKING_BASE_MODELS,
    STACKING_META_LEARNER, STACKING_USE_ORIGINAL_FEATURES,
    STACKING_CV_N_SPLITS, STACKING_CV_STRATIFIED,
    STACKING_CACHE_OUTPUT_DIR, STACKING_CACHE_FORMAT
)

logger = logging.getLogger(__name__)


class OptunaStackingStep:
    """Step 3: Optuna Optimization & Stacking Configuration"""
    
    def __init__(self):
        """Initialize Step 3"""
        self.session_manager = SessionManager()
        
        # Default search spaces for each model
        self.default_search_spaces = {
            'random_forest': {
                'n_estimators': {'min': 50, 'max': 500, 'step': 10},
                'max_depth': {'min': 3, 'max': 20, 'step': 1},
                'max_features': {'choices': ['sqrt', 'log2', None]},
                'min_samples_split': {'min': 2, 'max': 20, 'step': 1},
                'min_samples_leaf': {'min': 1, 'max': 10, 'step': 1}
            },
            'adaboost': {
                'n_estimators': {'min': 50, 'max': 300, 'step': 10},
                'learning_rate': {'min': 0.01, 'max': 2.0, 'step': 0.01}
            },
            'gradient_boosting': {
                'n_estimators': {'min': 50, 'max': 300, 'step': 10},
                'learning_rate': {'min': 0.01, 'max': 0.3, 'step': 0.01},
                'max_depth': {'min': 3, 'max': 10, 'step': 1},
                'subsample': {'min': 0.6, 'max': 1.0, 'step': 0.1}
            },
            'xgboost': {
                'n_estimators': {'min': 50, 'max': 500, 'step': 10},
                'max_depth': {'min': 3, 'max': 10, 'step': 1},
                'eta': {'min': 0.01, 'max': 0.3, 'step': 0.01},
                'subsample': {'min': 0.6, 'max': 1.0, 'step': 0.1},
                'colsample_bytree': {'min': 0.6, 'max': 1.0, 'step': 0.1},
                'min_child_weight': {'min': 1, 'max': 10, 'step': 1},
                'reg_lambda': {'min': 0, 'max': 10, 'step': 0.1},
                'reg_alpha': {'min': 0, 'max': 10, 'step': 0.1}
            },
            'lightgbm': {
                'n_estimators': {'min': 50, 'max': 500, 'step': 10},
                'num_leaves': {'min': 10, 'max': 100, 'step': 5},
                'learning_rate': {'min': 0.01, 'max': 0.3, 'step': 0.01},
                'feature_fraction': {'min': 0.6, 'max': 1.0, 'step': 0.1},
                'bagging_fraction': {'min': 0.6, 'max': 1.0, 'step': 0.1},
                'min_child_samples': {'min': 5, 'max': 50, 'step': 5},
                'lambda_l1': {'min': 0, 'max': 10, 'step': 0.1},
                'lambda_l2': {'min': 0, 'max': 10, 'step': 0.1}
            },
            'catboost': {
                'iterations': {'min': 50, 'max': 500, 'step': 10},
                'depth': {'min': 3, 'max': 10, 'step': 1},
                'learning_rate': {'min': 0.01, 'max': 0.3, 'step': 0.01},
                'l2_leaf_reg': {'min': 1, 'max': 10, 'step': 1},
                'border_count': {'min': 32, 'max': 255, 'step': 32}
            }
        }
    
    def render(self) -> None:
        """Render the complete Step 3 interface"""
        st.title("🔧 Step 3: Optuna Optimization & Stacking Configuration")
        
        st.markdown("""
        **What you'll do here:**
        1. 🎯 Configure Optuna hyperparameter optimization
        2. 🔍 Set search spaces for each model
        3. 🏗️ Configure Stacking ensemble learning
        4. ⚙️ Set cross-validation parameters
        5. ✅ Review and validate configuration
        """)
        
        # Get selected models from previous steps
        step_data = self.session_manager.get_step_data(1)
        if not step_data or 'dataframe' not in step_data:
            st.error("❌ Please complete Step 1 first to select models")
            return
        
        # Create tabs for different configuration sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Optuna Configuration", 
            "🗳️ Voting/Weight Ensemble", 
            "🏗️ Stacking Ensemble", 
            "📊 Review & Validate"
        ])
        
        with tab1:
            self._render_optuna_configuration()
        
        with tab2:
            self._render_voting_weight_ensemble()
        
        with tab3:
            self._render_stacking_configuration()
        
        with tab4:
            self._render_review_validation()
        
        # Step completion
        self._render_step_completion()
    
    def _render_optuna_configuration(self):
        """Render Optuna configuration interface"""
        st.subheader("🎯 Optuna Hyperparameter Optimization")
        
        # Enable Optuna
        enable_optuna = st.checkbox(
            "Enable Optuna Optimization",
            value=OPTUNA_ENABLE,
            help="Enable automatic hyperparameter optimization using Optuna"
        )
        
        if enable_optuna:
            st.success("✅ Optuna optimization enabled")
            
            # Basic Optuna settings
            col1, col2 = st.columns(2)
            
            with col1:
                trials = st.number_input(
                    "Number of Trials",
                    min_value=10,
                    max_value=1000,
                    value=OPTUNA_TRIALS,
                    help="Number of optimization trials to run"
                )
                
                timeout = st.number_input(
                    "Timeout (seconds)",
                    min_value=0,
                    value=OPTUNA_TIMEOUT or 0,
                    help="Maximum time for optimization (0 = no timeout)"
                )
            
            with col2:
                direction = st.selectbox(
                    "Optimization Direction",
                    ["maximize", "minimize"],
                    index=0 if OPTUNA_DIRECTION == "maximize" else 1,
                    help="Whether to maximize or minimize the objective"
                )
                
                metric = st.selectbox(
                    "Optimization Metric",
                    ["accuracy", "f1_score", "precision", "recall"],
                    index=0,
                    help="Metric to optimize"
                )
            
            # Model selection for optimization
            st.markdown("**📋 Select Models for Optimization:**")
            
            # Get available models (this would come from previous steps)
            available_models = [
                'random_forest', 'adaboost', 'gradient_boosting',
                'xgboost', 'lightgbm', 'catboost', 'knn', 'decision_tree',
                'naive_bayes', 'logistic_regression', 'svm', 'linear_svc'
            ]
            
            selected_models = st.multiselect(
                "Choose models to optimize:",
                available_models,
                default=['random_forest', 'xgboost', 'lightgbm', 'catboost'],
                help="Select which models to include in optimization"
            )
            
            if selected_models:
                st.info(f"Selected {len(selected_models)} models for optimization")
                
                # Search space configuration
                st.markdown("**🔍 Search Space Configuration:**")
                
                for model in selected_models:
                    if model in self.default_search_spaces:
                        with st.expander(f"🔧 {model.replace('_', ' ').title()} Parameters", expanded=False):
                            self._render_search_space_config(model)
                    else:
                        st.info(f"ℹ️ {model.replace('_', ' ').title()} uses default parameters")
            
            # Save Optuna configuration
            optuna_config = {
                'enable': enable_optuna,
                'trials': trials,
                'timeout': timeout if timeout > 0 else None,
                'direction': direction,
                'metric': metric,
                'selected_models': selected_models,
                'search_spaces': self._get_search_spaces_config(selected_models)
            }
            
            self.session_manager.update_step_data(3, 'optuna_config', optuna_config)
            
        else:
            st.info("ℹ️ Optuna optimization disabled")
            self.session_manager.update_step_data(3, 'optuna_config', {'enable': False})
    
    def _render_search_space_config(self, model: str):
        """Render search space configuration for a specific model"""
        if model not in self.default_search_spaces:
            st.info(f"No configurable parameters for {model}")
            return
        
        search_space = self.default_search_spaces[model]
        
        for param, config in search_space.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if 'choices' in config:
                    # Categorical parameter
                    choices = config['choices']
                    
                    st.selectbox(
                        f"{param}:",
                        choices,
                        index=0,
                        key=f"optuna_{model}_{param}"
                    )
                else:
                    # Numerical parameter
                    min_val = config['min']
                    max_val = config['max']
                    step = config.get('step', 1)
                    
                    if step < 1:
                        # Float parameter
                        st.slider(
                            f"{param}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val + max_val) / 2,
                            step=step,
                            key=f"optuna_{model}_{param}"
                        )
                    else:
                        # Integer parameter
                        st.slider(
                            f"{param}:",
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int((min_val + max_val) / 2),
                            step=int(step),
                            key=f"optuna_{model}_{param}"
                        )
            
            with col2:
                st.checkbox(
                    "Enable",
                    value=True,
                    key=f"enable_{model}_{param}"
                )
    
    def _get_search_spaces_config(self, selected_models: List[str]) -> Dict[str, Dict]:
        """Get search spaces configuration from UI"""
        search_spaces = {}
        
        for model in selected_models:
            if model in self.default_search_spaces:
                model_config = {}
                for param in self.default_search_spaces[model].keys():
                    if st.session_state.get(f"enable_{model}_{param}", True):
                        model_config[param] = st.session_state.get(f"optuna_{model}_{param}")
                search_spaces[model] = model_config
        
        return search_spaces
    
    def _render_voting_weight_ensemble(self):
        """Render Voting/Weight ensemble configuration for traditional models"""
        st.subheader("🗳️ Voting/Weight Ensemble Configuration")
        
        st.markdown("""
        **Voting/Weight Ensemble** combines predictions from multiple traditional models using:
        - **Hard Voting**: Majority vote for classification
        - **Soft Voting**: Weighted average of probabilities
        - **Custom Weights**: Assign specific weights to each model
        
        **Suitable for**: Traditional ML models (Random Forest, SVM, Logistic Regression, etc.)
        """)
        
        # Enable Voting/Weight ensemble
        enable_voting = st.checkbox(
            "Enable Voting/Weight Ensemble",
            value=False,
            help="Enable ensemble learning with voting or weighted predictions"
        )
        
        if enable_voting:
            # Traditional models for voting (not tree-based)
            traditional_models = [
                'random_forest', 'adaboost', 'gradient_boosting',
                'knn', 'decision_tree', 'naive_bayes', 
                'logistic_regression', 'svm', 'linear_svc'
            ]
            
            # Model selection for voting
            st.markdown("**📋 Select Traditional Models for Voting/Weight Ensemble:**")
            selected_voting_models = st.multiselect(
                "Choose models for voting ensemble:",
                traditional_models,
                default=['random_forest', 'logistic_regression', 'svm'],
                help="Select which traditional models to include in voting ensemble"
            )
            
            if selected_voting_models:
                st.info(f"Selected {len(selected_voting_models)} models for voting ensemble")
                
                # Voting method selection
                st.markdown("**🗳️ Voting Method:**")
                voting_method = st.selectbox(
                    "Choose voting method:",
                    ["hard", "soft"],
                    help="Hard voting uses majority vote, soft voting uses probability average"
                )
                
                # Weight configuration
                st.markdown("**⚖️ Model Weights:**")
                use_custom_weights = st.checkbox(
                    "Use custom weights",
                    help="Assign specific weights to each model"
                )
                
                weights = {}
                if use_custom_weights:
                    st.markdown("**Set weights for each model:**")
                    total_weight = 0
                    for model in selected_voting_models:
                        weight = st.number_input(
                            f"Weight for {model}:",
                            min_value=0.0,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            key=f"voting_weight_{model}"
                        )
                        weights[model] = weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        st.info(f"Total weight: {total_weight:.1f}")
                        # Normalize weights
                        normalized_weights = {k: v/total_weight for k, v in weights.items()}
                        st.json(normalized_weights)
                else:
                    # Equal weights
                    weights = {model: 1.0 for model in selected_voting_models}
                    st.info("Using equal weights for all models")
                
                # Save voting configuration
                voting_config = {
                    'enable_voting': enable_voting,
                    'selected_models': selected_voting_models,
                    'voting_method': voting_method,
                    'use_custom_weights': use_custom_weights,
                    'weights': weights
                }
                
                self.session_manager.update_step_data(3, 'voting_config', voting_config)
                
                # Display configuration summary
                st.markdown("**📊 Configuration Summary:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Models", len(selected_voting_models))
                    st.metric("Voting Method", voting_method.title())
                with col2:
                    st.metric("Custom Weights", "Yes" if use_custom_weights else "No")
                    st.metric("Total Weight", f"{sum(weights.values()):.1f}")
            else:
                st.warning("Please select at least one model for voting ensemble")
        else:
            # Clear voting configuration if disabled
            self.session_manager.update_step_data(3, 'voting_config', {'enable_voting': False})

    def _render_stacking_configuration(self):
        """Render Stacking configuration interface for tree-based models"""
        st.subheader("🏗️ Stacking Ensemble Configuration")
        
        st.markdown("""
        **Stacking Ensemble** uses a meta-learner to combine predictions from tree-based models.
        This is specifically designed for tree models (XGBoost, LightGBM, CatBoost, etc.)
        The meta-learner learns how to best combine the base model predictions.
        
        **Suitable for**: Tree-based models (XGBoost, LightGBM, CatBoost, Random Forest, etc.)
        """)
        
        # Enable Stacking
        enable_stacking = st.checkbox(
            "Enable Stacking Ensemble",
            value=STACKING_ENABLE,
            help="Enable stacking ensemble learning with meta-learner"
        )
        
        if enable_stacking:
            st.success("✅ Stacking ensemble enabled")
            
            # Get selected models for stacking
            optuna_config = self.session_manager.get_step_data(3).get('optuna_config', {})
            available_models = optuna_config.get('selected_models', [])
            
            if len(available_models) < STACKING_REQUIRE_MIN_BASE_MODELS:
                st.warning(f"⚠️ Need at least {STACKING_REQUIRE_MIN_BASE_MODELS} models for stacking. "
                          f"Currently have {len(available_models)} models.")
                enable_stacking = False
            
            if enable_stacking:
                # Base models selection (only tree-based models)
                st.markdown("**📋 Select Tree-Based Models for Stacking:**")
                
                # Tree-based models only
                tree_models = [
                    'random_forest', 'xgboost', 'lightgbm', 'catboost',
                    'adaboost', 'gradient_boosting', 'extra_trees'
                ]
                
                recommended_models = [m for m in STACKING_BASE_MODELS if m in tree_models]
                base_models = st.multiselect(
                    "Choose tree-based models:",
                    tree_models,
                    default=recommended_models,
                    help=f"Tree-based models only. Recommended: {', '.join(STACKING_BASE_MODELS)}"
                )
                
                if len(base_models) < STACKING_REQUIRE_MIN_BASE_MODELS:
                    st.error(f"❌ Need at least {STACKING_REQUIRE_MIN_BASE_MODELS} base models")
                    enable_stacking = False
                
                if enable_stacking:
                    # Meta-learner selection
                    st.markdown("**🎯 Meta-learner Configuration:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        meta_learner = st.selectbox(
                            "Meta-learner:",
                            ["logistic_regression", "lightgbm"],
                            index=0 if STACKING_META_LEARNER == "logistic_regression" else 1,
                            help="Final estimator for stacking"
                        )
                    
                    with col2:
                        use_original_features = st.checkbox(
                            "Use Original Features",
                            value=STACKING_USE_ORIGINAL_FEATURES,
                            help="Include original features along with predictions"
                        )
                    
                    # Cross-validation settings
                    st.markdown("**🔄 Cross-Validation Settings:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cv_folds = st.number_input(
                            "Number of CV Folds",
                            min_value=3,
                            max_value=10,
                            value=STACKING_CV_N_SPLITS,
                            help="Number of cross-validation folds"
                        )
                    
                    with col2:
                        stratified = st.checkbox(
                            "Stratified CV",
                            value=STACKING_CV_STRATIFIED,
                            help="Use stratified cross-validation"
                        )
                    
                    # Cache settings
                    st.markdown("**💾 Cache Settings:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cache_format = st.selectbox(
                            "Cache Format",
                            ["parquet", "csv"],
                            index=0 if STACKING_CACHE_FORMAT == "parquet" else 1,
                            help="Format for storing OOF predictions"
                        )
                    
                    with col2:
                        cache_dir = st.text_input(
                            "Cache Directory",
                            value=STACKING_CACHE_OUTPUT_DIR,
                            help="Directory for stacking cache"
                        )
                    
                    # Save Stacking configuration
                    stacking_config = {
                        'enable': enable_stacking,
                        'base_models': base_models,
                        'meta_learner': meta_learner,
                        'use_original_features': use_original_features,
                        'cv_folds': cv_folds,
                        'stratified': stratified,
                        'cache_format': cache_format,
                        'cache_dir': cache_dir
                    }
                    
                    self.session_manager.update_step_data(3, 'stacking_config', stacking_config)
                    
                    # Display configuration summary
                    st.markdown("**📊 Configuration Summary:**")
                    st.json(stacking_config)
        
        if not enable_stacking:
            st.info("ℹ️ Stacking ensemble disabled")
            self.session_manager.update_step_data(3, 'stacking_config', {'enable': False})
    
    def _render_review_validation(self):
        """Render review and validation interface"""
        st.subheader("📊 Configuration Review & Validation")
        
        # Get configurations
        optuna_config = self.session_manager.get_step_data(3).get('optuna_config', {})
        voting_config = self.session_manager.get_step_data(3).get('voting_config', {})
        stacking_config = self.session_manager.get_step_data(3).get('stacking_config', {})
        
        # Optuna review
        st.markdown("**🎯 Optuna Configuration:**")
        if optuna_config.get('enable', False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Trials", optuna_config.get('trials', 0))
            
            with col2:
                timeout = optuna_config.get('timeout')
                st.metric("Timeout", f"{timeout}s" if timeout else "None")
            
            with col3:
                st.metric("Models", len(optuna_config.get('selected_models', [])))
            
            st.success("✅ Optuna optimization configured")
        else:
            st.info("ℹ️ Optuna optimization disabled")
        
        # Voting ensemble review
        st.markdown("**🗳️ Voting/Weight Ensemble Configuration:**")
        if voting_config.get('enable_voting', False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Models", len(voting_config.get('selected_models', [])))
            
            with col2:
                st.metric("Voting Method", voting_config.get('voting_method', 'hard').title())
            
            with col3:
                st.metric("Custom Weights", "Yes" if voting_config.get('use_custom_weights', False) else "No")
            
            st.success("✅ Voting/Weight ensemble enabled")
        else:
            st.info("ℹ️ Voting/Weight ensemble disabled")
        
        # Stacking review
        st.markdown("**🏗️ Stacking Configuration:**")
        if stacking_config.get('enable', False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Base Models", len(stacking_config.get('base_models', [])))
            
            with col2:
                st.metric("CV Folds", stacking_config.get('cv_folds', 0))
            
            with col3:
                meta_learner = stacking_config.get('meta_learner', 'N/A')
                st.metric("Meta-learner", meta_learner.replace('_', ' ').title())
            
            st.success("✅ Stacking ensemble configured")
        else:
            st.info("ℹ️ Stacking ensemble disabled")
        
        # Validation
        st.markdown("**✅ Configuration Validation:**")
        
        validation_passed = True
        validation_messages = []
        
        # Check Optuna configuration
        if optuna_config.get('enable', False):
            if optuna_config.get('trials', 0) < 10:
                validation_passed = False
                validation_messages.append("❌ Optuna trials should be at least 10")
            
            if not optuna_config.get('selected_models'):
                validation_passed = False
                validation_messages.append("❌ No models selected for Optuna optimization")
        
        # Check Stacking configuration
        if stacking_config.get('enable', False):
            base_models = stacking_config.get('base_models', [])
            if len(base_models) < STACKING_REQUIRE_MIN_BASE_MODELS:
                validation_passed = False
                validation_messages.append(f"❌ Need at least {STACKING_REQUIRE_MIN_BASE_MODELS} base models for stacking")
            
            if not base_models:
                validation_passed = False
                validation_messages.append("❌ No base models selected for stacking")
        
        # Display validation results
        if validation_passed:
            st.success("✅ All configurations are valid")
        else:
            st.error("❌ Configuration validation failed:")
            for message in validation_messages:
                st.error(message)
        
        # Save validation result
        self.session_manager.update_step_data(3, 'validation_passed', validation_passed)
        self.session_manager.update_step_data(3, 'validation_messages', validation_messages)
    
    def _render_step_completion(self):
        """Render step completion section"""
        st.subheader("✅ Step Completion")
        
        step_data = self.session_manager.get_step_data(3)
        is_complete = (
            'optuna_config' in step_data and
            'voting_config' in step_data and
            'stacking_config' in step_data and
            step_data.get('validation_passed', False)
        )
        
        if is_complete:
            st.success("🎯 Step 3 completed successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                optuna_enabled = step_data.get('optuna_config', {}).get('enable', False)
                st.metric("Optuna", "Enabled" if optuna_enabled else "Disabled")
            
            with col2:
                voting_enabled = step_data.get('voting_config', {}).get('enable_voting', False)
                st.metric("Voting Ensemble", "Enabled" if voting_enabled else "Disabled")
            
            with col3:
                stacking_enabled = step_data.get('stacking_config', {}).get('enable', False)
                st.metric("Stacking", "Enabled" if stacking_enabled else "Disabled")
            
            with col4:
                models_count = len(step_data.get('optuna_config', {}).get('selected_models', []))
                st.metric("Models", models_count)
            
            self.session_manager.set_progress(3, 1.0)
            
        else:
            st.info("📝 Complete the configuration to proceed to the next step.")
            self.session_manager.set_progress(3, 0.0)
    
    def validate_step(self) -> bool:
        """Validate if Step 3 is complete"""
        step_data = self.session_manager.get_step_data(3)
        
        required_fields = ['optuna_config', 'voting_config', 'stacking_config', 'validation_passed']
        for field in required_fields:
            if field not in step_data:
                return False
        
        return step_data.get('validation_passed', False)
