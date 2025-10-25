"""
Step 5: SHAP Visualization & Model Interpretation

Created: 2025-01-27
"""

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, List
from datetime import datetime
import os
from pathlib import Path

from ..session_manager import SessionManager
from config import SHAP_ENABLE, SHAP_SAMPLE_SIZE, SHAP_OUTPUT_DIR
from visualization import generate_comprehensive_shap_analysis
from confusion_matrix_cache import confusion_matrix_cache

logger = logging.getLogger(__name__)


class SHAPVisualizationStep:
    """Step 5: SHAP Visualization & Model Interpretation"""
    
    def __init__(self):
        """Initialize Step 5"""
        self.session_manager = SessionManager()
        self.confusion_matrix_cache = confusion_matrix_cache
    
    def render(self) -> None:
        """Render the complete Step 5 interface"""
        st.title("📊 Step 5: SHAP Visualization & Model Interpretation")
        
        st.markdown("""
        **What you'll do here:**
        1. 🎯 Select trained models for interpretation
        2. 📊 Generate SHAP visualizations
        3. 🔍 Analyze feature importance
        4. 📈 Create confusion matrices from cache
        5. 💾 Download and save results
        """)
        
        # Check if models are available
        available_caches = self.confusion_matrix_cache.list_available_caches()
        
        if not available_caches:
            st.error("❌ No trained models found. Please complete training first.")
            return
        
        # Create tabs for different visualization sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Model Selection", 
            "📊 SHAP Analysis", 
            "📈 Confusion Matrices", 
            "💾 Results Summary"
        ])
        
        with tab1:
            self._render_model_selection(available_caches)
        
        with tab2:
            self._render_shap_analysis()
        
        with tab3:
            self._render_confusion_matrices()
        
        with tab4:
            self._render_results_summary()
        
        # Step completion
        self._render_step_completion()
    
    def _render_model_selection(self, available_caches: List[Dict[str, Any]]):
        """Render model selection interface"""
        st.subheader("🎯 Select Models for Analysis")
        
        # Filter models with eval_predictions
        models_with_predictions = [cache for cache in available_caches if cache['has_eval_predictions']]
        
        if not models_with_predictions:
            st.error("❌ No models with evaluation predictions found")
            return
        
        st.info(f"Found {len(models_with_predictions)} trained models with evaluation data")
        
        # Model selection
        model_options = []
        for cache in models_with_predictions:
            model_name = f"{cache['model_key']} ({cache['dataset_id']})"
            model_options.append((model_name, cache))
        
        selected_model_names = st.multiselect(
            "Select models to analyze:",
            [option[0] for option in model_options],
            default=[option[0] for option in model_options[:3]],  # Select first 3 by default
            help="Choose which trained models to include in the analysis"
        )
        
        # Get selected model data
        selected_models = []
        for name in selected_model_names:
            for option_name, cache_data in model_options:
                if name == option_name:
                    selected_models.append(cache_data)
                    break
        
        if selected_models:
            st.success(f"✅ Selected {len(selected_models)} models for analysis")
            
            # Display selected models info
            st.markdown("**📋 Selected Models:**")
            
            for i, model in enumerate(selected_models, 1):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{i}.** {model['model_key']}")
                
                with col2:
                    st.write(f"Dataset: {model['dataset_id']}")
                
                with col3:
                    st.write(f"Accuracy: {model.get('accuracy', 'N/A')}")
                
                with col4:
                    has_shap = "✅" if model['has_shap_sample'] else "❌"
                    st.write(f"SHAP: {has_shap}")
            
            # Save selected models
            self.session_manager.update_step_data(5, 'selected_models', selected_models)
            
            # SHAP configuration
            st.markdown("**⚙️ SHAP Configuration:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                enable_shap = st.checkbox(
                    "Enable SHAP Analysis",
                    value=SHAP_ENABLE,
                    help="Generate SHAP visualizations for selected models"
                )
                
                sample_size = st.number_input(
                    "Sample Size for SHAP",
                    min_value=100,
                    max_value=10000,
                    value=SHAP_SAMPLE_SIZE,
                    help="Number of samples to use for SHAP analysis"
                )
            
            with col2:
                output_dir = st.text_input(
                    "Output Directory",
                    value=SHAP_OUTPUT_DIR,
                    help="Directory to save SHAP plots"
                )
                
                plot_types = st.multiselect(
                    "Plot Types",
                    ["summary", "bar", "dependence", "waterfall"],
                    default=["summary", "bar", "dependence"],
                    help="Types of SHAP plots to generate"
                )
            
            # Save SHAP configuration
            shap_config = {
                'enable': enable_shap,
                'sample_size': sample_size,
                'output_dir': output_dir,
                'plot_types': plot_types
            }
            
            self.session_manager.update_step_data(5, 'shap_config', shap_config)
            
        else:
            st.warning("⚠️ Please select at least one model for analysis")
            self.session_manager.update_step_data(5, 'selected_models', [])
    
    def _render_shap_analysis(self):
        """Render SHAP analysis interface"""
        st.subheader("📊 SHAP Analysis")
        
        # Get configuration
        shap_config = self.session_manager.get_step_data(5).get('shap_config', {})
        selected_models = self.session_manager.get_step_data(5).get('selected_models', [])
        
        if not shap_config.get('enable', False):
            st.info("ℹ️ SHAP analysis is disabled")
            return
        
        if not selected_models:
            st.warning("⚠️ No models selected for SHAP analysis")
            return
        
        # SHAP analysis controls
        st.markdown("**🔧 SHAP Analysis Controls:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Generate SHAP Analysis", type="primary"):
                self._generate_shap_analysis(selected_models, shap_config)
        
        with col2:
            if st.button("📊 Preview SHAP Sample"):
                self._preview_shap_sample(selected_models[0])
        
        # Display SHAP results if available
        shap_results = self.session_manager.get_step_data(5).get('shap_results', {})
        
        if shap_results:
            st.markdown("**📈 SHAP Analysis Results:**")
            
            for model_key, results in shap_results.items():
                with st.expander(f"📊 {model_key} SHAP Results", expanded=False):
                    self._display_shap_results(model_key, results)
    
    def _generate_shap_analysis(self, selected_models: List[Dict], shap_config: Dict):
        """Generate SHAP analysis for selected models"""
        with st.spinner("Generating SHAP analysis..."):
            try:
                shap_results = {}
                
                for model_info in selected_models:
                    model_key = model_info['model_key']
                    
                    st.write(f"🔍 Analyzing {model_key}...")
                    
                    try:
                        # Load model and data from cache
                        from cache_manager import cache_manager
                        
                        cache_data = cache_manager.load_model_cache(
                            model_info['model_key'],
                            model_info['dataset_id'],
                            model_info['config_hash']
                        )
                        
                        model = cache_data['model']
                        shap_sample = cache_data.get('shap_sample')
                        
                        if shap_sample is None:
                            st.warning(f"⚠️ No SHAP sample available for {model_key}")
                            continue
                        
                        # Prepare sample data
                        sample_size = min(shap_config['sample_size'], len(shap_sample))
                        X_sample = shap_sample.iloc[:sample_size]
                        
                        # Generate comprehensive SHAP analysis
                        output_dir = shap_config['output_dir']
                        plot_types = shap_config['plot_types']
                        
                        result = generate_comprehensive_shap_analysis(
                            model=model,
                            X_sample=X_sample,
                            feature_names=cache_data.get('feature_names'),
                            output_dir=output_dir,
                            model_name=model_key,
                            plot_types=plot_types
                        )
                        
                        shap_results[model_key] = result
                        st.success(f"✅ SHAP analysis completed for {model_key}")
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing {model_key}: {str(e)}")
                        continue
                
                # Save results
                self.session_manager.update_step_data(5, 'shap_results', shap_results)
                
                if shap_results:
                    st.success(f"🎉 SHAP analysis completed for {len(shap_results)} models!")
                else:
                    st.warning("⚠️ No SHAP analysis results generated")
                
            except Exception as e:
                st.error(f"❌ Error generating SHAP analysis: {str(e)}")
    
    def _preview_shap_sample(self, model_info: Dict):
        """Preview SHAP sample data"""
        try:
            from cache_manager import cache_manager
            
            cache_data = cache_manager.load_model_cache(
                model_info['model_key'],
                model_info['dataset_id'],
                model_info['config_hash']
            )
            
            shap_sample = cache_data.get('shap_sample')
            
            if shap_sample is None:
                st.warning("No SHAP sample available")
                return
            
            st.markdown("**📋 SHAP Sample Preview:**")
            st.dataframe(shap_sample.head(10))
            
            st.markdown("**📊 Sample Statistics:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", len(shap_sample))
            
            with col2:
                st.metric("Columns", len(shap_sample.columns))
            
            with col3:
                memory_mb = shap_sample.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Memory", f"{memory_mb:.2f} MB")
                
        except Exception as e:
            st.error(f"❌ Error previewing SHAP sample: {str(e)}")
    
    def _display_shap_results(self, model_key: str, results: Dict):
        """Display SHAP results for a specific model"""
        try:
            # Display generated plots
            if 'plots' in results:
                st.markdown("**📊 Generated Plots:**")
                
                for plot_type, plot_path in results['plots'].items():
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption=f"{plot_type.title()} Plot")
                    else:
                        st.warning(f"Plot not found: {plot_path}")
            
            # Display feature importance
            if 'feature_importance' in results:
                st.markdown("**🎯 Feature Importance:**")
                
                importance_df = pd.DataFrame(results['feature_importance'])
                st.dataframe(importance_df)
            
            # Display summary statistics
            if 'summary' in results:
                st.markdown("**📈 Summary Statistics:**")
                st.json(results['summary'])
                
        except Exception as e:
            st.error(f"❌ Error displaying results for {model_key}: {str(e)}")
    
    def _render_confusion_matrices(self):
        """Render confusion matrices interface"""
        st.subheader("📈 Confusion Matrices from Cache")
        
        selected_models = self.session_manager.get_step_data(5).get('selected_models', [])
        
        if not selected_models:
            st.warning("⚠️ No models selected for confusion matrix generation")
            st.info("💡 Please go to 'Model Selection' tab and select some models first")
            return
        
        # Confusion matrix configuration
        st.markdown("**⚙️ Confusion Matrix Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            normalize = st.selectbox(
                "Normalization",
                ["true", "pred", "all", None],
                index=0,
                help="How to normalize the confusion matrix"
            )
        
        with col2:
            save_plots = st.checkbox(
                "Save Plots",
                value=True,
                help="Save confusion matrix plots to disk"
            )
            
            show_metrics = st.checkbox(
                "Show Metrics",
                value=True,
                help="Display classification metrics"
            )
        
        # Generate confusion matrices
        if st.button("📊 Generate Confusion Matrices", type="primary"):
            self._generate_confusion_matrices(selected_models, normalize, save_plots, show_metrics)
        
        # Display confusion matrix results from session
        cm_results = self.session_manager.get_step_data(5).get('confusion_matrix_results', {})
        
        if cm_results:
            st.markdown("**📈 Confusion Matrix Results (from session):**")
            
            for model_key, result in cm_results.items():
                with st.expander(f"📊 {model_key} Confusion Matrix", expanded=False):
                    self._display_confusion_matrix_result(model_key, result, show_metrics)
    
    def _generate_confusion_matrices(self, selected_models: List[Dict], normalize: str, 
                                   save_plots: bool, show_metrics: bool):
        """Generate confusion matrices for selected models"""
        with st.spinner("Generating confusion matrices..."):
            try:
                cm_results = {}
                
                for model_info in selected_models:
                    model_key = model_info['model_key']
                    
                    st.write(f"📊 Generating confusion matrix for {model_key}...")
                    
                    try:
                        # Generate confusion matrix
                        result = self.confusion_matrix_cache.generate_confusion_matrix_from_cache(
                            model_key=model_info['model_key'],
                            dataset_id=model_info['dataset_id'],
                            config_hash=model_info['config_hash'],
                            normalize=normalize
                        )
                        
                        # Save plot if requested
                        if save_plots:
                            output_dir = Path(SHAP_OUTPUT_DIR)
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            plot_filename = f"{model_key}_{model_info['dataset_id']}_confusion_matrix.png"
                            plot_path = output_dir / plot_filename
                            
                            result['plot'].savefig(plot_path, dpi=300, bbox_inches='tight')
                            result['plot_path'] = str(plot_path)
                        
                        cm_results[model_key] = result
                        st.success(f"✅ Confusion matrix generated for {model_key}")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating confusion matrix for {model_key}: {str(e)}")
                        continue
                
                # Save results
                self.session_manager.update_step_data(5, 'confusion_matrix_results', cm_results)
                
                if cm_results:
                    st.success(f"🎉 Confusion matrices generated for {len(cm_results)} models!")
                    
                    # Display results immediately
                    st.markdown("**📈 Confusion Matrix Results:**")
                    
                    for model_key, result in cm_results.items():
                        with st.expander(f"📊 {model_key} Confusion Matrix", expanded=True):
                            self._display_confusion_matrix_result(model_key, result, show_metrics)
                else:
                    st.warning("⚠️ No confusion matrices generated")
                
            except Exception as e:
                st.error(f"❌ Error generating confusion matrices: {str(e)}")
    
    def _display_confusion_matrix_result(self, model_key: str, result: Dict, show_metrics: bool):
        """Display confusion matrix result"""
        try:
            # Display confusion matrix plot
            if 'plot' in result:
                st.pyplot(result['plot'])
            
            # Display plot path if saved
            if 'plot_path' in result:
                st.info(f"📁 Plot saved to: {result['plot_path']}")
            
            # Display metrics if requested
            if show_metrics and 'metrics' in result:
                st.markdown("**📊 Classification Metrics:**")
                
                metrics = result['metrics']
                
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                
                with col2:
                    st.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
                
                with col3:
                    st.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")
                
                with col4:
                    st.metric("Classes", len(metrics['class_metrics']))
                
                # Per-class metrics
                if 'class_metrics' in metrics:
                    st.markdown("**📋 Per-Class Metrics:**")
                    
                    class_metrics_df = pd.DataFrame(metrics['class_metrics']).T
                    st.dataframe(class_metrics_df)
            
        except Exception as e:
            st.error(f"❌ Error displaying confusion matrix result: {str(e)}")
    
    def _render_results_summary(self):
        """Render results summary interface"""
        st.subheader("💾 Results Summary")
        
        # Get all results
        shap_results = self.session_manager.get_step_data(5).get('shap_results', {})
        cm_results = self.session_manager.get_step_data(5).get('confusion_matrix_results', {})
        selected_models = self.session_manager.get_step_data(5).get('selected_models', [])
        
        if not selected_models:
            st.warning("⚠️ No analysis results available")
            return
        
        # Summary statistics
        st.markdown("**📊 Analysis Summary:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Analyzed", len(selected_models))
        
        with col2:
            st.metric("SHAP Results", len(shap_results))
        
        with col3:
            st.metric("Confusion Matrices", len(cm_results))
        
        with col4:
            total_plots = len(shap_results) + len(cm_results)
            st.metric("Total Plots", total_plots)
        
        # Model performance comparison
        if cm_results:
            st.markdown("**🏆 Model Performance Comparison:**")
            
            performance_data = []
            for model_key, result in cm_results.items():
                metrics = result['metrics']
                performance_data.append({
                    'Model': model_key,
                    'Accuracy': metrics['accuracy'],
                    'Macro F1': metrics['macro_f1'],
                    'Weighted F1': metrics['weighted_f1']
                })
            
            if performance_data:
                performance_df = pd.DataFrame(performance_data)
                performance_df = performance_df.sort_values('Accuracy', ascending=False)
                
                st.dataframe(performance_df, width='stretch')
                
                # Best model
                best_model = performance_df.iloc[0]
                st.success(f"🏆 Best performing model: **{best_model['Model']}** "
                          f"(Accuracy: {best_model['Accuracy']:.3f})")
        
        # Download options
        st.markdown("**💾 Download Options:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download SHAP Results"):
                self._download_shap_results(shap_results)
        
        with col2:
            if st.button("📈 Download Confusion Matrices"):
                self._download_confusion_matrices(cm_results)
        
        # Generate comprehensive report
        if st.button("📋 Generate Comprehensive Report", type="primary"):
            self._generate_comprehensive_report(shap_results, cm_results, selected_models)
    
    def _download_shap_results(self, shap_results: Dict):
        """Download SHAP results"""
        try:
            if not shap_results:
                st.warning("No SHAP results to download")
                return
            
            # Create summary report
            report_data = []
            for model_key, result in shap_results.items():
                report_data.append({
                    'Model': model_key,
                    'SHAP Analysis': 'Completed',
                    'Plots Generated': len(result.get('plots', {})),
                    'Feature Importance': 'Available' if 'feature_importance' in result else 'N/A'
                })
            
            report_df = pd.DataFrame(report_data)
            
            # Convert to CSV
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download SHAP Report (CSV)",
                data=csv,
                file_name=f"shap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error downloading SHAP results: {str(e)}")
    
    def _download_confusion_matrices(self, cm_results: Dict):
        """Download confusion matrix results"""
        try:
            if not cm_results:
                st.warning("No confusion matrix results to download")
                return
            
            # Create summary report
            report_data = []
            for model_key, result in cm_results.items():
                metrics = result['metrics']
                report_data.append({
                    'Model': model_key,
                    'Accuracy': metrics['accuracy'],
                    'Macro F1': metrics['macro_f1'],
                    'Weighted F1': metrics['weighted_f1'],
                    'Classes': len(metrics['class_metrics'])
                })
            
            report_df = pd.DataFrame(report_data)
            
            # Convert to CSV
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Confusion Matrix Report (CSV)",
                data=csv,
                file_name=f"confusion_matrix_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error downloading confusion matrix results: {str(e)}")
    
    def _generate_comprehensive_report(self, shap_results: Dict, cm_results: Dict, selected_models: List):
        """Generate comprehensive analysis report"""
        with st.spinner("Generating comprehensive report..."):
            try:
                # Create comprehensive report
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'models_analyzed': len(selected_models),
                    'shap_analysis': {
                        'enabled': len(shap_results) > 0,
                        'models_completed': len(shap_results),
                        'results': shap_results
                    },
                    'confusion_matrices': {
                        'enabled': len(cm_results) > 0,
                        'models_completed': len(cm_results),
                        'results': cm_results
                    },
                    'summary': {
                        'total_models': len(selected_models),
                        'successful_analyses': len(shap_results) + len(cm_results),
                        'best_model': None
                    }
                }
                
                # Find best model
                if cm_results:
                    best_accuracy = 0
                    best_model = None
                    for model_key, result in cm_results.items():
                        accuracy = result['metrics']['accuracy']
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model_key
                    
                    report['summary']['best_model'] = {
                        'name': best_model,
                        'accuracy': best_accuracy
                    }
                
                # Save report
                self.session_manager.update_step_data(5, 'comprehensive_report', report)
                
                st.success("✅ Comprehensive report generated!")
                
                # Display report summary
                st.markdown("**📋 Report Summary:**")
                st.json(report['summary'])
                
            except Exception as e:
                st.error(f"❌ Error generating comprehensive report: {str(e)}")
    
    def _render_step_completion(self):
        """Render step completion section"""
        st.subheader("✅ Step Completion")
        
        step_data = self.session_manager.get_step_data(5)
        is_complete = (
            'selected_models' in step_data and
            len(step_data.get('selected_models', [])) > 0 and
            ('shap_results' in step_data or 'confusion_matrix_results' in step_data)
        )
        
        if is_complete:
            st.success("🎯 Step 5 completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                models_count = len(step_data.get('selected_models', []))
                st.metric("Models Analyzed", models_count)
            
            with col2:
                shap_count = len(step_data.get('shap_results', {}))
                st.metric("SHAP Results", shap_count)
            
            with col3:
                cm_count = len(step_data.get('confusion_matrix_results', {}))
                st.metric("Confusion Matrices", cm_count)
            
            self.session_manager.set_progress(5, 1.0)
            
        else:
            st.info("📝 Complete the analysis to proceed to the next step.")
            self.session_manager.set_progress(5, 0.0)
    
    def validate_step(self) -> bool:
        """Validate if Step 5 is complete"""
        step_data = self.session_manager.get_step_data(5)
        
        return (
            'selected_models' in step_data and
            len(step_data.get('selected_models', [])) > 0 and
            ('shap_results' in step_data or 'confusion_matrix_results' in step_data)
        )
