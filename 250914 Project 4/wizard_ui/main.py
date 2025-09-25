"""
Main Wizard Application Entry Point

This file serves as the main entry point for the wizard UI application,
integrating all steps and providing a unified interface.

Created: 2025-01-27
"""

import streamlit as st
import logging
from typing import Dict, Any

from .core import WizardManager
from .session_manager import SessionManager
from .navigation import NavigationController
from .steps.step1_dataset import DatasetSelectionStep
from .steps.step3_optuna_stacking import OptunaStackingStep
from .steps.step5_shap_visualization import SHAPVisualizationStep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WizardApp:
    """Main wizard application class"""
    
    def __init__(self):
        """Initialize the wizard application"""
        self.wizard_manager = WizardManager(total_steps=7)
        self.session_manager = SessionManager()
        self.navigation = NavigationController()
        
        # Initialize step components
        self.steps = {
            1: DatasetSelectionStep(),
            3: OptunaStackingStep(),
            5: SHAPVisualizationStep()
        }
        
        logger.info("Wizard application initialized")
    
    def render(self) -> None:
        """Render the complete wizard application"""
        st.set_page_config(
            page_title="Enhanced ML Pipeline Wizard",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Render header
        self._render_header()
        
        # Render sidebar navigation
        self._render_sidebar()
        
        # Render main content
        self._render_main_content()
        
        # Render footer
        self._render_footer()
    
    def _render_header(self) -> None:
        """Render application header"""
        st.title("🚀 Enhanced Machine Learning Pipeline Wizard")
        
        st.markdown("""
        **Welcome to the Enhanced ML Pipeline Wizard!**
        
        This wizard will guide you through the complete machine learning workflow,
        from data preprocessing to model interpretation with SHAP analysis.
        """)
        
        # Display current progress
        current_step = self.wizard_manager.get_current_step()
        total_steps = self.wizard_manager.total_steps
        
        progress = current_step / total_steps
        st.progress(progress)
        
        st.markdown(f"**Step {current_step} of {total_steps}**")
    
    def _render_sidebar(self) -> None:
        """Render sidebar navigation"""
        with st.sidebar:
            st.header("📋 Navigation")
            
            # Step navigation
            current_step = self.wizard_manager.get_current_step()
            
            for step_num in range(1, self.wizard_manager.total_steps + 1):
                step_info = self.wizard_manager.step_info[step_num]
                
                # Determine step status
                if step_num < current_step:
                    status_icon = "✅"
                    status_color = "green"
                elif step_num == current_step:
                    status_icon = "🔄"
                    status_color = "blue"
                else:
                    status_icon = "⏳"
                    status_color = "gray"
                
                # Create step button
                if st.button(
                    f"{status_icon} Step {step_num}: {step_info.title}",
                    key=f"nav_step_{step_num}",
                    disabled=step_num > current_step + 1
                ):
                    if step_num <= current_step + 1:
                        self.wizard_manager.set_current_step(step_num)
                        st.rerun()
            
            st.markdown("---")
            
            # Quick actions
            st.header("⚡ Quick Actions")
            
            if st.button("🏠 Reset Wizard"):
                self._reset_wizard()
            
            if st.button("💾 Save Session"):
                self._save_session()
            
            if st.button("📥 Load Session"):
                self._load_session()
            
            st.markdown("---")
            
            # Session info
            st.header("📊 Session Info")
            
            session_data = self.session_manager.get_all_step_data()
            completed_steps = sum(1 for step_data in session_data.values() 
                                if step_data.get('completed', False))
            
            st.metric("Completed Steps", f"{completed_steps}/{self.wizard_manager.total_steps}")
            
            # Display step summaries
            for step_num, step_data in session_data.items():
                if step_data.get('completed', False):
                    st.success(f"✅ Step {step_num} completed")
    
    def _render_main_content(self) -> None:
        """Render main content area"""
        current_step = self.wizard_manager.get_current_step()
        
        # Render current step
        if current_step in self.steps:
            try:
                self.steps[current_step].render()
            except Exception as e:
                st.error(f"❌ Error rendering step {current_step}: {str(e)}")
                logger.error(f"Error rendering step {current_step}: {e}")
        else:
            st.info(f"📝 Step {current_step} is not yet implemented")
            st.markdown("""
            **Coming Soon:**
            - Step 2: Data Preprocessing & Sampling
            - Step 4: Model Configuration & Vectorization  
            - Step 6: Results Analysis & Export
            - Step 7: Text Classification & Inference
            """)
        
        # Render navigation controls
        self._render_navigation_controls()
    
    def _render_navigation_controls(self) -> None:
        """Render navigation controls"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=self.wizard_manager.current_step <= 1):
                if self.wizard_manager.previous_step():
                    st.rerun()
        
        with col2:
            current_step = self.wizard_manager.get_current_step()
            step_info = self.wizard_manager.step_info[current_step]
            
            # Check if current step is complete
            is_complete = self._is_step_complete(current_step)
            
            if is_complete:
                if current_step < self.wizard_manager.total_steps:
                    if st.button("➡️ Next Step", type="primary"):
                        if self.wizard_manager.next_step():
                            st.rerun()
                else:
                    st.success("🎉 All steps completed!")
            else:
                st.info(f"📝 Complete Step {current_step} to continue")
        
        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()
    
    def _render_footer(self) -> None:
        """Render application footer"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📚 Documentation**")
            st.markdown("- [User Guide](link)")
            st.markdown("- [API Reference](link)")
        
        with col2:
            st.markdown("**🔧 Configuration**")
            st.markdown("- [Settings](link)")
            st.markdown("- [Preferences](link)")
        
        with col3:
            st.markdown("**💬 Support**")
            st.markdown("- [Help Center](link)")
            st.markdown("- [Contact Us](link)")
        
        st.markdown("---")
        st.markdown("**Enhanced ML Pipeline Wizard** - Built with Streamlit 🚀")
    
    def _is_step_complete(self, step_number: int) -> bool:
        """Check if a step is complete"""
        if step_number in self.steps:
            try:
                return self.steps[step_number].validate_step()
            except Exception as e:
                logger.error(f"Error validating step {step_number}: {e}")
                return False
        return False
    
    def _reset_wizard(self) -> None:
        """Reset the wizard to initial state"""
        self.wizard_manager.set_current_step(1)
        self.session_manager.clear_all_data()
        st.success("🔄 Wizard reset successfully!")
        st.rerun()
    
    def _save_session(self) -> None:
        """Save current session"""
        try:
            self.session_manager.save_session()
            st.success("💾 Session saved successfully!")
        except Exception as e:
            st.error(f"❌ Error saving session: {str(e)}")
    
    def _load_session(self) -> None:
        """Load saved session"""
        try:
            self.session_manager.load_session()
            st.success("📥 Session loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading session: {str(e)}")


def main():
    """Main application entry point"""
    app = WizardApp()
    app.render()


if __name__ == "__main__":
    main()
