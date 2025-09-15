"""
Navigation Controls for Wizard UI

Provides navigation functionality:
- Previous/Next step navigation
- Step jumping
- Home/Reset functionality
- Progress-based navigation

Created: 2025-01-27
"""

import streamlit as st
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class NavigationController:
    """
    Controls navigation between wizard steps
    
    Responsibilities:
    - Step navigation (Previous/Next)
    - Step jumping and validation
    - Home/Reset functionality
    - Progress-based navigation
    """
    
    def __init__(self, wizard_manager, session_manager):
        """
        Initialize navigation controller
        
        Args:
            wizard_manager: WizardManager instance
            session_manager: SessionManager instance
        """
        self.wizard_manager = wizard_manager
        self.session_manager = session_manager
        logger.info("Navigation Controller initialized")
    
    def render_navigation_buttons(self, current_step: int, 
                                 total_steps: int,
                                 on_previous: Optional[Callable] = None,
                                 on_next: Optional[Callable] = None,
                                 on_home: Optional[Callable] = None) -> None:
        """
        Render navigation buttons for the current step
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            on_previous: Callback for previous button
            on_next: Callback for next button
            on_home: Callback for home button
        """
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if current_step > 1:
                if st.button("◀ Previous", key=f"prev_{current_step}"):
                    self.previous_step(on_previous)
        
        with col2:
            if current_step < total_steps:
                if st.button("Next ▶", key=f"next_{current_step}"):
                    self.next_step(on_next)
        
        with col3:
            if st.button("🏠 Start Over", key=f"home_{current_step}"):
                self.go_home(on_home)
        
        with col4:
            if current_step < total_steps:
                if st.button("Skip to End", key=f"skip_{current_step}"):
                    self.skip_to_end()
    
    def render_step_navigation(self, current_step: int, 
                              total_steps: int,
                              completed_steps: list[int]) -> None:
        """
        Render step-by-step navigation
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            completed_steps: List of completed step numbers
        """
        st.subheader("📍 Step Navigation")
        
        # Create step indicators
        cols = st.columns(total_steps)
        
        for step_num in range(1, total_steps + 1):
            with cols[step_num - 1]:
                self._render_step_indicator(
                    step_num, current_step, step_num in completed_steps
                )
    
    def _render_step_indicator(self, step_number: int, 
                              current_step: int, 
                              is_completed: bool) -> None:
        """
        Render individual step indicator
        
        Args:
            step_number: Step number to render
            current_step: Currently active step
            is_completed: Whether step is completed
        """
        if step_number == current_step:
            # Current step
            st.markdown(f"**{step_number}**")
            st.markdown("📍")
        elif is_completed:
            # Completed step
            st.markdown(f"~~{step_number}~~")
            st.markdown("✅")
        else:
            # Future step
            st.markdown(f"{step_number}")
            st.markdown("⏳")
    
    def previous_step(self, callback: Optional[Callable] = None) -> bool:
        """
        Navigate to previous step
        
        Args:
            callback: Optional callback function to execute
            
        Returns:
            True if navigation successful, False otherwise
        """
        if self.wizard_manager.previous_step():
            logger.info("Navigated to previous step")
            
            # Execute callback if provided
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in previous step callback: {str(e)}")
            
            # Rerun to update UI
            st.rerun()
            return True
        else:
            logger.warning("Cannot navigate to previous step")
            return False
    
    def next_step(self, callback: Optional[Callable] = None) -> bool:
        """
        Navigate to next step
        
        Args:
            callback: Optional callback function to execute
            
        Returns:
            True if navigation successful, False otherwise
        """
        if self.wizard_manager.next_step():
            logger.info("Navigated to next step")
            
            # Execute callback if provided
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in next step callback: {str(e)}")
            
            # Rerun to update UI
            st.rerun()
            return True
        else:
            logger.warning("Cannot navigate to next step")
            return False
    
    def go_to_step(self, step_number: int, 
                   callback: Optional[Callable] = None) -> bool:
        """
        Navigate to specific step
        
        Args:
            step_number: Target step number
            callback: Optional callback function to execute
            
        Returns:
            True if navigation successful, False otherwise
        """
        if self.wizard_manager.go_to_step(step_number):
            logger.info(f"Navigated to step {step_number}")
            
            # Execute callback if provided
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in go to step callback: {str(e)}")
            
            # Rerun to update UI
            st.rerun()
            return True
        else:
            logger.warning(f"Cannot navigate to step {step_number}")
            return False
    
    def go_home(self, callback: Optional[Callable] = None) -> bool:
        """
        Navigate to home (step 1)
        
        Args:
            callback: Optional callback function to execute
            
        Returns:
            True if navigation successful, False otherwise
        """
        return self.go_to_step(1, callback)
    
    def skip_to_end(self) -> bool:
        """
        Skip to the last step
        
        Returns:
            True if navigation successful, False otherwise
        """
        return self.go_to_step(self.wizard_manager.total_steps)
    
    def render_quick_navigation(self, current_step: int, 
                               total_steps: int) -> None:
        """
        Render quick navigation options
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
        """
        st.sidebar.subheader("🚀 Quick Navigation")
        
        # Step selection dropdown
        step_options = [f"Step {i}" for i in range(1, total_steps + 1)]
        selected_step = st.sidebar.selectbox(
            "Go to Step:",
            step_options,
            index=current_step - 1,
            key="quick_nav_dropdown"
        )
        
        # Extract step number from selection
        if selected_step:
            target_step = int(selected_step.split()[-1])
            if target_step != current_step:
                if st.sidebar.button("Go", key="quick_nav_go"):
                    self.go_to_step(target_step)
        
        # Quick action buttons
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🏠 Home", key="sidebar_home"):
            self.go_home()
        
        if st.sidebar.button("🔄 Reset", key="sidebar_reset"):
            self.reset_wizard()
        
        if st.sidebar.button("📊 Progress", key="sidebar_progress"):
            self.show_progress_summary()
    
    def render_breadcrumb(self, current_step: int, 
                          total_steps: int) -> None:
        """
        Render breadcrumb navigation
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
        """
        st.markdown("---")
        
        # Create breadcrumb
        breadcrumb_parts = []
        
        for step_num in range(1, total_steps + 1):
            if step_num == current_step:
                breadcrumb_parts.append(f"**Step {step_num}**")
            elif step_num < current_step:
                breadcrumb_parts.append(f"[Step {step_num}](#)")
            else:
                breadcrumb_parts.append(f"Step {step_num}")
        
        breadcrumb = " > ".join(breadcrumb_parts)
        st.markdown(f"**Navigation:** {breadcrumb}")
    
    def render_progress_bar(self, current_step: int, 
                           total_steps: int) -> None:
        """
        Render progress bar
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
        """
        progress = current_step / total_steps
        
        st.progress(progress)
        st.caption(f"Step {current_step} of {total_steps} ({progress:.1%} complete)")
    
    def reset_wizard(self) -> bool:
        """
        Reset wizard to initial state
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            self.wizard_manager.reset_wizard()
            self.session_manager.reset_session()
            logger.info("Wizard reset to initial state")
            
            # Rerun to update UI
            st.rerun()
            return True
            
        except Exception as e:
            logger.error(f"Error resetting wizard: {str(e)}")
            return False
    
    def show_progress_summary(self) -> None:
        """Show progress summary in sidebar"""
        summary = self.wizard_manager.get_step_summary()
        
        st.sidebar.subheader("📊 Progress Summary")
        st.sidebar.metric("Current Step", summary['current_step'])
        st.sidebar.metric("Total Steps", summary['total_steps'])
        st.sidebar.metric("Overall Progress", f"{summary['overall_progress']:.1%}")
        
        # Step status breakdown
        st.sidebar.markdown("**Step Status:**")
        for step_num, step_info in summary['steps'].items():
            status_icon = "✅" if step_info['completed'] else "⏳"
            st.sidebar.markdown(f"{status_icon} Step {step_num}: {step_info['title']}")
    
    def can_navigate_to_step(self, step_number: int) -> bool:
        """
        Check if navigation to step is allowed
        
        Args:
            step_number: Target step number
            
        Returns:
            True if navigation allowed, False otherwise
        """
        return self.wizard_manager.can_advance_to_step(step_number)
    
    def get_navigation_summary(self) -> Dict[str, Any]:
        """
        Get navigation summary information
        
        Returns:
            Dictionary containing navigation summary
        """
        current_step = self.wizard_manager.get_current_step()
        total_steps = self.wizard_manager.total_steps
        
        return {
            'current_step': current_step,
            'total_steps': total_steps,
            'can_go_previous': current_step > 1,
            'can_go_next': current_step < total_steps,
            'progress_percentage': (current_step / total_steps) * 100,
            'steps_remaining': total_steps - current_step,
            'steps_completed': current_step - 1
        }
    
    def render_navigation_help(self) -> None:
        """Render navigation help information"""
        with st.expander("ℹ️ Navigation Help"):
            st.markdown("""
            **Navigation Controls:**
            
            - **◀ Previous**: Go back to the previous step
            - **Next ▶**: Advance to the next step (if validation passes)
            - **🏠 Start Over**: Return to step 1 and reset all data
            - **Skip to End**: Jump to the final step
            
            **Quick Navigation:**
            
            - Use the sidebar dropdown to jump to any step
            - Step indicators show your current progress
            - Completed steps are marked with ✅
            - Current step is highlighted with 📍
            
            **Tips:**
            
            - You can always go back to previous steps
            - Some steps require validation before proceeding
            - Use the progress bar to track your completion
            - Reset the wizard anytime to start fresh
            """)
