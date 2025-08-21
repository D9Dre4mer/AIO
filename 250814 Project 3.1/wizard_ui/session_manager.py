"""
Session State Management for Wizard UI

Manages Streamlit session state for:
- Wizard data persistence
- User input storage
- Progress tracking
- Error handling
- Configuration persistence

Created: 2025-01-27
"""

import streamlit as st
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages Streamlit session state for the wizard interface
    
    Responsibilities:
    - Data persistence across wizard steps
    - User input storage and retrieval
    - Progress tracking and status management
    - Error handling and recovery
    - Configuration persistence
    """
    
    def __init__(self):
        """Initialize the session manager"""
        self._initialize_session_state()
        logger.info("Session Manager initialized")
    
    def _initialize_session_state(self) -> None:
        """Initialize all required session state variables"""
        # Wizard state
        if 'wizard_step' not in st.session_state:
            st.session_state.wizard_step = 1
        
        if 'wizard_data' not in st.session_state:
            st.session_state.wizard_data = {}
        
        if 'wizard_progress' not in st.session_state:
            st.session_state.wizard_progress = {}
        
        # Step-specific data - Initialize all step keys
        step_keys = [
            'step1_dataset', 'step2_preprocessing', 'step3_columns',
            'step4_configuration', 'step5_training', 'step6_results',
            'step7_inference'
        ]
        
        for step_key in step_keys:
            if step_key not in st.session_state:
                st.session_state[step_key] = {}
        
        # Global configuration
        if 'wizard_config' not in st.session_state:
            st.session_state.wizard_config = {
                'auto_advance': True,
                'validation_enabled': True,
                'progress_tracking': True,
                'session_persistence': True
            }
        
        # Error tracking
        if 'wizard_errors' not in st.session_state:
            st.session_state.wizard_errors = {}
        
        # User preferences
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'theme': 'light',
                'compact_view': False,
                'auto_save': True
            }
    
    def get_step_data(self, step_number: int) -> Dict[str, Any]:
        """
        Get data for a specific step
        
        Args:
            step_number: Step number to get data for
            
        Returns:
            Dictionary containing step data
        """
        step_key = f"step{step_number}"
        # Check if step key exists in session state
        if step_key in st.session_state:
            return st.session_state[step_key]
        # If not exists, return empty dict and initialize
        st.session_state[step_key] = {}
        return {}
    
    def set_step_data(self, step_number: int, data: Dict[str, Any]) -> None:
        """
        Set data for a specific step
        
        Args:
            step_number: Step number to set data for
            data: Data to store for the step
        """
        step_key = f"step{step_number}"
        # Always set the data, creating the key if it doesn't exist
        st.session_state[step_key] = data
        logger.debug(f"Data set for step {step_number}")
    
    def update_step_data(self, step_number: int, key: str, value: Any) -> None:
        """
        Update a specific key in step data
        
        Args:
            step_number: Step number to update
            key: Key to update
            value: New value
        """
        step_data = self.get_step_data(step_number)
        step_data[key] = value
        self.set_step_data(step_number, step_data)
    
    def set_step_config(self, step_name: str, config: Dict[str, Any]) -> None:
        """
        Set complete configuration for a step
        
        Args:
            step_name: Step name (e.g., 'step1', 'step2')
            config: Complete configuration dictionary
        """
        # Convert step name to step number
        if step_name.startswith('step'):
            try:
                step_number = int(step_name[4:])
                self.set_step_data(step_number, config)
                logger.debug(f"Step {step_number} configuration set")
            except ValueError:
                logger.error(f"Invalid step name: {step_name}")
        else:
            logger.error(f"Invalid step name format: {step_name}")
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a step
        
        Args:
            step_name: Step name (e.g., 'step1', 'step2')
            
        Returns:
            Step configuration dictionary
        """
        if step_name.startswith('step'):
            try:
                step_number = int(step_name[4:])
                return self.get_step_data(step_number)
            except ValueError:
                logger.error(f"Invalid step name: {step_name}")
                return {}
        else:
            logger.error(f"Invalid step name format: {step_name}")
            return {}
    
    def get_wizard_data(self, key: str, default: Any = None) -> Any:
        """
        Get data from wizard_data session state
        
        Args:
            key: Key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            Value associated with key or default
        """
        return st.session_state.wizard_data.get(key, default)
    
    def set_wizard_data(self, key: str, value: Any) -> None:
        """
        Set data in wizard_data session state
        
        Args:
            key: Key to set
            value: Value to store
        """
        st.session_state.wizard_data[key] = value
        logger.debug(f"Wizard data set: {key} = {value}")
    
    def get_progress(self, step_number: int) -> float:
        """
        Get progress for a specific step
        
        Args:
            step_number: Step number to get progress for
            
        Returns:
            Progress value (0.0 to 1.0)
        """
        return st.session_state.wizard_progress.get(f"step_{step_number}", 0.0)
    
    def set_progress(self, step_number: int, progress: float) -> None:
        """
        Set progress for a specific step
        
        Args:
            step_number: Step number to set progress for
            progress: Progress value (0.0 to 1.0)
        """
        st.session_state.wizard_progress[f"step_{step_number}"] = max(0.0, min(1.0, progress))
        logger.debug(f"Progress set for step {step_number}: {progress:.1%}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get wizard configuration value
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        return st.session_state.wizard_config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set wizard configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        st.session_state.wizard_config[key] = value
        logger.debug(f"Configuration set: {key} = {value}")
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """
        Get user preference value
        
        Args:
            key: Preference key
            default: Default value if key doesn't exist
            
        Returns:
            Preference value or default
        """
        return st.session_state.user_preferences.get(key, default)
    
    def set_user_preference(self, key: str, value: Any) -> None:
        """
        Set user preference value
        
        Args:
            key: Preference key
            value: Preference value
        """
        st.session_state.user_preferences[key] = value
        logger.debug(f"User preference set: {key} = {value}")
    
    def store_error(self, step_number: int, error: Exception) -> None:
        """
        Store error information for a step
        
        Args:
            step_number: Step number where error occurred
            error: Exception that occurred
        """
        st.session_state.wizard_errors[f"step_{step_number}"] = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': st.session_state.get('_timestamp', 'unknown')
        }
        logger.error(f"Error stored for step {step_number}: {str(error)}")
    
    def get_error(self, step_number: int) -> Optional[Dict[str, Any]]:
        """
        Get error information for a step
        
        Args:
            step_number: Step number to get error for
            
        Returns:
            Error information dictionary or None
        """
        return st.session_state.wizard_errors.get(f"step_{step_number}")
    
    def clear_error(self, step_number: int) -> None:
        """
        Clear error for a specific step
        
        Args:
            step_number: Step number to clear error for
        """
        error_key = f"step_{step_number}"
        if error_key in st.session_state.wizard_errors:
            del st.session_state.wizard_errors[error_key]
            logger.debug(f"Error cleared for step {step_number}")
    
    def clear_all_errors(self) -> None:
        """Clear all stored errors"""
        st.session_state.wizard_errors = {}
        logger.info("All errors cleared")
    
    def save_session_state(self) -> Dict[str, Any]:
        """
        Save current session state to a dictionary
        
        Returns:
            Dictionary containing session state
        """
        session_snapshot = {
            'wizard_step': st.session_state.wizard_step,
            'wizard_data': dict(st.session_state.wizard_data),
            'wizard_progress': dict(st.session_state.wizard_progress),
            'wizard_config': dict(st.session_state.wizard_config),
            'user_preferences': dict(st.session_state.user_preferences),
            'step_data': {
                'step1_dataset': dict(st.session_state.step1_dataset),
                'step2_preprocessing': dict(st.session_state.step2_preprocessing),
                'step3_columns': dict(st.session_state.step3_columns),
                'step4_configuration': dict(st.session_state.step4_configuration),
                'step5_training': dict(st.session_state.step5_training),
                'step6_results': dict(st.session_state.step6_results),
                'step7_inference': dict(st.session_state.step7_inference)
            }
        }
        logger.info("Session state saved")
        return session_snapshot
    
    def restore_session_state(self, session_data: Dict[str, Any]) -> bool:
        """
        Restore session state from a dictionary
        
        Args:
            session_data: Dictionary containing session state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'wizard_step' in session_data:
                st.session_state.wizard_step = session_data['wizard_step']
            
            if 'wizard_data' in session_data:
                st.session_state.wizard_data = session_data['wizard_data']
            
            if 'wizard_progress' in session_data:
                st.session_state.wizard_progress = session_data['wizard_progress']
            
            if 'wizard_config' in session_data:
                st.session_state.wizard_config = session_data['wizard_config']
            
            if 'user_preferences' in session_data:
                st.session_state.user_preferences = session_data['user_preferences']
            
            if 'step_data' in session_data:
                step_data = session_data['step_data']
                for step_key, step_values in step_data.items():
                    if hasattr(st.session_state, step_key):
                        setattr(st.session_state, step_key, step_values)
            
            logger.info("Session state restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore session state: {str(e)}")
            return False
    
    def reset_session(self) -> None:
        """Reset all session state to initial values"""
        self._initialize_session_state()
        logger.info("Session state reset to initial values")
    
    def export_session_data(self) -> str:
        """
        Export session data as JSON string
        
        Returns:
            JSON string containing session data
        """
        session_data = self.save_session_state()
        return json.dumps(session_data, indent=2, default=str)
    
    def import_session_data(self, json_data: str) -> bool:
        """
        Import session data from JSON string
        
        Args:
            json_data: JSON string containing session data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session_data = json.loads(json_data)
            return self.restore_session_state(session_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to import session data: {str(e)}")
            return False
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current session state
        
        Returns:
            Dictionary containing session summary
        """
        return {
            'current_step': st.session_state.wizard_step,
            'total_data_keys': len(st.session_state.wizard_data),
            'total_progress_keys': len(st.session_state.wizard_progress),
            'total_errors': len(st.session_state.wizard_errors),
            'config_keys': list(st.session_state.wizard_config.keys()),
            'user_preferences': list(st.session_state.user_preferences.keys()),
            'step_data_summary': {
                'step1_dataset': len(st.session_state.step1_dataset),
                'step2_preprocessing': len(st.session_state.step2_preprocessing),
                'step3_columns': len(st.session_state.step3_columns),
                'step4_configuration': len(st.session_state.step4_configuration),
                'step5_training': len(st.session_state.step5_training),
                'step6_results': len(st.session_state.step6_results),
                'step7_inference': len(st.session_state.step7_inference)
            }
        }

    def set_current_step(self, step_number: int) -> None:
        """
        Set current step number
        
        Args:
            step_number: Step number to set as current
        """
        st.session_state.wizard_step = step_number
        logger.debug(f"Current step set to: {step_number}")
    
    def get_current_step(self) -> int:
        """
        Get current step number
        
        Returns:
            Current step number
        """
        return st.session_state.wizard_step
