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
from datetime import datetime

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
        
        # FIXED: Auto-restore session state from backup
        self.auto_restore_session()
        
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
        Set data for a specific step with comprehensive error handling and rerun prevention
        
        Args:
            step_number: Step number to set data for
            data: Data to store for the step
        """
        try:
            step_key = f"step{step_number}"
            
            # Prevent rerun loop by checking if data actually changed
            current_data = st.session_state.get(step_key, {})
            if current_data == data:
                logger.debug(f"Step {step_number} data unchanged, skipping update")
                return
            
            # Always set the data, creating the key if it doesn't exist
            st.session_state[step_key] = data
            logger.debug(f"Data set for step {step_number}")
            print(f"SUCCESS: Step {step_number} data saved to session state")
        except Exception as e:
            logger.error(f"Failed to set step {step_number} data: {str(e)}")
            print(f"ERROR: Failed to set step {step_number} data: {str(e)}")
            import traceback
            print(f"set_step_data traceback: {traceback.format_exc()}")
            # Continue execution even if session state fails
    
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
        
        # DISABLED: Auto-save session state to prevent crashes
        # self.auto_save_session()
        print("DEBUG: Auto-save disabled to prevent crashes")
    
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
        try:
            if 'wizard_config' not in st.session_state:
                return default
            return st.session_state.wizard_config.get(key, default)
        except (AttributeError, KeyError):
            # Fallback if session state is not properly initialized
            return default
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set wizard configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        try:
            if 'wizard_config' not in st.session_state:
                st.session_state.wizard_config = {}
            st.session_state.wizard_config[key] = value
            logger.debug(f"Configuration set: {key} = {value}")
        except (AttributeError, KeyError) as e:
            logger.warning(f"Failed to set config {key}: {e}")
            # Silently fail to avoid breaking the app
    
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
        Save current session state to a dictionary - OPTIMIZED VERSION
        
        Only saves essential data, not large objects like dataframes
        """
        def safe_dict_conversion(obj):
            """Safely convert session state objects to dict, skipping non-serializable items"""
            try:
                if hasattr(obj, 'keys'):
                    result = {}
                    for key, value in obj.items():
                        try:
                            # Skip large objects that shouldn't be in session state
                            if key in ['dataframe', 'df', 'training_results', 'comprehensive_results', 'model']:
                                logger.debug(f"Skipping large object: {key}")
                                continue
                            
                            # Try to serialize the value
                            json.dumps(value, default=str)
                            result[key] = value
                        except (TypeError, ValueError):
                            # Skip non-serializable values (like model objects)
                            logger.debug(f"Skipping non-serializable key: {key}")
                            continue
                    return result
                else:
                    return obj
            except Exception as e:
                logger.warning(f"Failed to convert object to dict: {e}")
                return {}
        
        # Only save essential session state data
        session_snapshot = {
            'wizard_step': st.session_state.wizard_step,
            'wizard_data': safe_dict_conversion(st.session_state.wizard_data),
            'wizard_progress': safe_dict_conversion(st.session_state.wizard_progress),
            'wizard_config': safe_dict_conversion(st.session_state.wizard_config),
            'user_preferences': safe_dict_conversion(st.session_state.user_preferences),
            'step_data': {
                'step1_dataset': safe_dict_conversion(st.session_state.step1_dataset),
                'step2_preprocessing': safe_dict_conversion(st.session_state.step2_preprocessing),
                'step3_columns': safe_dict_conversion(st.session_state.step3_columns),
                'step4_configuration': safe_dict_conversion(st.session_state.step4_configuration),
                'step5_training': safe_dict_conversion(st.session_state.step5_training),
                'step6_results': safe_dict_conversion(st.session_state.step6_results),
                'step7_inference': safe_dict_conversion(st.session_state.step7_inference)
            }
        }
        logger.info("Session state saved (optimized - no large objects)")
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
    
    def clear_large_objects(self) -> None:
        """Clear large objects from session state to reduce memory usage with comprehensive error handling"""
        try:
            large_object_keys = ['dataframe', 'df', 'training_results', 'comprehensive_results', 'model']
            
            for step_key in ['step1_dataset', 'step2_preprocessing', 'step3_columns', 
                            'step4_configuration', 'step5_training', 'step6_results', 'step7_inference']:
                try:
                    if step_key in st.session_state:
                        step_data = st.session_state[step_key]
                        if isinstance(step_data, dict):
                            for key in large_object_keys:
                                if key in step_data:
                                    del step_data[key]
                                    logger.debug(f"Cleared large object: {step_key}.{key}")
                except Exception as step_error:
                    logger.warning(f"Failed to clear large objects from {step_key}: {str(step_error)}")
                    print(f"WARNING: Failed to clear large objects from {step_key}: {str(step_error)}")
                    # Continue with other steps
            
            logger.info("Large objects cleared from session state")
            print("SUCCESS: Large objects cleared from session state")
            
        except Exception as e:
            logger.error(f"Failed to clear large objects: {str(e)}")
            print(f"ERROR: Failed to clear large objects: {str(e)}")
            import traceback
            print(f"clear_large_objects traceback: {traceback.format_exc()}")
            # Continue execution even if cleanup fails
    
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
    
    def save_to_file(self, file_path: str = None) -> bool:
        """
        Save session state to file for persistence
        
        Args:
            file_path: Path to save session file (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path is None:
                # Use default path in wizard_ui directory
                import os
                default_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(default_dir, "session_backup.json")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save session state
            session_data = self.save_session_state()
            
            # Add timestamp
            session_data['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0',
                'source': 'SessionManager.save_to_file'
            }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Session state saved to file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session state to file: {str(e)}")
            return False
    
    def load_from_file(self, file_path: str = None) -> bool:
        """
        Load session state from file
        
        Args:
            file_path: Path to load session file from (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path is None:
                # Use default path in wizard_ui directory
                import os
                default_dir = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(default_dir, "session_backup.json")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Session backup file not found: {file_path}")
                return False
            
            # Read from file
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Remove metadata
            if '_metadata' in session_data:
                metadata = session_data.pop('_metadata')
                logger.info(f"Loading session from backup saved at: {metadata.get('saved_at', 'Unknown')}")
            
            # Restore session state
            success = self.restore_session_state(session_data)
            
            if success:
                logger.info(f"Session state loaded from file: {file_path}")
            else:
                logger.error(f"Failed to restore session state from file: {file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load session state from file: {str(e)}")
            return False
    
    def auto_save_enabled(self) -> bool:
        """Check if auto-save is enabled"""
        try:
            if 'wizard_config' not in st.session_state:
                return True  # Default to enabled if not initialized
            return st.session_state.wizard_config.get('auto_save', True)
        except (AttributeError, KeyError):
            # Fallback if session state is not properly initialized
            return True
    
    def enable_auto_save(self) -> None:
        """Enable auto-save functionality"""
        try:
            if 'wizard_config' not in st.session_state:
                st.session_state.wizard_config = {}
            st.session_state.wizard_config['auto_save'] = True
            logger.info("Auto-save enabled")
        except (AttributeError, KeyError) as e:
            logger.warning(f"Failed to enable auto-save: {e}")
    
    def disable_auto_save(self) -> None:
        """Disable auto-save functionality"""
        try:
            if 'wizard_config' not in st.session_state:
                st.session_state.wizard_config = {}
            st.session_state.wizard_config['auto_save'] = False
            logger.info("Auto-save disabled")
        except (AttributeError, KeyError) as e:
            logger.warning(f"Failed to disable auto-save: {e}")
    
    def auto_save_session(self) -> None:
        """Auto-save session state if enabled"""
        if self.auto_save_enabled():
            try:
                self.save_to_file()
                logger.debug("Auto-save completed")
            except Exception as e:
                logger.warning(f"Auto-save failed: {str(e)}")
    
    def auto_restore_session(self) -> None:
        """Auto-restore session state on startup"""
        try:
            if self.load_from_file():
                logger.info("Session state auto-restored from backup")
            else:
                logger.info("No session backup found, using default state")
        except Exception as e:
            logger.warning(f"Auto-restore failed: {str(e)}")
    
    def get_session_backup_info(self) -> Dict[str, Any]:
        """
        Get information about session backup file
        
        Returns:
            Dictionary with backup file information
        """
        try:
            import os
            default_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(default_dir, "session_backup.json")
            
            if not os.path.exists(file_path):
                return {
                    'exists': False,
                    'file_path': file_path,
                    'size': 0,
                    'modified': None
                }
            
            stat = os.stat(file_path)
            return {
                'exists': True,
                'file_path': file_path,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'size_human': self._format_file_size(stat.st_size)
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup info: {str(e)}")
            return {
                'exists': False,
                'error': str(e)
            }
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
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
            'config_keys': list(st.session_state.wizard_config.keys()) if 'wizard_config' in st.session_state else [],
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
