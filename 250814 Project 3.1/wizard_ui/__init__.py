"""
Wizard UI Package for Topic Modeling Project

A comprehensive multi-step wizard interface built with Streamlit,
following Progressive Disclosure principles for optimal user experience.

This package provides:
- Step-by-step workflow management
- Session state management
- Progress tracking and validation
- Navigation controls
- Responsive design components

Created: 2025-01-27
Version: 1.0.0
"""

from .core import WizardManager
from .session_manager import SessionManager
from .validation import StepValidator
from .navigation import NavigationController

__version__ = "1.0.0"
__all__ = [
    "WizardManager",
    "SessionManager", 
    "StepValidator",
    "NavigationController"
]

# Package-level configuration
WIZARD_CONFIG = {
    "total_steps": 7,
    "auto_advance": True,
    "validation_enabled": True,
    "progress_tracking": True,
    "session_persistence": True
}
