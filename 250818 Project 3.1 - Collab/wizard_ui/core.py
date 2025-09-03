"""
Core Wizard Management System

Main controller for the wizard interface that manages:
- Step transitions and navigation
- Overall workflow state
- Integration between components
- Error handling and recovery

Created: 2025-01-27
"""

import streamlit as st
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Enumeration for step completion status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ERROR = "error"

@dataclass
class StepInfo:
    """Information about a wizard step"""
    step_number: int
    title: str
    description: str
    status: StepStatus
    validation_required: bool
    dependencies: list[int]
    estimated_time: str
    progress: float = 0.0


class WizardManager:
    """
    Main wizard controller that manages the entire workflow
    
    Responsibilities:
    - Step management and transitions
    - State coordination between components
    - Error handling and recovery
    - Progress tracking
    """
    
    def __init__(self, total_steps: int = 7):
        """
        Initialize the wizard manager
        
        Args:
            total_steps: Total number of steps in the wizard
        """
        self.total_steps = total_steps
        self.current_step = 1
        self.step_history = []
        self.step_info = self._initialize_step_info()
        
        # Initialize session state if not exists
        if 'wizard_step' not in st.session_state:
            st.session_state.wizard_step = 1
        if 'wizard_data' not in st.session_state:
            st.session_state.wizard_data = {}
        if 'wizard_progress' not in st.session_state:
            st.session_state.wizard_progress = {}
            
        logger.info(f"Wizard Manager initialized with {total_steps} steps")
    
    def _initialize_step_info(self) -> Dict[int, StepInfo]:
        """Initialize step information based on wireframe design"""
        return {
            1: StepInfo(
                step_number=1,
                title="Dataset Selection & Upload",
                description="Choose dataset source and upload files",
                status=StepStatus.PENDING,
                validation_required=True,
                dependencies=[],
                estimated_time="2-3 minutes"
            ),
            2: StepInfo(
                step_number=2,
                title="Data Preprocessing & Sampling",
                description="Configure sampling and preprocessing options",
                status=StepStatus.PENDING,
                validation_required=True,
                dependencies=[1],
                estimated_time="3-5 minutes"
            ),
            3: StepInfo(
                step_number=3,
                title="Column Selection & Validation",
                description="Select text and label columns",
                status=StepStatus.PENDING,
                validation_required=True,
                dependencies=[2],
                estimated_time="2-3 minutes"
            ),
            4: StepInfo(
                step_number=4,
                title="Model Configuration & Vectorization",
                description="Configure models and vectorization methods",
                status=StepStatus.PENDING,
                validation_required=True,
                dependencies=[3],
                estimated_time="3-4 minutes"
            ),
            5: StepInfo(
                step_number=5,
                title="Training Execution & Monitoring",
                description="Execute training and monitor progress",
                status=StepStatus.PENDING,
                validation_required=False,
                dependencies=[4],
                estimated_time="10-30 minutes"
            ),
            6: StepInfo(
                step_number=6,
                title="Results Analysis & Export",
                description="Analyze results and export findings",
                status=StepStatus.PENDING,
                validation_required=False,
                dependencies=[5],
                estimated_time="2-3 minutes"
            ),
            7: StepInfo(
                step_number=7,
                title="Text Classification & Inference",
                description="Classify new text using trained models",
                status=StepStatus.PENDING,
                validation_required=False,
                dependencies=[6],
                estimated_time="1-2 minutes"
            )
        }
    
    def get_current_step(self) -> int:
        """Get the current step number"""
        return st.session_state.wizard_step
    
    def set_current_step(self, step_number: int) -> bool:
        """
        Set the current step number
        
        Args:
            step_number: Step number to set
            
        Returns:
            True if successful, False otherwise
        """
        if 1 <= step_number <= self.total_steps:
            # Update step history
            if self.current_step != step_number:
                self.step_history.append(self.current_step)
            
            self.current_step = step_number
            st.session_state.wizard_step = step_number
            
            # Update step status
            if step_number > 1:
                self.step_info[step_number - 1].status = StepStatus.COMPLETED
            
            self.step_info[step_number].status = StepStatus.IN_PROGRESS
            
            logger.info(f"Current step set to {step_number}")
            return True
        else:
            logger.warning(f"Invalid step number: {step_number}")
            return False
    
    def next_step(self) -> bool:
        """
        Advance to the next step
        
        Returns:
            True if successful, False if at last step
        """
        if self.current_step < self.total_steps:
            return self.set_current_step(self.current_step + 1)
        return False
    
    def previous_step(self) -> bool:
        """
        Go back to the previous step
        
        Returns:
            True if successful, False if at first step
        """
        if self.current_step > 1:
            return self.set_current_step(self.current_step - 1)
        return False
    
    def go_to_step(self, step_number: int) -> bool:
        """
        Navigate to a specific step
        
        Args:
            step_number: Target step number
            
        Returns:
            True if successful, False otherwise
        """
        return self.set_current_step(step_number)
    
    def get_step_info(self, step_number: int) -> Optional[StepInfo]:
        """
        Get information about a specific step
        
        Args:
            step_number: Step number to get info for
            
        Returns:
            StepInfo object or None if invalid
        """
        return self.step_info.get(step_number)
    
    def get_current_step_info(self) -> Optional[StepInfo]:
        """Get information about the current step"""
        return self.get_step_info(self.current_step)
    
    def update_step_progress(self, step_number: int, progress: float) -> None:
        """
        Update progress for a specific step
        
        Args:
            step_number: Step number to update
            progress: Progress percentage (0.0 to 1.0)
        """
        if step_number in self.step_info:
            self.step_info[step_number].progress = max(0.0, min(1.0, progress))
            st.session_state.wizard_progress[f"step_{step_number}"] = progress
            logger.debug(f"Step {step_number} progress updated to {progress:.1%}")
    
    def get_overall_progress(self) -> float:
        """Calculate overall progress across all steps"""
        total_progress = sum(step.progress for step in self.step_info.values())
        return total_progress / self.total_steps
    
    def can_advance_to_step(self, step_number: int) -> bool:
        """
        Check if user can advance to a specific step
        
        Args:
            step_number: Target step number
            
        Returns:
            True if step is accessible, False otherwise
        """
        if step_number < 1 or step_number > self.total_steps:
            return False
        
        # Check dependencies
        step_info = self.step_info[step_number]
        for dep_step in step_info.dependencies:
            if self.step_info[dep_step].status != StepStatus.COMPLETED:
                return False
        
        return True
    
    def mark_step_completed(self, step_number: int) -> None:
        """
        Mark a step as completed
        
        Args:
            step_number: Step number to mark as completed
        """
        if step_number in self.step_info:
            self.step_info[step_number].status = StepStatus.COMPLETED
            self.step_info[step_number].progress = 1.0
            logger.info(f"Step {step_number} marked as completed")
    
    def mark_step_blocked(self, step_number: int, reason: str = "") -> None:
        """
        Mark a step as blocked
        
        Args:
            step_number: Step number to mark as blocked
            reason: Reason for blocking
        """
        if step_number in self.step_info:
            self.step_info[step_number].status = StepStatus.BLOCKED
            logger.warning(f"Step {step_number} blocked: {reason}")
    
    def reset_wizard(self) -> None:
        """Reset the wizard to initial state"""
        self.current_step = 1
        self.step_history = []
        
        # Reset all step statuses
        for step_info in self.step_info.values():
            step_info.status = StepStatus.PENDING
            step_info.progress = 0.0
        
        # Reset session state
        st.session_state.wizard_step = 1
        st.session_state.wizard_data = {}
        st.session_state.wizard_progress = {}
        
        logger.info("Wizard reset to initial state")
    
    def get_step_summary(self) -> Dict[str, Any]:
        """Get a summary of all steps and their status"""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "overall_progress": self.get_overall_progress(),
            "steps": {
                step_num: {
                    "title": info.title,
                    "status": info.status.value,
                    "progress": info.progress,
                    "completed": info.status == StepStatus.COMPLETED
                }
                for step_num, info in self.step_info.items()
            }
        }
    
    def validate_current_step(self) -> bool:
        """
        Validate the current step
        
        Returns:
            True if step is valid, False otherwise
        """
        current_info = self.get_current_step_info()
        if not current_info:
            return False
        
        # Check if dependencies are met
        for dep_step in current_info.dependencies:
            if self.step_info[dep_step].status != StepStatus.COMPLETED:
                logger.warning(f"Step {self.current_step} blocked: dependency {dep_step} not completed")
                return False
        
        return True
    
    def handle_error(self, error: Exception, step_number: Optional[int] = None) -> None:
        """
        Handle errors that occur during wizard execution
        
        Args:
            error: The error that occurred
            step_number: Step number where error occurred (defaults to current step)
        """
        if step_number is None:
            step_number = self.current_step
        
        # Mark step as having error
        if step_number in self.step_info:
            self.step_info[step_number].status = StepStatus.ERROR
        
        # Log error
        logger.error(f"Error in step {step_number}: {str(error)}")
        
        # Store error in session state for UI display
        st.session_state.wizard_data[f"step_{step_number}_error"] = str(error)
    
    def is_wizard_complete(self) -> bool:
        """Check if all steps are completed"""
        return all(
            step.status == StepStatus.COMPLETED 
            for step in self.step_info.values()
        )
