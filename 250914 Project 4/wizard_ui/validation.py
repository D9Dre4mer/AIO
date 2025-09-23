"""
Step Validation Logic for Wizard UI

Provides validation for each wizard step:
- Input validation
- Data integrity checks
- Dependency validation
- Error reporting and recovery

Created: 2025-01-27
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Enumeration for validation status"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    status: ValidationStatus
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    critical: bool = False


class StepValidator:
    """
    Validates wizard steps and their data
    
    Responsibilities:
    - Input validation for each step
    - Data integrity checks
    - Dependency validation
    - Error reporting and suggestions
    """
    
    def __init__(self):
        """Initialize the step validator"""
        self.validation_rules = self._initialize_validation_rules()
        logger.info("Step Validator initialized")
    
    def _initialize_validation_rules(self) -> Dict[int, Dict[str, Any]]:
        """Initialize validation rules for each step"""
        return {
            1: {  # Dataset Selection & Upload
                'required_fields': ['dataset_source', 'dataset_file'],
                'file_validation': True,
                'size_limits': {'max_size_mb': 100},
                'format_validation': True
            },
            2: {  # Data Preprocessing & Sampling
                'required_fields': ['sample_size', 'preprocessing_options'],
                'numeric_validation': True,
                'range_validation': {'sample_size': (1000, 500000)}
            },
            3: {  # Column Selection & Validation
                'required_fields': ['text_column', 'label_column'],
                'column_validation': True,
                'data_type_validation': True
            },
            4: {  # Model Configuration & Vectorization
                'required_fields': ['model_selection', 'vectorization_method'],
                'model_validation': True,
                'parameter_validation': True
            },
            5: {  # Training Execution & Monitoring
                'required_fields': ['training_started'],
                'training_validation': False  # No validation required
            },
            6: {  # Results Analysis & Export
                'required_fields': ['results_available'],
                'results_validation': False  # No validation required
            },
            7: {  # Text Classification & Inference
                'required_fields': ['input_text'],
                'text_validation': True,
                'model_availability': True
            }
        }
    
    def validate_step(self, step_number: int, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate a specific step
        
        Args:
            step_number: Step number to validate
            step_data: Data for the step
            
        Returns:
            List of validation results
        """
        if step_number not in self.validation_rules:
            return [ValidationResult(
                ValidationStatus.INVALID,
                f"No validation rules for step {step_number}",
                critical=True
            )]
        
        rules = self.validation_rules[step_number]
        results = []
        
        # Check required fields
        if 'required_fields' in rules:
            results.extend(self._validate_required_fields(
                step_data, rules['required_fields']
            ))
        
        # File validation for step 1
        if step_number == 1 and rules.get('file_validation', False):
            results.extend(self._validate_dataset_file(step_data))
        
        # Column validation for step 3
        if step_number == 3 and rules.get('column_validation', False):
            results.extend(self._validate_column_selection(step_data))
        
        # Model validation for step 4
        if step_number == 4 and rules.get('model_validation', False):
            results.extend(self._validate_model_configuration(step_data))
        
        # Text validation for step 7
        if step_number == 7 and rules.get('text_validation', False):
            results.extend(self._validate_text_input(step_data))
        
        return results
    
    def _validate_required_fields(self, step_data: Dict[str, Any], 
                                 required_fields: List[str]) -> List[ValidationResult]:
        """Validate that all required fields are present"""
        results = []
        
        for field in required_fields:
            if field not in step_data or step_data[field] is None:
                results.append(ValidationResult(
                    ValidationStatus.INVALID,
                    f"Required field '{field}' is missing",
                    field=field,
                    critical=True
                ))
            elif isinstance(step_data[field], str) and not step_data[field].strip():
                results.append(ValidationResult(
                    ValidationStatus.INVALID,
                    f"Required field '{field}' cannot be empty",
                    field=field,
                    critical=True
                ))
        
        return results
    
    def _validate_dataset_file(self, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate dataset file upload"""
        results = []
        
        if 'dataset_file' in step_data and step_data['dataset_file'] is not None:
            file = step_data['dataset_file']
            
            # Check file size
            if hasattr(file, 'size'):
                file_size_mb = file.size / (1024 * 1024)
                if file_size_mb > 100:  # 100MB limit
                    results.append(ValidationResult(
                        ValidationStatus.WARNING,
                        f"File size ({file_size_mb:.1f}MB) is large and may take time to process",
                        field='dataset_file',
                        suggestion="Consider using a smaller sample for testing"
                    ))
            
            # Check file format
            if hasattr(file, 'name'):
                file_name = file.name.lower()
                if not any(file_name.endswith(ext) for ext in ['.csv', '.json', '.xlsx', '.xls']):
                    results.append(ValidationResult(
                        ValidationStatus.INVALID,
                        "Unsupported file format. Please use CSV, JSON, or Excel files",
                        field='dataset_file',
                        critical=True
                    ))
        
        return results
    
    def _validate_column_selection(self, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate column selection"""
        results = []
        
        text_col = step_data.get('text_column')
        label_col = step_data.get('label_column')
        
        if text_col and label_col:
            if text_col == label_col:
                results.append(ValidationResult(
                    ValidationStatus.INVALID,
                    "Text column and label column must be different",
                    field='text_column',
                    suggestion="Select different columns for text and labels"
                ))
        
        return results
    
    def _validate_model_configuration(self, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate model configuration"""
        results = []
        
        model_selection = step_data.get('model_selection', [])
        vectorization_method = step_data.get('vectorization_method', [])
        
        if not model_selection:
            results.append(ValidationResult(
                ValidationStatus.INVALID,
                "At least one model must be selected",
                field='model_selection',
                critical=True
            ))
        
        if not vectorization_method:
            results.append(ValidationResult(
                ValidationStatus.INVALID,
                "At least one vectorization method must be selected",
                field='vectorization_method',
                critical=True
            ))
        
        return results
    
    def _validate_text_input(self, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate text input for classification"""
        results = []
        
        input_text = step_data.get('input_text', '')
        
        if not input_text or not input_text.strip():
            results.append(ValidationResult(
                ValidationStatus.INVALID,
                "Input text cannot be empty",
                field='input_text',
                critical=True
            ))
        elif len(input_text.strip()) < 10:
            results.append(ValidationResult(
                ValidationStatus.WARNING,
                "Input text is very short. Longer text may provide better classification",
                field='input_text',
                suggestion="Consider providing more context in your text"
            ))
        
        return results
    
    def validate_step_dependencies(self, step_number: int, 
                                 completed_steps: List[int]) -> List[ValidationResult]:
        """
        Validate that step dependencies are met
        
        Args:
            step_number: Step number to validate dependencies for
            completed_steps: List of completed step numbers
            
        Returns:
            List of validation results
        """
        results = []
        
        if step_number == 1:
            # Step 1 has no dependencies
            return results
        
        dependencies = {
            2: [1],  # Preprocessing depends on dataset
            3: [2],  # Column selection depends on preprocessing
            4: [3],  # Configuration depends on column selection
            5: [4],  # Training depends on configuration
            6: [5],  # Results depend on training
            7: [6]   # Inference depends on results
        }
        
        if step_number in dependencies:
            required_steps = dependencies[step_number]
            for required_step in required_steps:
                if required_step not in completed_steps:
                    results.append(ValidationResult(
                        ValidationStatus.INVALID,
                        f"Step {step_number} requires step {required_step} to be completed first",
                        critical=True
                    ))
        
        return results
    
    def validate_data_integrity(self, step_number: int, 
                               step_data: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate data integrity for a step
        
        Args:
            step_number: Step number to validate
            step_data: Data for the step
            
        Returns:
            List of validation results
        """
        results = []
        
        if step_number == 1:  # Dataset validation
            results.extend(self._validate_dataset_integrity(step_data))
        elif step_number == 2:  # Preprocessing validation
            results.extend(self._validate_preprocessing_integrity(step_data))
        elif step_number == 3:  # Column validation
            results.extend(self._validate_column_integrity(step_data))
        
        return results
    
    def _validate_dataset_integrity(self, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate dataset integrity"""
        results = []
        
        # Check if dataset has been loaded
        if 'dataset_loaded' in step_data and step_data['dataset_loaded']:
            dataset_info = step_data.get('dataset_info', {})
            
            if 'shape' in dataset_info:
                rows, cols = dataset_info['shape']
                if rows == 0:
                    results.append(ValidationResult(
                        ValidationStatus.INVALID,
                        "Dataset contains no rows",
                        field='dataset_info',
                        critical=True
                    ))
                elif cols < 2:
                    results.append(ValidationResult(
                        ValidationStatus.WARNING,
                        "Dataset has very few columns. Ensure it contains text and label columns",
                        field='dataset_info'
                    ))
        
        return results
    
    def _validate_preprocessing_integrity(self, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate preprocessing integrity"""
        results = []
        
        sample_size = step_data.get('sample_size', 0)
        if sample_size > 0:
            if sample_size < 100:
                results.append(ValidationResult(
                    ValidationStatus.WARNING,
                    "Sample size is very small. Results may not be reliable",
                    field='sample_size',
                    suggestion="Consider using a larger sample size"
                ))
        
        return results
    
    def _validate_column_integrity(self, step_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate column integrity"""
        results = []
        
        text_column = step_data.get('text_column')
        label_column = step_data.get('label_column')
        
        if text_column and label_column:
            # Check if columns exist in dataset
            available_columns = step_data.get('available_columns', [])
            if available_columns:
                if text_column not in available_columns:
                    results.append(ValidationResult(
                        ValidationStatus.INVALID,
                        f"Text column '{text_column}' not found in dataset",
                        field='text_column',
                        critical=True
                    ))
                
                if label_column not in available_columns:
                    results.append(ValidationResult(
                        ValidationStatus.INVALID,
                        f"Label column '{label_column}' not found in dataset",
                        field='label_column',
                        critical=True
                    ))
        
        return results
    
    def get_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Get a summary of validation results
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary containing validation summary
        """
        total_checks = len(validation_results)
        valid_checks = sum(1 for r in validation_results if r.status == ValidationStatus.VALID)
        invalid_checks = sum(1 for r in validation_results if r.status == ValidationStatus.INVALID)
        warnings = sum(1 for r in validation_results if r.status == ValidationStatus.WARNING)
        critical_errors = sum(1 for r in validation_results if r.critical)
        
        return {
            'total_checks': total_checks,
            'valid_checks': valid_checks,
            'invalid_checks': invalid_checks,
            'warnings': warnings,
            'critical_errors': critical_errors,
            'is_valid': invalid_checks == 0,
            'can_proceed': critical_errors == 0
        }
    
    def format_validation_message(self, result: ValidationResult) -> str:
        """
        Format a validation result as a user-friendly message
        
        Args:
            result: Validation result to format
            
        Returns:
            Formatted message string
        """
        status_icons = {
            ValidationStatus.VALID: "✅",
            ValidationStatus.INVALID: "❌",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.INFO: "ℹ️"
        }
        
        icon = status_icons.get(result.status, "")
        message = f"{icon} {result.message}"
        
        if result.suggestion:
            message += f"\n💡 Suggestion: {result.suggestion}"
        
        return message
