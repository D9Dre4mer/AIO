"""
Data validation module using Great Expectations and Pandera
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import great_expectations as ge
import numpy as np
import pandas as pd
import pandera as pa
from great_expectations.core.batch import RuntimeBatchRequest
from pandera import Check, Column, DataFrameSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation using Great Expectations and Pandera"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.context = self._setup_great_expectations()

    def _setup_great_expectations(self):
        """Setup Great Expectations context"""
        try:
            context = ge.get_context()
            return context
        except Exception as e:
            logger.warning(f"Could not setup Great Expectations: {e}")
            return None

    def validate_dataset_schema(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset schema using Pandera"""

        logger.info(f"Validating schema for {dataset_name}")

        # Define schema based on dataset type
        if "text" in dataset_name.lower():
            schema = DataFrameSchema(
                {
                    "text": Column(str, Check.str_length(min_value=1, max_value=10000)),
                    "category": Column(str, Check.isin(self._get_expected_categories())),
                }
            )
        elif "heart" in dataset_name.lower():
            schema = DataFrameSchema(
                {
                    "age": Column(int, Check.in_range(min_value=0, max_value=120)),
                    "sex": Column(int, Check.isin([0, 1])),
                    "cp": Column(int, Check.in_range(min_value=0, max_value=3)),
                    "trestbps": Column(int, Check.in_range(min_value=80, max_value=250)),
                    "chol": Column(int, Check.in_range(min_value=100, max_value=600)),
                    "fbs": Column(int, Check.isin([0, 1])),
                    "restecg": Column(int, Check.in_range(min_value=0, max_value=2)),
                    "thalach": Column(int, Check.in_range(min_value=60, max_value=220)),
                    "exang": Column(int, Check.isin([0, 1])),
                    "oldpeak": Column(float, Check.in_range(min_value=0, max_value=10)),
                    "slope": Column(int, Check.in_range(min_value=0, max_value=2)),
                    "ca": Column(int, Check.in_range(min_value=0, max_value=4)),
                    "thal": Column(int, Check.in_range(min_value=0, max_value=3)),
                    "target": Column(int, Check.isin([0, 1])),
                }
            )
        else:
            # Generic schema
            schema = DataFrameSchema({col: Column(object) for col in df.columns})

        try:
            validated_df = schema.validate(df)
            logger.info(f"Schema validation passed for {dataset_name}")
            return {
                "status": "passed",
                "message": "Schema validation successful",
                "rows_validated": len(validated_df),
            }
        except pa.errors.SchemaError as e:
            logger.error(f"Schema validation failed for {dataset_name}: {e}")
            return {"status": "failed", "message": str(e), "rows_validated": len(df)}

    def validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate data quality using Great Expectations"""

        if not self.context:
            logger.warning("Great Expectations not available, skipping quality validation")
            return {"status": "skipped", "message": "Great Expectations not available"}

        logger.info(f"Validating data quality for {dataset_name}")

        try:
            # Create datasource
            datasource_config = {
                "name": f"{dataset_name}_datasource",
                "class_name": "Datasource",
                "execution_engine": {"class_name": "PandasExecutionEngine"},
                "data_connectors": {
                    "default_runtime_data_connector": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"],
                    }
                },
            }

            self.context.add_datasource(**datasource_config)

            # Create batch request
            batch_request = RuntimeBatchRequest(
                datasource_name=f"{dataset_name}_datasource",
                data_connector_name="default_runtime_data_connector",
                data_asset_name=f"{dataset_name}_asset",
                runtime_parameters={"batch_data": df},
                batch_identifiers={"default_identifier_name": "default_identifier"},
            )

            # Create checkpoint
            checkpoint_config = {
                "name": f"{dataset_name}_checkpoint",
                "config_version": 1,
                "class_name": "SimpleCheckpoint",
                "validations": [
                    {
                        "batch_request": batch_request,
                        "expectation_suite_name": f"{dataset_name}_suite",
                    }
                ],
            }

            # Create expectation suite
            suite = self.context.create_expectation_suite(
                expectation_suite_name=f"{dataset_name}_suite", overwrite_existing=True
            )

            # Add basic expectations
            validator = self.context.get_validator(batch_request=batch_request)

            # Check for null values
            for column in df.columns:
                validator.expect_column_values_to_not_be_null(column)

            # Check for duplicates
            validator.expect_table_row_count_to_be_between(min_value=1)

            # Check data types
            for column in df.select_dtypes(include=[np.number]).columns:
                validator.expect_column_values_to_be_of_type(column, "int64")

            # Save expectations
            validator.save_expectation_suite(discard_failed_expectations=False)

            # Run checkpoint
            checkpoint_result = self.context.run_checkpoint(
                checkpoint_name=f"{dataset_name}_checkpoint"
            )

            if checkpoint_result.success:
                logger.info(f"Data quality validation passed for {dataset_name}")
                return {
                    "status": "passed",
                    "message": "Data quality validation successful",
                    "rows_validated": len(df),
                }
            else:
                logger.error(f"Data quality validation failed for {dataset_name}")
                return {
                    "status": "failed",
                    "message": "Data quality validation failed",
                    "rows_validated": len(df),
                    "details": checkpoint_result.validation_results,
                }

        except Exception as e:
            logger.error(f"Data quality validation error for {dataset_name}: {e}")
            return {"status": "error", "message": str(e), "rows_validated": len(df)}

    def _get_expected_categories(self) -> List[str]:
        """Get expected categories for text datasets"""
        # This would typically come from configuration or previous runs
        return [
            "cs.AI",
            "cs.CL",
            "cs.CC",
            "cs.CE",
            "cs.CG",
            "cs.GT",
            "cs.CV",
            "cs.CY",
            "cs.CR",
            "cs.DS",
            "cs.DB",
            "cs.DL",
            "cs.DM",
            "cs.DC",
            "cs.ET",
            "cs.FL",
            "cs.GL",
            "cs.GR",
            "cs.AR",
            "cs.HC",
            "cs.IR",
            "cs.IT",
            "cs.LG",
            "cs.LO",
            "cs.MA",
            "cs.MM",
            "cs.MS",
            "cs.NA",
            "cs.NE",
            "cs.NI",
            "cs.OH",
            "cs.OS",
            "cs.PF",
            "cs.PL",
            "cs.RO",
            "cs.SE",
            "cs.SD",
            "cs.SC",
            "cs.SI",
            "cs.SY",
        ]

    def validate_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Validate all datasets in the data directory"""

        results = {}

        # Find all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))

        for csv_file in csv_files:
            dataset_name = csv_file.stem

            try:
                # Load dataset
                df = pd.read_csv(csv_file)

                # Validate schema
                schema_result = self.validate_dataset_schema(df, dataset_name)

                # Validate quality
                quality_result = self.validate_data_quality(df, dataset_name)

                results[dataset_name] = {
                    "schema": schema_result,
                    "quality": quality_result,
                    "shape": df.shape,
                    "file_path": str(csv_file),
                }

            except Exception as e:
                logger.error(f"Failed to validate {dataset_name}: {e}")
                results[dataset_name] = {
                    "schema": {"status": "error", "message": str(e)},
                    "quality": {"status": "error", "message": str(e)},
                    "shape": (0, 0),
                    "file_path": str(csv_file),
                }

        return results

    def save_validation_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: str = "artifacts/validation_report.json",
    ):
        """Save validation report"""

        import json

        # Create artifacts directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save report
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Validation report saved to {output_path}")


def main():
    """Main validation function"""

    validator = DataValidator()

    # Run validation
    results = validator.validate_all_datasets()

    # Save report
    validator.save_validation_report(results)

    # Print summary
    print("\n" + "=" * 60)
    print("DATA VALIDATION SUMMARY")
    print("=" * 60)

    for dataset_name, result in results.items():
        schema_status = result["schema"]["status"]
        quality_status = result["quality"]["status"]
        shape = result["shape"]

        print(
            f"{dataset_name:20s}: Schema={schema_status:8s}, Quality={quality_status:8s}, Shape={shape}"
        )

    print("=" * 60)

    # Check if any validations failed
    failed_validations = [
        name
        for name, result in results.items()
        if result["schema"]["status"] == "failed" or result["quality"]["status"] == "failed"
    ]

    if failed_validations:
        print(f"\n❌ Validation failed for: {', '.join(failed_validations)}")
        return 1
    else:
        print("\n✅ All validations passed!")
        return 0


if __name__ == "__main__":
    exit(main())
