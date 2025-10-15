"""
Evidently drift monitoring and reporting
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestNumberOfColumns, TestNumberOfRows
from evidently.ui.workspace import Workspace, WorkspaceBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """Evidently drift monitoring"""

    def __init__(self, workspace_path: str = "evidently_workspace"):
        self.workspace_path = workspace_path
        self.workspace = self._setup_workspace()

    def _setup_workspace(self) -> Workspace:
        """Setup Evidently workspace"""
        try:
            workspace = Workspace.create(workspace_path=self.workspace_path)
            return workspace
        except Exception as e:
            logger.warning(f"Could not setup Evidently workspace: {e}")
            return None

    def create_column_mapping(
        self, feature_columns: List[str], target_column: str = None
    ) -> ColumnMapping:
        """Create column mapping for Evidently"""
        return ColumnMapping(
            target=target_column,
            numerical_features=feature_columns,
            categorical_features=[],
            prediction="prediction" if target_column else None,
        )

    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = None,
        report_name: str = "drift_report",
    ) -> Dict[str, Any]:
        """Generate drift report using Evidently"""

        logger.info(f"Generating drift report: {report_name}")

        try:
            # Create column mapping
            column_mapping = self.create_column_mapping(feature_columns, target_column)

            # Create report
            report = Report(
                metrics=[
                    DataDriftPreset(),
                    DataQualityPreset(),
                    TargetDriftPreset() if target_column else None,
                ]
            )

            # Generate report
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

            # Save report
            report_path = f"artifacts/{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            os.makedirs("artifacts", exist_ok=True)
            report.save_html(report_path)

            # Extract metrics
            metrics = self._extract_metrics_from_report(report)

            # Save metrics to JSON
            metrics_path = f"artifacts/{report_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Drift report saved to {report_path}")

            return {
                "status": "success",
                "report_path": report_path,
                "metrics_path": metrics_path,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to generate drift report: {e}")
            return {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}

    def _extract_metrics_from_report(self, report: Report) -> Dict[str, Any]:
        """Extract metrics from Evidently report"""
        metrics = {}

        try:
            # Get report data
            report_data = report.as_dict()

            # Extract data drift metrics
            if "data_drift" in report_data:
                drift_metrics = report_data["data_drift"]
                metrics["data_drift"] = {
                    "drift_score": drift_metrics.get("drift_score", 0),
                    "drift_detected": drift_metrics.get("drift_detected", False),
                    "number_of_drifted_columns": drift_metrics.get("number_of_drifted_columns", 0),
                }

            # Extract data quality metrics
            if "data_quality" in report_data:
                quality_metrics = report_data["data_quality"]
                metrics["data_quality"] = {
                    "number_of_rows": quality_metrics.get("number_of_rows", 0),
                    "number_of_columns": quality_metrics.get("number_of_columns", 0),
                    "missing_values": quality_metrics.get("missing_values", {}),
                    "infinite_values": quality_metrics.get("infinite_values", {}),
                }

            # Extract target drift metrics
            if "target_drift" in report_data:
                target_metrics = report_data["target_drift"]
                metrics["target_drift"] = {
                    "drift_score": target_metrics.get("drift_score", 0),
                    "drift_detected": target_metrics.get("drift_detected", False),
                }

        except Exception as e:
            logger.warning(f"Could not extract metrics from report: {e}")
            metrics["extraction_error"] = str(e)

        return metrics

    def run_drift_tests(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = None,
    ) -> Dict[str, Any]:
        """Run drift tests using Evidently"""

        logger.info("Running drift tests...")

        try:
            # Create column mapping
            column_mapping = self.create_column_mapping(feature_columns, target_column)

            # Create test suite
            test_suite = TestSuite(
                tests=[
                    TestNumberOfColumns(),
                    TestNumberOfRows(),
                ]
                + [TestColumnDrift(column_name=col) for col in feature_columns[:5]]
            )  # Test first 5 columns

            # Run tests
            test_suite.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

            # Save test results
            test_path = f"artifacts/drift_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            test_suite.save_html(test_path)

            # Extract test results
            test_results = self._extract_test_results(test_suite)

            logger.info(
                f"Drift tests completed: {test_results['passed']}/{test_results['total']} passed"
            )

            return {
                "status": "success",
                "test_path": test_path,
                "test_results": test_results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to run drift tests: {e}")
            return {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}

    def _extract_test_results(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Extract test results from test suite"""
        try:
            test_data = test_suite.as_dict()

            total_tests = 0
            passed_tests = 0
            failed_tests = 0

            if "tests" in test_data:
                for test in test_data["tests"]:
                    total_tests += 1
                    if test.get("status") == "SUCCESS":
                        passed_tests += 1
                    else:
                        failed_tests += 1

            return {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            }

        except Exception as e:
            logger.warning(f"Could not extract test results: {e}")
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0, "error": str(e)}

    def monitor_production_data(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = None,
        drift_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Monitor production data for drift"""

        logger.info("Monitoring production data for drift...")

        # Generate drift report
        report_result = self.generate_drift_report(
            reference_data, current_data, feature_columns, target_column, "production_drift"
        )

        if report_result["status"] != "success":
            return report_result

        # Run drift tests
        test_result = self.run_drift_tests(
            reference_data, current_data, feature_columns, target_column
        )

        # Analyze results
        metrics = report_result["metrics"]
        drift_detected = False
        drift_score = 0

        if "data_drift" in metrics:
            drift_score = metrics["data_drift"].get("drift_score", 0)
            drift_detected = metrics["data_drift"].get("drift_detected", False)

        # Determine alert level
        alert_level = "none"
        if drift_score > drift_threshold:
            alert_level = "high"
        elif drift_score > drift_threshold * 0.7:
            alert_level = "medium"
        elif drift_detected:
            alert_level = "low"

        # Create monitoring summary
        monitoring_summary = {
            "status": "completed",
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "alert_level": alert_level,
            "threshold": drift_threshold,
            "report_result": report_result,
            "test_result": test_result,
            "timestamp": datetime.now().isoformat(),
        }

        # Save monitoring summary
        summary_path = "artifacts/monitoring_summary.json"
        with open(summary_path, "w") as f:
            json.dump(monitoring_summary, f, indent=2)

        logger.info(
            f"Monitoring completed: drift_score={drift_score:.3f}, alert_level={alert_level}"
        )

        return monitoring_summary

    def schedule_drift_monitoring(
        self,
        reference_data_path: str,
        current_data_path: str,
        feature_columns: List[str],
        target_column: str = None,
        schedule_interval: str = "daily",
    ) -> Dict[str, Any]:
        """Schedule drift monitoring (placeholder for actual scheduling)"""

        logger.info(f"Scheduling drift monitoring: {schedule_interval}")

        try:
            # Load data
            reference_data = pd.read_csv(reference_data_path)
            current_data = pd.read_csv(current_data_path)

            # Run monitoring
            result = self.monitor_production_data(
                reference_data, current_data, feature_columns, target_column
            )

            # Schedule next run (placeholder)
            next_run = datetime.now() + timedelta(days=1 if schedule_interval == "daily" else 7)

            return {
                "status": "scheduled",
                "schedule_interval": schedule_interval,
                "next_run": next_run.isoformat(),
                "monitoring_result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to schedule drift monitoring: {e}")
            return {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}


def main():
    """Main drift monitoring function"""

    # Initialize drift monitor
    monitor = DriftMonitor()

    # Create sample data for testing
    np.random.seed(42)

    # Reference data (training data)
    reference_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.normal(0, 1, 1000),
            "feature_3": np.random.normal(0, 1, 1000),
            "target": np.random.randint(0, 2, 1000),
        }
    )

    # Current data (production data) - with some drift
    current_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.2, 1.1, 500),  # Slight drift
            "feature_2": np.random.normal(0, 1, 500),
            "feature_3": np.random.normal(0, 1, 500),
            "target": np.random.randint(0, 2, 500),
        }
    )

    feature_columns = ["feature_1", "feature_2", "feature_3"]

    # Run monitoring
    result = monitor.monitor_production_data(
        reference_data, current_data, feature_columns, "target"
    )

    print("\n" + "=" * 60)
    print("DRIFT MONITORING SUMMARY")
    print("=" * 60)
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Drift Score: {result['drift_score']:.3f}")
    print(f"Alert Level: {result['alert_level']}")
    print(f"Threshold: {result['threshold']}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
