"""
Prometheus metrics and Grafana dashboard configuration
"""

import json
import os
from datetime import datetime
from typing import Any, Dict


def create_grafana_dashboard_config() -> Dict[str, Any]:
    """Create Grafana dashboard configuration for ML monitoring"""

    dashboard = {
        "dashboard": {
            "id": None,
            "title": "ML Model Monitoring Dashboard",
            "tags": ["ml", "monitoring", "prometheus"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(http_requests_total[5m])",
                            "legendFormat": "{{method}} {{endpoint}}",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "reqps", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                },
                {
                    "id": 2,
                    "title": "Response Time",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p95",
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p50",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "s", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                },
                {
                    "id": 3,
                    "title": "Prediction Rate",
                    "type": "stat",
                    "targets": [
                        {"expr": "rate(predictions_total[5m])", "legendFormat": "Predictions/sec"}
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "reqps", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                },
                {
                    "id": 4,
                    "title": "Prediction Duration",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m]))",
                            "legendFormat": "p95",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "s", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                },
                {
                    "id": 5,
                    "title": "Request Rate Over Time",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "rate(http_requests_total[5m])",
                            "legendFormat": "{{method}} {{endpoint}}",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "reqps", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                },
                {
                    "id": 6,
                    "title": "Response Time Over Time",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p95",
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p50",
                        },
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "s", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24},
                },
                {
                    "id": 7,
                    "title": "Error Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100',
                            "legendFormat": "Error Rate %",
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "percent", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 32},
                },
                {
                    "id": 8,
                    "title": "Model Performance",
                    "type": "stat",
                    "targets": [
                        {"expr": "mlflow_model_accuracy", "legendFormat": "Accuracy"},
                        {"expr": "mlflow_model_f1_score", "legendFormat": "F1 Score"},
                    ],
                    "fieldConfig": {
                        "defaults": {"unit": "short", "color": {"mode": "palette-classic"}}
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 32},
                },
            ],
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "30s",
            "schemaVersion": 30,
            "version": 1,
            "uid": "ml-monitoring-dashboard",
        }
    }

    return dashboard


def create_prometheus_config() -> Dict[str, Any]:
    """Create Prometheus configuration"""

    config = {
        "global": {"scrape_interval": "15s", "evaluation_interval": "15s"},
        "rule_files": ["rules/*.yml"],
        "scrape_configs": [
            {
                "job_name": "ml-api",
                "static_configs": [{"targets": ["localhost:8000"]}],
                "metrics_path": "/metrics",
                "scrape_interval": "5s",
            },
            {
                "job_name": "mlflow",
                "static_configs": [{"targets": ["localhost:5000"]}],
                "metrics_path": "/metrics",
                "scrape_interval": "30s",
            },
        ],
        "alerting": {"alertmanagers": [{"static_configs": [{"targets": ["localhost:9093"]}]}]},
    }

    return config


def create_alerting_rules() -> Dict[str, Any]:
    """Create Prometheus alerting rules"""

    rules = {
        "groups": [
            {
                "name": "ml-api-alerts",
                "rules": [
                    {
                        "alert": "HighErrorRate",
                        "expr": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05',
                        "for": "2m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High error rate detected",
                            "description": "Error rate is {{ $value | humanizePercentage }}",
                        },
                    },
                    {
                        "alert": "HighResponseTime",
                        "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1",
                        "for": "2m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High response time detected",
                            "description": "95th percentile response time is {{ $value }}s",
                        },
                    },
                    {
                        "alert": "LowPredictionRate",
                        "expr": "rate(predictions_total[5m]) < 0.1",
                        "for": "5m",
                        "labels": {"severity": "info"},
                        "annotations": {
                            "summary": "Low prediction rate",
                            "description": "Prediction rate is {{ $value }} predictions/sec",
                        },
                    },
                    {
                        "alert": "ModelDriftDetected",
                        "expr": "evidently_drift_score > 0.5",
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Model drift detected",
                            "description": "Drift score is {{ $value }}",
                        },
                    },
                ],
            }
        ]
    }

    return rules


def create_grafana_datasource_config() -> Dict[str, Any]:
    """Create Grafana datasource configuration"""

    datasource = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://localhost:9090",
        "access": "proxy",
        "isDefault": True,
        "jsonData": {"httpMethod": "POST"},
    }

    return datasource


def save_configurations():
    """Save all monitoring configurations"""

    # Create directories
    os.makedirs("monitoring/grafana/dashboards", exist_ok=True)
    os.makedirs("monitoring/prometheus/rules", exist_ok=True)
    os.makedirs("monitoring/grafana/datasources", exist_ok=True)

    # Save Grafana dashboard
    dashboard_config = create_grafana_dashboard_config()
    with open("monitoring/grafana/dashboards/ml-monitoring-dashboard.json", "w") as f:
        json.dump(dashboard_config, f, indent=2)

    # Save Prometheus config
    prometheus_config = create_prometheus_config()
    with open("monitoring/prometheus/prometheus.yml", "w") as f:
        import yaml

        yaml.dump(prometheus_config, f, default_flow_style=False)

    # Save alerting rules
    alerting_rules = create_alerting_rules()
    with open("monitoring/prometheus/rules/ml-alerts.yml", "w") as f:
        yaml.dump(alerting_rules, f, default_flow_style=False)

    # Save Grafana datasource
    datasource_config = create_grafana_datasource_config()
    with open("monitoring/grafana/datasources/prometheus.json", "w") as f:
        json.dump(datasource_config, f, indent=2)

    print("Monitoring configurations saved successfully!")


def create_docker_compose_monitoring():
    """Create Docker Compose for monitoring stack"""

    docker_compose = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/rules:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/datasources:/var/lib/grafana/datasources
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    depends_on:
      - prometheus
"""

    with open("monitoring/docker-compose.monitoring.yml", "w") as f:
        f.write(docker_compose)

    print("Docker Compose for monitoring stack created!")


if __name__ == "__main__":
    save_configurations()
    create_docker_compose_monitoring()
