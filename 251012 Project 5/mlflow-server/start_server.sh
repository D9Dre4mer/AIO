#!/bin/bash
# MLflow Server Startup Script for Production

set -e

# Configuration
MLFLOW_HOST=${MLFLOW_HOST:-"0.0.0.0"}
MLFLOW_PORT=${MLFLOW_PORT:-"5000"}
MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI:-"sqlite:///mlflow.db"}
MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-"./mlruns"}
MLFLOW_WORKERS=${MLFLOW_WORKERS:-"4"}
MLFLOW_TIMEOUT=${MLFLOW_TIMEOUT:-"120"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    MLflow Server Startup Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if MLflow is installed
check_mlflow() {
    print_status "Checking MLflow installation..."
    if ! command -v mlflow &> /dev/null; then
        print_error "MLflow is not installed!"
        print_status "Installing MLflow..."
        pip install -r requirements.txt
    else
        print_status "MLflow is installed: $(mlflow --version)"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p mlruns
    mkdir -p logs
    mkdir -p config
    print_status "Directories created successfully"
}

# Check if port is available
check_port() {
    print_status "Checking if port $MLFLOW_PORT is available..."
    if lsof -Pi :$MLFLOW_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $MLFLOW_PORT is already in use!"
        print_status "Attempting to kill existing process..."
        lsof -ti:$MLFLOW_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    print_status "Port $MLFLOW_PORT is available"
}

# Start MLflow server
start_server() {
    print_status "Starting MLflow server..."
    print_status "Configuration:"
    echo "  Host: $MLFLOW_HOST"
    echo "  Port: $MLFLOW_PORT"
    echo "  Backend Store: $MLFLOW_BACKEND_STORE_URI"
    echo "  Artifact Root: $MLFLOW_ARTIFACT_ROOT"
    echo "  Workers: $MLFLOW_WORKERS"
    echo "  Timeout: $MLFLOW_TIMEOUT"
    
    # Start server with gunicorn for production
    exec gunicorn \
        --bind $MLFLOW_HOST:$MLFLOW_PORT \
        --workers $MLFLOW_WORKERS \
        --timeout $MLFLOW_TIMEOUT \
        --access-logfile logs/access.log \
        --error-logfile logs/error.log \
        --log-level info \
        --preload \
        mlflow.server:app
}

# Health check
health_check() {
    print_status "Performing health check..."
    sleep 5
    
    if curl -f http://localhost:$MLFLOW_PORT/health >/dev/null 2>&1; then
        print_status "MLflow server is running successfully!"
        print_status "MLflow UI: http://localhost:$MLFLOW_PORT"
        print_status "API Endpoint: http://localhost:$MLFLOW_PORT/api/2.0/mlflow"
    else
        print_error "Health check failed!"
        exit 1
    fi
}

# Main execution
main() {
    check_mlflow
    create_directories
    check_port
    
    if [ "$1" = "--daemon" ]; then
        print_status "Starting MLflow server in daemon mode..."
        start_server &
        health_check
    else
        print_status "Starting MLflow server in foreground..."
        start_server
    fi
}

# Handle signals
trap 'print_status "Shutting down MLflow server..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"
