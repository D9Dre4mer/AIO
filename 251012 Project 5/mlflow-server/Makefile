# MLflow Server Makefile
# Provides convenient commands for managing MLflow server

.PHONY: help install start stop restart status logs clean test configure

# Default target
help:
	@echo "MLflow Server Management Commands:"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make start       - Start MLflow server"
	@echo "  make stop        - Stop MLflow server"
	@echo "  make restart     - Restart MLflow server"
	@echo "  make status      - Check server status"
	@echo "  make logs        - Show server logs"
	@echo "  make clean       - Clean up logs and temp files"
	@echo "  make test        - Test server connection"
	@echo "  make configure   - Configure MLflow client"
	@echo "  make docker      - Start with Docker Compose"
	@echo "  make docker-prod - Start production setup"
	@echo ""

# Install dependencies
install:
	@echo "Installing MLflow dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Start MLflow server
start:
	@echo "Starting MLflow server..."
	@if [ -f start_server.sh ]; then \
		chmod +x start_server.sh && ./start_server.sh; \
	elif [ -f start_server.bat ]; then \
		start_server.bat; \
	else \
		echo "No startup script found!"; \
	fi

# Stop MLflow server
stop:
	@echo "Stopping MLflow server..."
	@if command -v lsof >/dev/null 2>&1; then \
		lsof -ti:5000 | xargs kill -9 2>/dev/null || true; \
	elif command -v netstat >/dev/null 2>&1; then \
		for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do taskkill /PID %%a /F >nul 2>&1; \
	fi
	@echo "MLflow server stopped"

# Restart MLflow server
restart: stop start

# Check server status
status:
	@echo "Checking MLflow server status..."
	@if curl -f http://localhost:5000/health >/dev/null 2>&1; then \
		echo "✅ MLflow server is running on http://localhost:5000"; \
	else \
		echo "❌ MLflow server is not running"; \
	fi

# Show server logs
logs:
	@echo "Showing MLflow server logs..."
	@if [ -f logs/access.log ]; then \
		echo "=== Access Logs ==="; \
		tail -f logs/access.log; \
	elif [ -f logs/error.log ]; then \
		echo "=== Error Logs ==="; \
		tail -f logs/error.log; \
	else \
		echo "No log files found"; \
	fi

# Clean up logs and temp files
clean:
	@echo "Cleaning up logs and temp files..."
	@if [ -d logs ]; then rm -rf logs/*; fi
	@if [ -d __pycache__ ]; then rm -rf __pycache__; fi
	@if [ -f *.pyc ]; then rm -f *.pyc; fi
	@echo "Cleanup completed"

# Test server connection
test:
	@echo "Testing MLflow server connection..."
	python configure_client.py

# Configure MLflow client
configure:
	@echo "Configuring MLflow client..."
	python configure_client.py

# Start with Docker Compose
docker:
	@echo "Starting MLflow server with Docker Compose..."
	docker-compose up -d
	@echo "MLflow server started with Docker"

# Start production setup
docker-prod:
	@echo "Starting MLflow server in production mode..."
	docker-compose --profile production up -d
	@echo "MLflow server started in production mode"

# Health check
health:
	@echo "Running health check..."
	python health_check.py

# Backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@if [ -f mlflow.db ]; then cp mlflow.db backups/mlflow_backup_$$(date +%Y%m%d_%H%M%S).db; fi
	@if [ -d mlruns ]; then tar -czf backups/mlruns_backup_$$(date +%Y%m%d_%H%M%S).tar.gz mlruns/; fi
	@echo "Backup completed"

# Restore from backup
restore:
	@echo "Available backups:"
	@ls -la backups/ 2>/dev/null || echo "No backups found"
	@echo "To restore, manually copy backup files to current directory"

# Setup monitoring
monitoring:
	@echo "Setting up monitoring..."
	docker-compose --profile monitoring up -d
	@echo "Monitoring setup completed"
	@echo "Prometheus UI: http://localhost:9090"
	@echo "MLflow UI: http://localhost:5000"
