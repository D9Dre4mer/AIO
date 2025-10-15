#!/usr/bin/env python3
"""
MLflow Server Health Check Script
Monitors MLflow server health and sends alerts if needed
"""

import requests
import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from datetime import datetime

# Configuration
MLFLOW_URL = os.getenv('MLFLOW_URL', 'http://localhost:5000')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))  # seconds
ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/health_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLflowHealthChecker:
    def __init__(self):
        self.failure_count = 0
        self.last_alert_time = None
        self.alert_cooldown = 300  # 5 minutes
        
    def check_health(self):
        """Check MLflow server health"""
        try:
            response = requests.get(f"{MLFLOW_URL}/health", timeout=10)
            if response.status_code == 200:
                logger.info("MLflow server is healthy")
                self.failure_count = 0
                return True
            else:
                logger.warning(f"MLflow server returned status code: {response.status_code}")
                self.failure_count += 1
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to MLflow server: {e}")
            self.failure_count += 1
            return False
    
    def check_experiments(self):
        """Check if experiments API is working"""
        try:
            response = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search", timeout=10)
            if response.status_code == 200:
                logger.info("MLflow experiments API is working")
                return True
            else:
                logger.warning(f"MLflow experiments API returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to MLflow experiments API: {e}")
            return False
    
    def send_alert(self, message):
        """Send alert email"""
        if not ALERT_EMAIL or not SMTP_USERNAME or not SMTP_PASSWORD:
            logger.warning("Email configuration not set, skipping alert")
            return
            
        current_time = time.time()
        if self.last_alert_time and (current_time - self.last_alert_time) < self.alert_cooldown:
            logger.info("Alert cooldown active, skipping alert")
            return
            
        try:
            msg = MIMEMultipart()
            msg['From'] = SMTP_USERNAME
            msg['To'] = ALERT_EMAIL
            msg['Subject'] = "MLflow Server Alert"
            
            body = f"""
            MLflow Server Alert
            
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Message: {message}
            
            Please check the MLflow server immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            text = msg.as_string()
            server.sendmail(SMTP_USERNAME, ALERT_EMAIL, text)
            server.quit()
            
            logger.info("Alert email sent successfully")
            self.last_alert_time = current_time
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
    
    def run(self):
        """Main health check loop"""
        logger.info("Starting MLflow health check monitor")
        
        while True:
            try:
                health_ok = self.check_health()
                experiments_ok = self.check_experiments()
                
                if not health_ok or not experiments_ok:
                    if self.failure_count >= 3:
                        self.send_alert(f"MLflow server has been down for {self.failure_count} consecutive checks")
                else:
                    logger.info("All checks passed")
                
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Health check monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in health check: {e}")
                time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    checker = MLflowHealthChecker()
    checker.run()
