#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for unique log file naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/sample_${TIMESTAMP}.log"

# Run the Python script in the background with nohup
# Redirect both stdout and stderr to the log file
# The & at the end runs it in the background
nohup python run_real_data.py > "${LOG_FILE}" 2>&1 &

# Save the process ID to a file for later reference
echo $! > logs/sample_${TIMESTAMP}.pid

echo "Started sample.py in background with PID: $!"
echo "Logs are being written to: ${LOG_FILE}" 