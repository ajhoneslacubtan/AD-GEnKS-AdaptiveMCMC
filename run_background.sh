#!/bin/bash

# Usage: ./run_in_background.sh your_script.py [log_prefix]

# Check if the script name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <python_script> [log_prefix]"
  exit 1
fi

SCRIPT_NAME="$1"
LOG_PREFIX="${2:-script}"  # default log prefix if not provided

# Create logs directory if it doesn't exist
mkdir -p logs

# Get current timestamp for unique log file naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${LOG_PREFIX}_${TIMESTAMP}.log"
PID_FILE="logs/${LOG_PREFIX}_${TIMESTAMP}.pid"

# Run the Python script in the background with nohup
nohup python "$SCRIPT_NAME" > "$LOG_FILE" 2>&1 &

# Save the process ID to a file for later reference
echo $! > "$PID_FILE"

echo "Started $SCRIPT_NAME in background with PID: $!"
echo "Logs are being written to: $LOG_FILE"
