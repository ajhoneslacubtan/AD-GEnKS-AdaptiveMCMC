#!/bin/bash

# Path to the expected output file
FILE="inference_data_lag3.nc"

# Wait until the file exists and is non-empty
while [ ! -s "$FILE" ]; do
    echo "Waiting for $FILE..."
    sleep 5
done

echo "$FILE is ready. Running your script..."
# Run your script here
bash run_background.sh
