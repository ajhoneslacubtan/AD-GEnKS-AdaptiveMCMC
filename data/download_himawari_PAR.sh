#!/bin/bash
# This script downloads the Himawari-9 Short Wave Radiation/PAR v2.1
# 5km resolution NetCDF files at 10-minute intervals for April 4, 2025.
# It retrieves the 5km resolution files (with "RFL021" in their name) from the FTP server.

# FTP credentials and server details
FTP_SERVER="ftp.ptree.jaxa.jp"
FTP_USER="ajjhones.lacubtan_g.msuiit.edu.ph"
FTP_PASS="SP+wari8"

# Date settings for 2025-04-04
YEAR="2025"
MONTH="04"
DAY="04"
DATE="${YEAR}${MONTH}${DAY}"

# Base directory on the FTP server for Level 2 PAR products (10-minute resolution)
FTP_BASE_DIR="/pub/himawari/L2/PAR/021/${YEAR}${MONTH}/${DAY}"

# Local directory where files will be saved
LOCAL_DIR="./data/Himawari9_PAR_${DATE}"
mkdir -p "$LOCAL_DIR"

# Define the 10-minute intervals
minutes=("00" "10" "20" "30" "40" "50")

# Loop over each hour (00 to 23)
for hour in $(seq -w 0 23); do
    echo "Processing hour ${hour}..."
    # For each 10-minute interval within the hour
    for min in "${minutes[@]}"; do
        FILE="H09_${DATE}_${hour}${min}_RFL021_FLDK.02401_02401.nc"
        echo "Downloading ${FILE}..."
        lftp -u "$FTP_USER","$FTP_PASS" "$FTP_SERVER" <<EOF
cd ${FTP_BASE_DIR}/${hour}
get ${FILE} -o ${LOCAL_DIR}/${FILE}
bye
EOF
        # Check if the file was downloaded successfully
        if [ -f "${LOCAL_DIR}/${FILE}" ]; then
            echo "Successfully downloaded ${FILE}"
        else
            echo "Warning: ${FILE} not found or download failed."
        fi
    done
done

echo "All downloads completed. Files are stored in ${LOCAL_DIR}"
