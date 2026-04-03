#!/bin/bash

TARGET_DIR=${1:-"/home/disk/mercury3/nacc/aimip_subission/"}
LOG_FILE="cf_validation_report.log"
FAILED_FILES="failed_files.txt"

# Reset logs
echo "AIMIP-1 CF-Compliance Audit - $(date)" > "$LOG_FILE"
> "$FAILED_FILES"

# Activate the conda environment
source /home/disk/brume/nacc/anaconda3/etc/profile.d/conda.sh
conda activate dlesym-aimip

# MODERN CONDA CHECK: Try calling the module directly
if ! python -m cfchecker.cfchecks -h &> /dev/null; then
    echo "Error: cfchecker module not found in your current python environment."
    echo "Current Python: $(which python)"
    exit 1
fi

echo "Starting recursive check in: $TARGET_DIR"
echo "------------------------------------------"

files=$(find "$TARGET_DIR" -name "*.nc")
for file in $files; do
    echo "Checking: $(basename "$file")"
    
    # RUN VIA PYTHON MODULE
    result=$(python -m cfchecker.cfchecks -v 1.8 "$file" 2>&1)
    
    if echo "$result" | grep -E "ERROR|FATAL" > /dev/null; then
        echo "❌ FAILED: $file"
        echo "FAIL: $file" >> "$LOG_FILE"
        echo "$result" | grep -E "ERROR|FATAL" >> "$LOG_FILE"
        echo "------------------------------------------" >> "$LOG_FILE"
        echo "$file" >> "$FAILED_FILES"
    else
        echo "✅ PASSED"
    fi
done