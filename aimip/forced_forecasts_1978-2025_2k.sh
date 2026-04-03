#!/bin/bash

# This script is used to run 5 forced forecasts the aimip historical ensemble

################ Environment Params ################
# Path to DLESyM module
MODULE_DIR="/home/disk/mercury2/nacc/AIMIP2026/DLESyM"
# set to -1 to use CPU
DEVICE_NUMBERS="2"

################ Forecast Params ################

# Destination directory for forecast files
# OUTPUT_DIR="/home/disk/mercury2/nacc/forecasts/aimip"
OUTPUT_DIR="/home/disk/mercury3/nacc/forecasts/aimip"
# Output logs sent here
OUTPUT_FILE="${OUTPUT_DIR}/forced_forecasts_1978-2025_2k.out"
# Path to models 
ATMOS_MODEL="${MODULE_DIR}/models/dlwp"
OCEAN_MODEL="${MODULE_DIR}/models/ocean-forcing-model_2k"
# Sufffixes for output filnames. We're forecasting for each month, so we have 12 suffixes. 
# ATMOS_OUTPUT_FILENAME_SUFFIXES=("01" "02" "03" "04" "05")
# OCEAN_OUTPUT_FILENAME_SUFFIXES=("01" "02" "03" "04" "05")
ATMOS_OUTPUT_FILENAME_SUFFIXES=("05")
OCEAN_OUTPUT_FILENAME_SUFFIXES=("05")
# INIT_STARTS=("1978-10-03" "1978-10-04" "1978-10-05" "1978-10-06" "1978-10-07")
# INIT_ENDS=("1978-10-03" "1978-10-04" "1978-10-05" "1978-10-06" "1978-10-07")
INIT_STARTS=("1978-10-07")
INIT_ENDS=("1978-10-07")
# Path to directory with atmos initialization data.
ATMOS_DATA_DIR="/home/disk/mercury2/nacc/AIMIP2026/init_data"
ATMOS_DATASET_NAME="aimip_1978-init"
# ocean initialization data. This is a dummy field for the 
# forced ocean model. Output is always taken from forcing dataset
OCEAN_DATA_DIR="/home/disk/mercury2/nacc/AIMIP2026/init_data"
OCEAN_DATASET_NAME="aimip_1978-init_ocean"
# 1978-2025 in hours, using same lead time for each of the 5 inits.
# all end between 12-19-2024 and 12-28-2024
LEAD_TIME="405312"

# Parameters for resolving intialization dates
FREQ="D"
TIME_CHUNK=1
STEP_CHUNK=1000

# This is where intermediate data is stored, make sure it's big enough. 
# Each 100 year forecast takes ~243GB
CACHE_DIR="${OUTPUT_DIR}/2k_cache/"



#############################################################
############ Boiler plate to execute forecast ###############
#############################################################

cd ${MODULE_DIR}
export PYTHONPATH=${MODULE_DIR}

source /home/disk/brume/nacc/anaconda3/etc/profile.d/conda.sh
conda activate dlesym-aimip

LEN=${#INIT_STARTS[@]}

# delete the output file if it exists
if [ -f ${OUTPUT_FILE} ]; then
    rm ${OUTPUT_FILE}
fi

for (( i=0; i<${LEN}; i++ ));
do
    RUN_CMD="python scripts/coupled_forecast_hdf5.py \
        --atmos-model-path ${ATMOS_MODEL} \
        --ocean-model-path ${OCEAN_MODEL} \
        --non-strict \
        --lead-time ${LEAD_TIME} \
        --forecast-init-start ${INIT_STARTS[$i]} \
        --forecast-init-end ${INIT_ENDS[$i]} \
        --freq ${FREQ} \
        --time-chunk ${TIME_CHUNK} \
        --step-chunk ${STEP_CHUNK} \
        --cache-dir ${CACHE_DIR} \
        --output-directory ${OUTPUT_DIR} \
        --atmos-output-filename atmos_aimip_forced_forecast_1978-2025_2k_n${ATMOS_OUTPUT_FILENAME_SUFFIXES[$i]} \
        --ocean-output-filename ocean_aimip_forcing_1978-2025_2k_n${OCEAN_OUTPUT_FILENAME_SUFFIXES[$i]} \
        --atmos-data-directory ${ATMOS_DATA_DIR} \
        --ocean-data-directory ${OCEAN_DATA_DIR} \
        --atmos-dataset-name ${ATMOS_DATASET_NAME} \
        --ocean-dataset-name ${OCEAN_DATASET_NAME} \
        --gpu ${DEVICE_NUMBERS}"

    # Set environment variables and run the command
    export HDF5_USE_FILE_LOCKING=False
    export HYDRA_FULL_ERROR=1 

    # if output file is given, redirect output
    if [[ -n "${OUTPUT_FILE}" ]]; then
        ${RUN_CMD} &>> ${OUTPUT_FILE}
    else
        ${RUN_CMD}
    fi
done

