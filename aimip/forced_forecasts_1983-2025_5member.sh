#!/bin/bash

# This script is used to run 5 forced forecasts the aimip historical ensemble

################ Environment Params ################
# Path to DLESyM module
MODULE_DIR="/home/disk/mercury2/nacc/AIMIP2026/DLESyM"
# set to -1 to use CPU
DEVICE_NUMBERS="0"

################ Forecast Params ################

# Destination directory for forecast files
OUTPUT_DIR="/home/disk/mercury2/nacc/forecasts/aimip"
# Output logs sent here
OUTPUT_FILE="${OUTPUT_DIR}/forced_forecasts_1983-2025_5member.out"
# Path to models 
ATMOS_MODEL="${MODULE_DIR}/models/dlwp"
OCEAN_MODEL="${MODULE_DIR}/models/ocean-forcing-model"
# Sufffixes for output filnames. We're forecasting for each month, so we have 12 suffixes. 
ATMOS_OUTPUT_FILENAME_SUFFIXES=("01" "02" "03" "04" "05")
OCEAN_OUTPUT_FILENAME_SUFFIXES=("01" "02" "03" "04" "05")
INIT_STARTS=("1983-10-01" "1983-10-02" "1983-10-03" "1983-10-04" "1983-10-05")  
INIT_ENDS=("1983-10-01" "1983-10-02" "1983-10-03" "1983-10-04" "1983-10-05")
# Path to directory with atmos initialization data.
ATMOS_DATA_DIR="/home/disk/rhodium/WEB/DLESyM_AGU-Advances"
ATMOS_DATASET_NAME="hpx64_1983-2017_3h_9varCoupledAtmos-sst"
# ocean initialization data. This is a dummy field for the 
# forced ocean model. Output is always taken from forcing dataset
OCEAN_DATA_DIR="/home/disk/rhodium/WEB/DLESyM_AGU-Advances"
OCEAN_DATASET_NAME="hpx64_1983-2017_3h_1varCoupledOcean-z1000-ws10-olr"
# 1983-2025 in hours, using same lead time for each of the 5 inits.
# all end between 12-19-2024 and 12-29-2024
LEAD_TIME="361440"

# Parameters for resolving intialization dates
FREQ="D"
TIME_CHUNK=1
STEP_CHUNK=1000

# This is where intermediate data is stored, make sure it's big enough. 
# Each 100 year forecast takes ~243GB
CACHE_DIR="${OUTPUT_DIR}/"



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
        --atmos-output-filename atmos_aimip_forced_forecast_1983-2025_n${ATMOS_OUTPUT_FILENAME_SUFFIXES[$i]} \
        --ocean-output-filename ocean_aimip_forcing_1983-2025_n${OCEAN_OUTPUT_FILENAME_SUFFIXES[$i]} \
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