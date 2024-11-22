#!/bin/bash

# This example presents a shell script for creating twelve 100 year forecasts, 
# initialized on the first day of each month. This script was configured to use
# a CUDA GPU as configured on our machines at the University of Washington.

################ Environment Params ################
MODULE_DIR="path/to/DLESyM"
# set to -1 to use CPU
DEVICE_NUMBERS="0"

################ Forecast Params ################

# Destination directory for forecast files
OUTPUT_DIR="path/to/output_dir"
# Output logs sent here
OUTPUT_FILE="${OUTPUT_DIR}/100yr_forecasts.out"
# Path to models 
ATMOS_MODEL="${MODULE_DIR}/models/dlwp"
OCEAN_MODEL="${MODULE_DIR}/models/dlom"
# Sufffixes for output filnames. We're forecasting for each month, so we have 12 suffixes. 
ATMOS_OUTPUT_FILENAME_SUFFIXES=("JanInit" "FebInit" "MarInit" "AprInit" "MayInit" "JunInit" "JulInit" "AugInit" "SepInit" "OctInit" "NovInit" "DecInit")
OCEAN_OUTPUT_FILENAME_SUFFIXES=("JanInit" "FebInit" "MarInit" "AprInit" "MayInit" "JunInit" "JulInit" "AugInit" "SepInit" "OctInit" "NovInit" "DecInit")
# corresponding initialization dates for each suffix
INIT_STARTS=("2017-01-01" "2017-02-01" "2017-03-01" "2017-04-01" "2017-05-01" "2017-06-01" "2016-07-01" "2016-08-01" "2016-09-01" "2016-10-01" "2016-11-01" "2016-12-01")
INIT_ENDS=("2017-01-02" "2017-02-02" "2017-03-02" "2017-04-02" "2017-05-02" "2017-06-02" "2016-07-02" "2016-08-02" "2016-09-02" "2016-10-02" "2016-11-02" "2016-12-02")
# Path to directory with initialization data. Initializaion for example forecasts are included in repo
DATA_DIR="${MODULE_DIR}/data"
# Name of dataset
ATMOS_DATASET_NAME="hpx64_9varCoupledAtmos-sst"
OCEAN_DATASET_NAME="hpx64_1varCoupledOcean-z1000-ws10-olr"
# 100 years in hours
LEAD_TIME="876672"

# Parameters for resolving intialization dates
FREQ="MS"
TIME_CHUNK=1
STEP_CHUNK=1000

# This is where intermediate data is stored, make sure it's big enough. 
# Each 100 year forecast takes ~243GB
CACHE_DIR="./"



#############################################################
############ Boiler plate to execute forecast ###############
#############################################################

cd ${MODULE_DIR}
export PYTHONPATH=${MODULE_DIR}

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
        --atmos-output-filename atmos_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-124_double_restart_100year${ATMOS_OUTPUT_FILENAME_SUFFIXES[$i]} \
        --ocean-output-filename ocean_hpx64_coupled-dlwp-olr_seed0+hpx64_coupled-dlom-olr_unet_dil-124_double_restart_100year${OCEAN_OUTPUT_FILENAME_SUFFIXES[$i]} \
        --data-directory ${DATA_DIR} \
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